import { MAX_BUFFERED_FRAMES } from "../../core/minimal.js";
import { global_dispatcher } from "../../core/dispatcher.js";
import { Renderer } from "../../renderer/renderer.js";
import { TextureArrayPools } from "../../renderer/texture_pool.js";
import { TextureManager } from "../../renderer/texture_manager.js";
import { JobSystem, JobStatus } from "../../utility/job_system.js";
import { Name } from "../../utility/names.js";
import { StreamProvider, StreamUpdateStatus } from "../stream_provider.js";
import { StreamingSystem } from "../streaming_system.js";
import { deserialize_json, serialize_json } from "../streaming_io.js";

export const texture_stream_provider_type = "texture";

const texture_stream_format_version = 1;
const default_uploads_per_frame = 8;
const default_pixels_per_frame = 4 * 1024 * 1024;
const texture_stream_requests = new WeakMap();

const TextureLoadPhase = Object.freeze({
  WAITING: 0,
  UPLOADING: 1,
});

export function bitmap_mip_chains_match_config(mip_chains, width, height, mip_levels) {
  if (!Array.isArray(mip_chains) || mip_chains.length === 0) {
    return false;
  }

  for (const mip_chain of mip_chains) {
    if (!Array.isArray(mip_chain) || mip_chain.length !== mip_levels) {
      return false;
    }

    for (let level = 0; level < mip_chain.length; level++) {
      const mip_bitmap = mip_chain[level];
      if (!mip_bitmap) {
        return false;
      }

      const expected_width = Math.max(1, width >> level);
      const expected_height = Math.max(1, height >> level);
      if (mip_bitmap.width !== expected_width || mip_bitmap.height !== expected_height) {
        return false;
      }
    }
  }

  return true;
}

export function close_bitmap_mip_chains(mip_chains) {
  for (const mip_chain of mip_chains ?? []) {
    if (!Array.isArray(mip_chain)) {
      continue;
    }
    for (const mip_bitmap of mip_chain) {
      mip_bitmap?.close?.();
    }
  }
}

function create_local_load_handle(load_operation) {
  const handle = {
    status: JobStatus.RUNNING,
    result: null,
    error: null,
    cancel() {
      handle.status = JobStatus.CANCELLED;
    },
  };

  handle.promise = load_operation()
    .then((result) => {
      if (handle.status === JobStatus.CANCELLED) {
        close_bitmap_mip_chains(result?.mip_chains);
      } else {
        handle.status = JobStatus.COMPLETED;
        handle.result = result;
        handle.error = null;
      }
      return result;
    })
    .catch((load_error) => {
      if (handle.status !== JobStatus.CANCELLED) {
        handle.status = JobStatus.FAILED;
        handle.error = load_error;
        handle.result = null;
      }
      throw load_error;
    });

  return handle;
}

function create_texture_load_job_payload(paths, config = {}) {
  const pool = config.pool_key ? TextureArrayPools.get_pool(config.pool_key) : null;
  const semantic_cap = config.pool_key
    ? TextureArrayPools.get_dimension_cap(config.pool_key)
    : Number.POSITIVE_INFINITY;
  const global_cap = TextureManager.get_max_texture_dimension();

  return {
    paths,
    base_url:
      typeof document !== "undefined"
        ? document.baseURI
        : typeof window !== "undefined"
          ? window.location.href
          : undefined,
    no_mips: !!config.no_mips,
    pool_dimension_cap: config.pool_key
      ? TextureArrayPools.get_dimension_cap(config.pool_key)
      : undefined,
    pooled_width: pool?.config.width,
    pooled_height: pool?.config.height,
    pooled_mip_levels: pool?.config.mip_levels,
    supports_bc: Renderer.get().has_bc,
    max_texture_dimension: Math.min(semantic_cap, global_cap),
    texture_usage:
      config.texture_usage ??
      (config.pool_key
        ? config.pool_key === "albedo" || config.pool_key === "emission"
          ? "color"
          : "data"
        : undefined),
  };
}

function resolve_texture_load_target(source_width, source_height, config = {}) {
  const global_cap = TextureManager.get_max_texture_dimension();
  if (!config.pool_key) {
    const width = Math.max(1, Math.min(source_width, global_cap));
    const height = Math.max(1, Math.min(source_height, global_cap));
    return {
      width,
      height,
      mip_levels: config.no_mips ? 1 : Math.floor(Math.log2(Math.max(width, height))) + 1,
    };
  }

  const cap = Math.min(TextureArrayPools.get_dimension_cap(config.pool_key), global_cap);
  const pool = TextureArrayPools.get_pool(config.pool_key);
  const width = Math.max(1, Math.min(source_width, cap));
  const height = Math.max(1, Math.min(source_height, cap));
  const mip_levels = config.no_mips ? 1 : Math.floor(Math.log2(Math.max(width, height))) + 1;

  return {
    width: Math.max(width, pool?.config.width ?? 0),
    height: Math.max(height, pool?.config.height ?? 0),
    mip_levels: Math.max(mip_levels, pool?.config.mip_levels ?? 1),
  };
}

async function load_image_bitmap(path) {
  const resolved_image = await new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = path;
  });

  return await createImageBitmap(resolved_image, {
    colorSpaceConversion: "none",
  });
}

async function build_bitmap_mip_chain(texture, width, height, mip_levels) {
  let mip0_source = texture;
  if (texture.width !== width || texture.height !== height) {
    mip0_source = await createImageBitmap(texture, {
      resizeWidth: width,
      resizeHeight: height,
      resizeQuality: "high",
    });
  }

  const mip_chain = [mip0_source];
  for (let level = 1; level < mip_levels; level++) {
    mip_chain.push(
      await createImageBitmap(mip0_source, {
        resizeWidth: Math.max(1, width >> level),
        resizeHeight: Math.max(1, height >> level),
        resizeQuality: "high",
      })
    );
  }

  if (mip0_source !== texture) {
    texture.close?.();
  }
  return mip_chain;
}

async function load_image_mip_chains(paths, config = {}) {
  if (!Array.isArray(paths) || paths.length === 0) {
    return {
      mip_chains: [],
      source_width: 0,
      source_height: 0,
      width: 0,
      height: 0,
      mip_levels: 1,
    };
  }

  const bitmaps = await Promise.all(paths.map((path) => load_image_bitmap(path)));
  const base = bitmaps[0];
  const target = resolve_texture_load_target(base.width, base.height, config);
  const mip_chains = await Promise.all(
    bitmaps.map((texture) =>
      build_bitmap_mip_chain(texture, target.width, target.height, target.mip_levels)
    )
  );

  return {
    mip_chains,
    source_width: base.width,
    source_height: base.height,
    width: target.width,
    height: target.height,
    mip_levels: target.mip_levels,
  };
}

function texture_dimension_to_image_dimension(texture_dimension) {
  switch (texture_dimension) {
    case "1d":
      return "1d";
    case "3d":
      return "3d";
    default:
      return "2d";
  }
}

export class TextureStreamingProvider extends StreamProvider {
  static provider_type = texture_stream_provider_type;

  constructor(options = {}) {
    super(options);
    this.uploads_per_frame = options.uploads_per_frame ?? default_uploads_per_frame;
    this.pixels_per_frame = options.pixels_per_frame ?? default_pixels_per_frame;
  }

  begin_stream(request) {
    const texture = request.target;
    if (!texture?.config?.paths?.length) {
      throw new Error("Texture streaming requires a target with one or more paths.");
    }

    const handle = this.submit_load_request(texture.config.paths, texture.config);
    handle.promise?.catch(() => {});

    return {
      handle,
      texture_data: null,
      phase: TextureLoadPhase.WAITING,
      next_layer: 0,
      next_mip: 0,
      reload_only: request.options.reload_only === true,
    };
  }

  begin_frame() {
    return {
      uploads_remaining: this.uploads_per_frame,
      pixels_remaining: this.pixels_per_frame,
      initial_upload_count: this.uploads_per_frame,
    };
  }

  update_stream(request, context) {
    const texture = request.target;
    const state = request.state;
    const budget = context.frame;

    if (state.phase === TextureLoadPhase.WAITING) {
      if (state.handle?.status === JobStatus.FAILED) {
        throw state.handle.error ?? new Error(`Texture load failed for '${texture.config.name}'.`);
      }
      if (state.handle?.status === JobStatus.CANCELLED) {
        return StreamUpdateStatus.CANCEL;
      }
      if (state.handle?.status !== JobStatus.COMPLETED) {
        return StreamUpdateStatus.PENDING;
      }

      state.texture_data = state.handle.result;
      state.handle = null;
      if (!this.prepare_streaming_upload(texture, state)) {
        return StreamUpdateStatus.PENDING;
      }
    }

    if (state.phase !== TextureLoadPhase.UPLOADING || !state.texture_data) {
      return StreamUpdateStatus.PENDING;
    }

    while (
      state.next_layer < state.texture_data.mip_chains.length &&
      budget.uploads_remaining > 0
    ) {
      const mip_chain = state.texture_data.mip_chains[state.next_layer];
      const mip_data = mip_chain[state.next_mip];
      const pixel_cost = Math.max(1, mip_data.width * mip_data.height);

      if (
        budget.uploads_remaining < budget.initial_upload_count &&
        pixel_cost > budget.pixels_remaining
      ) {
        break;
      }

      if (mip_data.data) {
        texture._upload_texture_data(state.next_layer, state.next_mip, mip_data);
      } else {
        texture._upload_bitmap(
          state.next_layer,
          state.next_mip,
          mip_data,
          texture.config.flip_y !== undefined ? texture.config.flip_y : true
        );
      }

      budget.uploads_remaining--;
      budget.pixels_remaining = Math.max(0, budget.pixels_remaining - pixel_cost);

      state.next_mip++;
      if (state.next_mip >= mip_chain.length) {
        state.next_layer++;
        state.next_mip = 0;
      }
    }

    if (state.next_layer >= state.texture_data.mip_chains.length) {
      return {
        status: StreamUpdateStatus.COMPLETE,
        result: texture,
      };
    }
    return StreamUpdateStatus.CONTINUE;
  }

  prepare_streaming_upload(texture, state) {
    const texture_data = state.texture_data;
    if (!texture_data) {
      return false;
    }

    if (state.reload_only) {
      texture.config.depth = texture_data.mip_chains.length;
      texture.config.width = texture_data.width;
      texture.config.height = texture_data.height;
      texture.config.mip_levels = texture_data.mip_levels;
    } else {
      const renderer = Renderer.get();
      const placeholder_image = texture.image;

      if (texture_data.format) {
        // Preserve the engine's established color-space contract. Source color
        // textures contain gamma-encoded values, but the material shaders and
        // lighting pipeline currently expect them through an unorm view.
        texture.config.format = texture_data.format;
      }

      texture.config.width = texture_data.source_width;
      texture.config.height = texture_data.source_height;
      texture.config.depth = texture_data.mip_chains.length;

      if (texture.config.pool_key) {
        texture.config.width = texture_data.width;
        texture.config.height = texture_data.height;
        texture.config.mip_levels = texture_data.mip_levels;

        const allocation = TextureArrayPools.allocate_loaded(texture.config, texture);
        texture.image = allocation.texture.image;
        texture.views = allocation.texture.views;
        texture.bindless_handle = allocation.index;

        const pool = TextureArrayPools.get_pool(texture.config.pool_key);
        texture.config.width = pool.config.width;
        texture.config.height = pool.config.height;
        texture.config.mip_levels = pool.config.mip_levels;
      } else {
        texture.config.width = texture_data.width;
        texture.config.height = texture_data.height;
        texture.config.mip_levels = texture_data.mip_levels;

        let texture_usage = texture.config.usage | GPUTextureUsage.COPY_DST;
        if (texture_data.compressed) {
          texture_usage &= ~(GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING);
        } else {
          texture_usage |= GPUTextureUsage.RENDER_ATTACHMENT;
        }

        texture.image = renderer.device.createTexture({
          label: texture.config.name,
          size: {
            width: texture.config.width,
            height: texture.config.height,
            depthOrArrayLayers: texture.config.depth,
          },
          mipLevelCount: texture.config.mip_levels,
          sampleCount: texture.config.sample_count,
          format: texture.config.format,
          usage: texture_usage,
          dimension: texture_dimension_to_image_dimension(texture.config.dimension),
        });
      }

      if (placeholder_image && placeholder_image !== texture.image) {
        const old_image = placeholder_image;
        const timestamp = typeof performance !== "undefined" ? performance.now() : Date.now();
        renderer.execution_queue.push_execution(
          () => old_image.destroy(),
          Name.from(`${texture.config.name}_placeholder_destroy_${timestamp}`),
          MAX_BUFFERED_FRAMES + 1
        );
      }
    }

    if (
      !bitmap_mip_chains_match_config(
        texture_data.mip_chains,
        texture.config.width,
        texture.config.height,
        texture.config.mip_levels
      )
    ) {
      close_bitmap_mip_chains(texture_data.mip_chains);
      state.texture_data = null;
      state.handle = this.submit_load_request(texture.config.paths, texture.config);
      state.handle.promise?.catch(() => {});
      state.phase = TextureLoadPhase.WAITING;
      state.next_layer = 0;
      state.next_mip = 0;
      return false;
    }

    state.phase = TextureLoadPhase.UPLOADING;
    state.next_layer = 0;
    state.next_mip = 0;
    return true;
  }

  complete_stream(request) {
    const texture = request.target;
    const state = request.state;

    if (!texture.config.pool_key) {
      texture._setup_views();
    }
    if (texture.config.material_notifier) {
      global_dispatcher.dispatch(texture.config.material_notifier, texture);
    }
    if (texture.config.pool_key && !state.reload_only) {
      global_dispatcher.dispatch(`texture_pool_${texture.config.pool_key}`, texture);
    }

    Renderer.get().mark_bind_groups_dirty(true);
    this.release_stream_state(state);
  }

  cancel_stream(request) {
    this.release_stream_state(request.state);
  }

  release_stream_state(state) {
    if (!state) {
      return;
    }
    state.handle?.cancel?.();
    close_bitmap_mip_chains(state.texture_data?.mip_chains ?? state.handle?.result?.mip_chains);
    state.handle = null;
    state.texture_data = null;
  }

  submit_load_request(paths, config = {}) {
    if (JobSystem.is_supported()) {
      return JobSystem.submit(
        "load_texture_bitmaps",
        create_texture_load_job_payload(paths, config)
      );
    }
    return create_local_load_handle(() => load_image_mip_chains(paths, config));
  }

  serialize(texture) {
    if (!texture?.config) {
      throw new Error("TextureStreamingProvider.serialize requires a texture.");
    }

    return serialize_json(
      {
        provider_type: this.provider_type,
        version: texture_stream_format_version,
        config: texture.config,
      },
      {
        replacer: (_key, value) =>
          typeof value === "function" || typeof value === "symbol" ? undefined : value,
      }
    );
  }

  deserialize(payload, context = {}) {
    const descriptor =
      typeof payload === "string" || payload instanceof ArrayBuffer || ArrayBuffer.isView(payload)
        ? deserialize_json(payload, "Texture stream descriptor")
        : payload;

    if (!descriptor || descriptor.provider_type !== this.provider_type) {
      throw new Error("Texture stream descriptor has an invalid provider type.");
    }
    if (descriptor.version !== texture_stream_format_version) {
      throw new Error(`Unsupported texture stream descriptor version '${descriptor.version}'.`);
    }
    if (!descriptor.config || typeof descriptor.config !== "object") {
      throw new Error("Texture stream descriptor is missing its configuration.");
    }

    const config = {
      ...descriptor.config,
      paths: Array.isArray(descriptor.config.paths)
        ? [...descriptor.config.paths]
        : descriptor.config.paths,
    };
    const target = context.target ?? context.create_target?.(config) ?? null;
    if (!target) {
      return config;
    }

    if (context.stream === false) {
      target.config = { ...target.config, ...config };
    } else if (typeof target.load === "function") {
      target.load(config);
    } else {
      throw new Error("Texture stream deserialization target must implement load(config).");
    }
    return target;
  }

  static install(system = StreamingSystem.get(), options = {}) {
    if (system.has_provider(this.provider_type)) {
      return system.get_provider(this.provider_type);
    }
    return system.register_provider(new TextureStreamingProvider(options));
  }

  static begin_streaming_load(texture, reload_only = false) {
    const system = StreamingSystem.get();
    this.install(system);

    texture_stream_requests.get(texture)?.cancel();
    const request = system.stream(this.provider_type, texture, {
      reload_only,
    });
    texture_stream_requests.set(texture, request);
    request.finished.then(() => {
      if (texture_stream_requests.get(texture) === request) {
        texture_stream_requests.delete(texture);
      }
    });
    return request;
  }

  static cancel_streaming_load(texture) {
    const request = texture_stream_requests.get(texture);
    texture_stream_requests.delete(texture);
    if (request) {
      return request.cancel();
    }
    return StreamingSystem.cancel_target(this.provider_type, texture);
  }
}
