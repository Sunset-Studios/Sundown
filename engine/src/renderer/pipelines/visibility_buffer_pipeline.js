import { Buffer } from "../buffer.js";
import { RenderTaskQueue } from "../task_queues/render_task_queue.js";
import { ResourceCache } from "../resource_cache.js";
import { RenderPassFlags, MaterialPassType, CacheTypes } from "../renderer_types.js";
import { Texture } from "../texture.js";
import { r32uint_format } from "../../utility/config_permutations.js";
import { Name } from "../../utility/names.js";

const visibility_entity_image_config = {
  name: "visibility_entity",
  format: r32uint_format,
  width: 0,
  height: 0,
  usage:
    GPUTextureUsage.RENDER_ATTACHMENT |
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.COPY_SRC,
  clear_value: { r: 0xffffffff, g: 0, b: 0, a: 0 },
  force: false,
};

const visibility_surface_image_config = {
  name: "visibility_surface",
  format: r32uint_format,
  width: 0,
  height: 0,
  usage:
    GPUTextureUsage.RENDER_ATTACHMENT |
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.COPY_SRC,
  force: false,
};

const visibility_bucket_image_config = {
  name: "visibility_bucket",
  format: r32uint_format,
  width: 0,
  height: 0,
  usage:
    GPUTextureUsage.RENDER_ATTACHMENT |
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.COPY_SRC,
  force: false,
};

const visibility_bucket_meshlet_compact_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "visibility/compact_visible_meshlets_by_bucket.wgsl",
    },
  },
};

const meshlet_stats_capture_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "visibility/capture_meshlet_draw_stats.wgsl",
    },
  },
};

const meshlet_draw_args_reset_data = new Uint32Array([124 * 3, 0, 0, 0]);

function read_meshlet_draw_args(buffer_name) {
  const buffer = ResourceCache.get().fetch(CacheTypes.BUFFER, Name.from(buffer_name));
  const raw_data = buffer?.config?.raw_data;
  if (!raw_data || raw_data.length < 4) {
    return null;
  }

  return {
    vertex_count: Number(raw_data[0] ?? 0),
    instance_count: Number(raw_data[1] ?? 0),
    first_vertex: Number(raw_data[2] ?? 0),
    first_instance: Number(raw_data[3] ?? 0),
  };
}

function get_meshlet_stats_buffer_force(buffer_name, force_recreate) {
  const existing_buffer = ResourceCache.get().fetch(CacheTypes.BUFFER, Name.from(buffer_name));
  return (
    force_recreate ||
    !existing_buffer?.config?.cpu_readback ||
    ((existing_buffer?.config?.usage ?? 0) & GPUBufferUsage.COPY_SRC) === 0
  );
}

export class VisibilityBufferPipeline {
  visibility_entity_image = null;
  visibility_surface_image = null;
  visibility_bucket_image = null;
  registered_visibility_entity_image = null;
  registered_visibility_surface_image = null;
  registered_visibility_bucket_image = null;

  shader_setup_cache = new Map();
  bucket_info_buffers = new Map();

  recreate_persistent_resources(image_extent, force_recreate = false) {
    visibility_entity_image_config.width = image_extent.width;
    visibility_entity_image_config.height = image_extent.height;
    visibility_entity_image_config.force = force_recreate;

    visibility_surface_image_config.width = image_extent.width;
    visibility_surface_image_config.height = image_extent.height;
    visibility_surface_image_config.force = force_recreate;

    visibility_bucket_image_config.width = image_extent.width;
    visibility_bucket_image_config.height = image_extent.height;
    visibility_bucket_image_config.force = force_recreate;

    this.visibility_entity_image = Texture.create(visibility_entity_image_config);
    this.visibility_surface_image = Texture.create(visibility_surface_image_config);
    this.visibility_bucket_image = Texture.create(visibility_bucket_image_config);
  }

  register_targets(render_graph) {
    this.registered_visibility_entity_image = render_graph.register_image(
      this.visibility_entity_image.config.name
    );
    this.registered_visibility_surface_image = render_graph.register_image(
      this.visibility_surface_image.config.name
    );
    this.registered_visibility_bucket_image = render_graph.register_image(
      this.visibility_bucket_image.config.name
    );

    return this.get_registered_targets();
  }

  get_registered_targets() {
    return {
      visibility_entity_image: this.registered_visibility_entity_image,
      visibility_surface_image: this.registered_visibility_surface_image,
      visibility_bucket_image: this.registered_visibility_bucket_image,
    };
  }

  build_bucket_meshlet_draw_lists(
    render_graph,
    {
      current_view,
      meshlet_draw_count,
      object_instances,
      source_meshlet_list,
      source_meshlet_draw_args,
      buckets,
      stage_name,
      force_recreate = false,
    }
  ) {
    const bucket_draw_lists = new Map();
    const meshlet_list_capacity = Math.max(meshlet_draw_count, 1);

    if (meshlet_draw_count <= 0 || !buckets?.length) {
      return bucket_draw_lists;
    }

    for (const bucket of buckets) {
      if (!bucket?.shader) {
        continue;
      }

      const resources = this._create_bucket_meshlet_resources(render_graph, {
        current_view,
        bucket,
        stage_name,
        meshlet_list_capacity,
        force_recreate,
      });
      const bucket_info_buffer = render_graph.register_buffer(
        this._get_bucket_info_buffer(bucket).config.name
      );

      render_graph.add_pass(
        `init_visibility_bucket_${stage_name}_draw_args_view_${current_view}_bucket_${bucket.key}`,
        RenderPassFlags.GraphLocal,
        {},
        (graph, frame_data, encoder) => {
          graph
            .get_physical_buffer(resources.draw_args)
            .write(meshlet_draw_args_reset_data);
        }
      );

      if (__DEV__ && resources.stats) {
        render_graph.add_pass(
          `init_visibility_bucket_${stage_name}_stats_view_${current_view}_bucket_${bucket.key}`,
          RenderPassFlags.GraphLocal,
          {},
          (graph, frame_data, encoder) => {
            graph.get_physical_buffer(resources.stats).write(new Uint32Array([0, 0, 0, 0]));
          }
        );
      }

      render_graph.add_pass(
        `compact_visibility_bucket_${stage_name}_meshlets_view_${current_view}_bucket_${bucket.key}`,
        RenderPassFlags.Compute,
        {
          inputs: [
            object_instances,
            source_meshlet_list,
            source_meshlet_draw_args,
            bucket_info_buffer,
            resources.meshlet_list,
            resources.draw_args,
          ],
          outputs: [resources.meshlet_list, resources.draw_args],
          shader_setup: visibility_bucket_meshlet_compact_shader_setup,
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          pass.dispatch(Math.ceil(meshlet_list_capacity / 128), 1, 1);
        }
      );

      if (__DEV__ && resources.stats) {
        render_graph.add_pass(
          `capture_visibility_bucket_${stage_name}_stats_view_${current_view}_bucket_${bucket.key}`,
          RenderPassFlags.Compute,
          {
            inputs: [resources.draw_args, resources.stats],
            outputs: [resources.stats],
            shader_setup: meshlet_stats_capture_shader_setup,
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            pass.dispatch(1, 1, 1);
          }
        );
      }

      bucket_draw_lists.set(bucket.key, resources);
    }

    return bucket_draw_lists;
  }

  add_depth_prepass(
    render_graph,
    {
      enabled,
      meshlet_draw_count,
      current_view,
      depth_image,
      frustum_meshlet_draw_args,
      inputs,
      bucket,
    }
  ) {
    if (!enabled || meshlet_draw_count <= 0 || !bucket?.shader) {
      return depth_image;
    }

    const shader_setup = this._get_shader_setup(
      bucket,
      MaterialPassType.Depth,
      "less",
      true
    );
    const bucket_info_buffer = render_graph.register_buffer(this._get_bucket_info_buffer(bucket).config.name);

    render_graph.add_pass(
      `visibility_buffer_depth_prepass_view_${current_view}_bucket_${bucket.key}`,
      RenderPassFlags.Graphics,
      {
        inputs: [
          ...inputs,
          bucket_info_buffer,
        ],
        outputs: [depth_image],
        shader_setup,
        b_skip_pass_pipeline_setup: true,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        RenderTaskQueue.submit_visibility_bucket_indirect_draw(
          pass,
          bucket,
          frustum_meshlet_draw_args ? graph.get_physical_buffer(frustum_meshlet_draw_args) : null,
          MaterialPassType.Depth
        );
      }
    );

    return depth_image;
  }

  add_visibility_raster_pass(
    render_graph,
    {
      meshlet_draw_count,
      depth_prepass_enabled,
      current_view,
      depth_image,
      occlusion_meshlet_draw_args,
      inputs,
      bucket,
    }
  ) {
    if (meshlet_draw_count <= 0 || !bucket?.shader) {
      return this.get_registered_targets();
    }

    const shader_setup = this._get_shader_setup(
      bucket,
      MaterialPassType.Raster,
      depth_prepass_enabled ? "less-equal" : "less",
      !depth_prepass_enabled
    );
    const bucket_info_buffer = render_graph.register_buffer(this._get_bucket_info_buffer(bucket).config.name);

    render_graph.add_pass(
      `visibility_buffer_raster_view_${current_view}_bucket_${bucket.key}`,
      RenderPassFlags.Graphics,
      {
        inputs: [
          ...inputs,
          bucket_info_buffer,
        ],
        outputs: [
          this.registered_visibility_entity_image,
          this.registered_visibility_surface_image,
          this.registered_visibility_bucket_image,
          depth_image,
        ],
        shader_setup,
        b_skip_pass_pipeline_setup: true,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        RenderTaskQueue.submit_visibility_bucket_indirect_draw(
          pass,
          bucket,
          occlusion_meshlet_draw_args ? graph.get_physical_buffer(occlusion_meshlet_draw_args) : null,
          MaterialPassType.Raster
        );
      }
    );

    return this.get_registered_targets();
  }

  add_gbuffer_resolve_pass(
    render_graph,
    {
      meshlet_draw_count,
      current_view,
      depth_image,
      inputs,
      outputs,
      bucket,
    }
  ) {
    if (meshlet_draw_count <= 0 || !bucket?.shader) {
      return outputs;
    }

    const shader_setup = this._get_shader_setup(
      bucket,
      MaterialPassType.Resolve
    );
    const bucket_info_buffer = render_graph.register_buffer(this._get_bucket_info_buffer(bucket).config.name);

    render_graph.add_pass(
      `visibility_gbuffer_resolve_view_${current_view}_bucket_${bucket.key}`,
      RenderPassFlags.Graphics,
      {
        inputs: [
          this.registered_visibility_entity_image,
          this.registered_visibility_surface_image,
          this.registered_visibility_bucket_image,
          depth_image,
          ...inputs,
          bucket_info_buffer,
        ],
        outputs,
        shader_setup,
        b_skip_pass_pipeline_setup: true,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        RenderTaskQueue.submit_visibility_bucket_resolve(pass, bucket);
      }
    );

    return outputs;
  }

  add_forward_raster_pass(
    render_graph,
    {
      meshlet_draw_count,
      current_view,
      depth_image,
      occlusion_meshlet_draw_args,
      inputs,
      outputs,
      bucket,
    }
  ) {
    if (meshlet_draw_count <= 0 || !bucket?.forward_shader) {
      return outputs;
    }

    const shader_setup = this._get_shader_setup(
      bucket,
      MaterialPassType.Forward,
      "less-equal",
      false
    );
    const bucket_info_buffer = render_graph.register_buffer(this._get_bucket_info_buffer(bucket).config.name);

    render_graph.add_pass(
      `visibility_forward_raster_view_${current_view}_bucket_${bucket.key}`,
      RenderPassFlags.Graphics,
      {
        inputs: [
          ...inputs,
          bucket_info_buffer,
        ],
        outputs: [
          ...outputs,
          depth_image,
        ],
        shader_setup,
        b_skip_pass_pipeline_setup: true,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        RenderTaskQueue.submit_visibility_bucket_indirect_draw(
          pass,
          bucket,
          occlusion_meshlet_draw_args ? graph.get_physical_buffer(occlusion_meshlet_draw_args) : null,
          MaterialPassType.Forward
        );
      }
    );

    return outputs;
  }

  _get_shader_setup(bucket, pass_type = MaterialPassType.Raster, depth_compare = null, depth_write_enabled = null) {
    const cache_key = `${bucket.key}|${pass_type}|${depth_compare ?? "none"}|${depth_write_enabled ?? "null"}`;
    let cached_setup = this.shader_setup_cache.get(cache_key);
    if (cached_setup) {
      return cached_setup;
    }

    let shader = bucket.shader;
    if (pass_type === MaterialPassType.Depth) {
      shader = bucket.depth_shader;
    } else if (pass_type === MaterialPassType.Resolve) {
      shader = bucket.resolve_shader;
    } else if (pass_type === MaterialPassType.Forward) {
      shader = bucket.forward_shader;
    }

    const defines = {
      ...(shader?.defines ?? {}),
    };

    cached_setup = {
      pipeline_shaders: {
        vertex: {
          path: shader.file_path,
          defines,
        },
        fragment: {
          path: shader.file_path,
          defines,
        },
      },
      rasterizer_state: {
        cull_mode: "none",
      },
    };

    if (depth_compare !== null) {
      cached_setup.depth_stencil_compare_op = depth_compare;
    }
    if (depth_write_enabled !== null) {
      cached_setup.b_depth_write_enabled = depth_write_enabled;
    }

    this.shader_setup_cache.set(cache_key, cached_setup);

    return cached_setup;
  }

  _get_bucket_info_buffer(bucket) {
    let buffer = this.bucket_info_buffers.get(bucket.key);
    if (buffer) {
      return buffer;
    }

    buffer = Buffer.create({
      name: `visibility_bucket_info_${bucket.key}`,
      raw_data: new Uint32Array([bucket.key >>> 0, 0, 0, 0]),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.bucket_info_buffers.set(bucket.key, buffer);
    return buffer;
  }

  _create_bucket_meshlet_resources(
    render_graph,
    { current_view, bucket, stage_name, meshlet_list_capacity, force_recreate }
  ) {
    const meshlet_list_name = `visibility_bucket_${stage_name}_meshlet_list_view_${current_view}_bucket_${bucket.key}`;
    const draw_args_name = `visibility_bucket_${stage_name}_draw_args_view_${current_view}_bucket_${bucket.key}`;
    const stats_name = `visibility_bucket_${stage_name}_stats_view_${current_view}_bucket_${bucket.key}`;
    const required_meshlet_list_size = meshlet_list_capacity * 4 * Uint32Array.BYTES_PER_ELEMENT;
    const existing_meshlet_list = ResourceCache.get().fetch(
      CacheTypes.BUFFER,
      Name.from(meshlet_list_name)
    );

    return {
      meshlet_list: render_graph.create_buffer({
        name: meshlet_list_name,
        raw_data: new Uint32Array(meshlet_list_capacity * 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        force:
          force_recreate ||
          ((existing_meshlet_list?.config.size ?? 0) < required_meshlet_list_size),
      }),
      draw_args: render_graph.create_buffer({
        name: draw_args_name,
        raw_data: meshlet_draw_args_reset_data,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT,
        force: force_recreate,
      }),
      stats: __DEV__
        ? render_graph.create_buffer({
            name: stats_name,
            raw_data: new Uint32Array([0, 0, 0, 0]),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            cpu_readback: true,
            own_readback: true,
            force: get_meshlet_stats_buffer_force(stats_name, force_recreate),
          })
        : null,
    };
  }

  get_bucket_meshlet_stats(current_view, buckets, stage_name) {
    if (!__DEV__) {
      return [];
    }

    const stats = [];

    for (const bucket of buckets ?? []) {
      const draw_args = read_meshlet_draw_args(
        `visibility_bucket_${stage_name}_stats_view_${current_view}_bucket_${bucket.key}`
      );

      stats.push({
        key: bucket.key,
        template_name: bucket.template_name,
        family: bucket.family,
        shader_name: bucket.shader?.file_path ?? "",
        draw_args,
        instance_count: draw_args?.instance_count ?? 0,
      });
    }

    return stats;
  }
}
