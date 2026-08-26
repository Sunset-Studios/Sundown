import { MAX_BUFFERED_FRAMES } from "../core/minimal.js";
import { Renderer } from "./renderer.js";
import { Name } from "../utility/names.js";
import { ResourceCache } from "./resource_cache.js";
import { ImageFlags } from "./renderer_types.js";
import { CacheTypes } from "./renderer_types.js";
import { TextureArrayPools } from "./texture_pool.js";
import { TextureStreamingProvider } from "../streaming/providers/texture_streaming_provider.js";
import {
  r8unorm_format,
} from "../utility/config_permutations.js";

/**
 * Configuration for a texture sampler.
 * @typedef {Object} TextureSamplerConfig
 * @property {string|null} name - The name of the sampler.
 * @property {string} mag_filter - The magnification filter. Default is "linear".
 * @property {string} min_filter - The minification filter. Default is "linear".
 * @property {string} mipmap_filter - The mipmap filter. Default is "linear".
 * @property {string} address_mode_u - The address mode for the U coordinate. Default is "repeat".
 * @property {string} address_mode_v - The address mode for the V coordinate. Default is "repeat".
 * @property {string} address_mode_w - The address mode for the W coordinate. Default is "repeat".
 */
class TextureSamplerConfig {
  name = null;
  mag_filter = "linear";
  min_filter = "linear";
  mipmap_filter = "linear";
  address_mode_u = "repeat";
  address_mode_v = "repeat";
  address_mode_w = "repeat";
}

/**
 * Configuration for a image resource.
 * @property {string} name - Name of the image.
 * @property {number} width - Width of the image.
 * @property {number} height - Height of the image.
 * @property {number} depth - Depth of the image (for 3D textures) or number of layers (for array textures).
 * @property {number} mip_levels - Number of mip levels in the image.
 * @property {string} format - Format of the image (e.g., "rgba8unorm").
 * @property {number} usage - Usage flags for the image (e.g., GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.SAMPLED).
 * @property {number} sample_count - Number of samples for multisampling.
 * @property {string} pool_key - Key for the texture pool, if one should be used.
 * @property {number} flags - Additional flags for the image (see ImageFlags enum).
 * @property {Object} clear_value - Clear value for the image (e.g., { r: 0, g: 0, b: 0, a: 1 }).
 * @property {Object} blend - Blend configuration for the image if used as a render target (e.g., { src_factor: "one", dst_factor: "zero" }).
 * @property {string} store_op - Store operation for the image (e.g., "store" or "discard").
 * @property {string} load_op - Load operation for the image (e.g., "load" or "clear").
 */
class TextureConfig {
  name = null;
  width = 0;
  height = 0;
  depth = 1;
  mip_levels = 1;
  format = "rgba8unorm";
  usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.SAMPLED;
  sample_count = 1;
  dimension = "2d";
  pool_key = null;
  flags = ImageFlags.None;
  clear_value = { r: 0, g: 0, b: 0, a: 0 };
  blend = null;
  load_op = "clear";
  store_op = "store";
  b_one_view_per_mip = false;
  b_one_view_per_layer = false;
}

export class TextureSampler {
  config = new TextureSamplerConfig();
  sampler = null;

  init(config) {
    const renderer = Renderer.get();

    this.config = { ...this.config, ...config };

    this.sampler = renderer.device.createSampler({
      label: this.config.name,
      addressModeU: this.config.address_mode_u,
      addressModeV: this.config.address_mode_v,
      addressModeW: this.config.address_mode_w,
      magFilter: this.config.mag_filter,
      minFilter: this.config.min_filter,
      mipmapFilter: this.config.mipmap_filter,
      compare: this.config.compare,
    });
  }

  static create(config) {
    let sampler = ResourceCache.get().fetch(CacheTypes.SAMPLER, Name.from(config.name));
    if (sampler) {
      return sampler;
    }

    sampler = new TextureSampler();
    sampler.init(config);

    ResourceCache.get().store(CacheTypes.SAMPLER, Name.from(config.name), sampler);

    return sampler;
  }
}
export class Texture {
  config = new TextureConfig();
  image = null;
  views = [];
  current_view = 0;
  bindless_handle = -1;

  // Create a GPU buffer to store the data
  init(config) {
    const renderer = Renderer.get();

    this.config = { ...this.config, ...config };
    this.config.type = config.format.includes("depth") ? "depth" : "color";

    if (Texture.is_compressed_format(this.config.format)) {
      this.config.usage &= ~(GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING);
      this.config.usage |= GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST;
    }

    if (this.config.type === "depth") {
      this.config.clear_value =
        this.config.depth_clear !== undefined ? this.config.depth_clear : 1.0;
      this.config.load_op = "clear";
    }

    if (this.config.pool_key) {
      // Set mip_levels before allocate so the pool is created with a full mip chain (first
      // allocation sets the pool's mip count). Otherwise the pool stays at 1 mip and distant
      // objects show moiré.
      const max_dim = Math.max(this.config.width, this.config.height);
      this.config.mip_levels = this.config.no_mips ? 1 : Math.floor(Math.log2(max_dim)) + 1;

      const allocation = TextureArrayPools.allocate(this.config);
      this.image = allocation.texture.image;
      this.views = allocation.texture.views;
      this.bindless_handle = allocation.index;

      const pool = TextureArrayPools.get_pool(this.config.pool_key);
      this.config.width = pool.config.width;
      this.config.height = pool.config.height;
      this.config.mip_levels = pool.config.mip_levels;
    } else {
      this.image = renderer.device.createTexture({
        label: this.config.name,
        size: {
          width: this.config.width,
          height: this.config.height,
          depthOrArrayLayers: this.config.dimension === "cube" ? 6 : this.config.depth,
        },
        mipLevelCount: this.config.mip_levels,
        sampleCount: this.config.sample_count,
        format: this.config.format,
        usage: this.config.usage,
        dimension: Texture.texture_dimension_to_image_dimension(this.config.dimension),
      });
      this._setup_views();
    }

    Renderer.get().mark_bind_groups_dirty(true /* pass_only */);
  }

  load(config) {
    if (!config.paths || config.paths.length === 0) return;

    const renderer = Renderer.get();

    this.config = { ...this.config, ...config };
    this.config.type = config.format.includes("depth") ? "depth" : "color";

    if (this.config.type === "depth") {
      this.config.clear_value =
        this.config.depth_clear !== undefined ? this.config.depth_clear : 1.0;
      this.config.load_op = "clear";
    }

    // Create a default texture while we wait for the images to load
    this.image = renderer.device.createTexture({
      label: this.config.name,
      size: {
        width: 1,
        height: 1,
        depthOrArrayLayers: this.config.dimension === "cube" ? 6 : this.config.depth,
      },
      mipLevelCount: this.config.mip_levels,
      sampleCount: this.config.sample_count,
      format: this.config.format,
      dimension: Texture.texture_dimension_to_image_dimension(this.config.dimension),
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.SAMPLED |
        GPUTextureUsage.COPY_DST,
    });
    // Create a default view for the texture while we wait for the images to load
    if (!this.config.pool_key) {
      this._setup_views();
    }

    TextureStreamingProvider.begin_streaming_load(this, false);

    Renderer.get().mark_bind_groups_dirty(true /* pass_only */);
  }

  reload_from_source() {
    if (!this.config.pool_key || !this.config.paths?.length || !this.image || this.bindless_handle < 0) {
      return;
    }

    TextureStreamingProvider.begin_streaming_load(this, true);
  }

  destroy() {
    TextureStreamingProvider.cancel_streaming_load(this);
    TextureArrayPools.release_reservation(this);

    ResourceCache.get().remove(CacheTypes.IMAGE, Name.from(this.config.name));

    if (this.image && !this.config.pool_key) {
      const old_image = this.image;
      Renderer.get().execution_queue.push_execution(
        () => old_image.destroy(),
        `image_${this.physical_id}`,
        MAX_BUFFERED_FRAMES + 1
      );
    }
    this.image = null;
    this.views = [];
  }

  set_image(image) {
    // We should not be able to set an image for a pooled (bindless-style) texture, as that breaks
    // part of the pooling invariant.
    if (this.config.pool_key) return;

    this.image = image;
    this.config.width = image.width;
    this.config.height = image.height;
    this.config.depth = image.depthOrArrayLayers;
    this.config.mip_levels = image.mipLevelCount;
    this.config.sample_count = image.sampleCount;
    this.config.usage = image.usage;
    this.config.format = image.format;
    this.config.dimension = image.dimension;
    this.config.type = this.config.format.includes("depth") ? "depth" : "color";

    if (this.config.type === "depth") {
      this.config.clear_value =
        this.config.depth_clear !== undefined ? this.config.depth_clear : 1.0;
      this.config.load_op = "clear";
    } else {
      this.config.clear_value = { r: 0, g: 0, b: 0, a: 0 };
      this.config.load_op = "clear";
    }

    this._setup_views();
  }

  create_view(view_config = {}) {
    const view_dim = view_config.dimension ?? this.config.dimension;
    let array_layer_count;
    if (view_dim === "cube") {
      array_layer_count = 6;
    } else if (view_dim === "2d-array" || view_dim === "cube-array") {
      array_layer_count = view_config.array_layers ?? this.config.depth;
    } else {
      // 1d, 2d or 3d textures all have exactly one array layer
      array_layer_count = 1;
    }

    const view_descriptor = {
      label: view_config.label ?? this.config.name,
      format: this.config.format,
      dimension: view_dim,
      aspect: view_config.aspect ?? "all",
      baseMipLevel: view_config.base_mip_level ?? 0,
      baseArrayLayer: view_config.base_array_layer ?? 0,
      arrayLayerCount: array_layer_count,
    };

    if (view_config.mip_levels) {
      view_descriptor.mipLevelCount = view_config.mip_levels;
    }

    return this.image.createView(view_descriptor);
  }

  get_view(index) {
    return this.views[index];
  }

  copy_buffer(encoder, buffer) {
    const origin = {
      x: 0,
      y: 0,
      z: this.config.pool_key ? this.bindless_handle : 0,
    };
    encoder.copyBufferToTexture(
      { buffer: buffer.buffer },
      { texture: this.image, origin },
      {
        width: this.config.width,
        height: this.config.height,
        depthOrArrayLayers: 1,
      }
    );
  }

  copy_texture(encoder, texture) {
    if (!texture || !texture.image || !this.image) return;

    const src_origin = {
      x: 0,
      y: 0,
      z: texture.config.pool_key ? texture.bindless_handle : 0,
    };
    const dst_origin = {
      x: 0,
      y: 0,
      z: this.config.pool_key ? this.bindless_handle : 0,
    };
    const mip_levels = Math.max(texture.config.mip_levels, 1);
    for (let mip = 0; mip < mip_levels; mip++) {
      let w = Math.max(1, texture.config.width >> mip);
      let h = Math.max(1, texture.config.height >> mip);
      if (Texture.is_compressed_format(texture.config.format)) {
        w = Math.ceil(w / 4) * 4;
        h = Math.ceil(h / 4) * 4;
      }
      encoder.copyTextureToTexture(
        { texture: texture.image, mipLevel: mip, origin: src_origin },
        { texture: this.image, mipLevel: mip, origin: dst_origin },
        { width: w, height: h, depthOrArrayLayers: texture.config.depth }
      );
    }
  }

  write(
    data,
    origin = [0, 0, 0],
    data_offset = 0,
    cols = 0,
    rows = 0,
    components = 1,
    data_type = Float32Array
  ) {
    const renderer = Renderer.get();
    const is_array_buffer = ArrayBuffer.isView(data);
    const raw_data = is_array_buffer ? data : data.flat();
    const buffer_data = is_array_buffer ? raw_data : new data_type(raw_data);

    const unpadded_bytes_per_row = cols * components;

    // WebGPU requires bytesPerRow to be a multiple of 256 bytes
    const align = 256;
    const padded_bytes_per_row = Math.ceil(unpadded_bytes_per_row / align) * align;

    // If the row length is already aligned we can upload directly,
    // otherwise we copy each row into a padded buffer.
    let upload_array;
    if (unpadded_bytes_per_row === padded_bytes_per_row) {
      upload_array = buffer_data;
    } else {
      upload_array = new Uint8Array(padded_bytes_per_row * rows);
      for (let row = 0; row < rows; ++row) {
        const src_offset = row * unpadded_bytes_per_row;
        const dst_offset = row * padded_bytes_per_row;
        upload_array.set(
          buffer_data.subarray(src_offset, src_offset + unpadded_bytes_per_row),
          dst_offset
        );
      }
    }

    renderer.device.queue.writeTexture(
      { texture: this.image, origin: origin },
      upload_array,
      {
        offset: data_offset,
        bytesPerRow: padded_bytes_per_row,
        rowsPerImage: rows,
      },
      { width: cols, height: rows }
    );
  }

  rename(new_name) {
    ResourceCache.get().remove(CacheTypes.IMAGE, Name.from(this.config.name));
    this.config.name = new_name;
    ResourceCache.get().store(CacheTypes.IMAGE, Name.from(this.config.name), this);
  }

  get physical_id() {
    return Name.from(this.config.name);
  }

  get view() {
    return this.views[this.current_view];
  }

  _setup_views(view_config = {}) {
    this.views = [];
    this.current_view = 0;

    if (!this.config.b_one_view_per_layer && !this.config.b_one_view_per_mip) {
      this.views.push(this.create_view(view_config));
    }

    if (this.config.b_one_view_per_layer) {
      for (let i = 0; i < this.config.depth; i++) {
        const config = {
          ...view_config,
          label: `${this.config.name}_layer_${i}`,
          base_array_layer: i,
          array_layers: 1,
        };
        this.views.push(this.create_view(config));
      }
    }
    if (this.config.b_one_view_per_mip) {
      // Full view first (index 0) so passes that sample the full pyramid bind view 0 and can sample all mips
      this.views.push(this.create_view(
        {
          ...view_config,
          base_mip_level: 0,
          mip_levels: this.config.mip_levels,
          label: `${this.config.name}_full`
        }
      ));
      for (let i = 0; i < this.config.mip_levels; i++) {
        const config = {
          ...view_config,
          label: `${this.config.name}_mip_${i}`,
          base_mip_level: i,
          mip_levels: 1,
        };
        this.views.push(this.create_view(config));
      }
    }
  }

  _upload_bitmap(layer, mip_level, mip_bitmap, flip_y, renderer = Renderer.get()) {
    const targetLayer = this.config.pool_key ? this.bindless_handle + layer : layer;
    const expected_width = Math.max(1, this.config.width >> mip_level);
    const expected_height = Math.max(1, this.config.height >> mip_level);
    const width = Math.min(mip_bitmap.width, expected_width);
    const height = Math.min(mip_bitmap.height, expected_height);

    renderer.device.queue.copyExternalImageToTexture(
      { source: mip_bitmap, flipY: flip_y },
      { texture: this.image, mipLevel: mip_level, origin: { x: 0, y: 0, z: targetLayer } },
      [width, height]
    );
  }

  _upload_texture_data(layer, mip_level, mip_data, renderer = Renderer.get()) {
    if (
      !mip_data?.data ||
      mip_data.width < 1 ||
      mip_data.height < 1 ||
      mip_data.bytes_per_row < 1 ||
      mip_data.rows_per_image < 1
    ) {
      throw new Error(
        `Texture '${this.config.name}' received invalid cooked data for layer ${layer}, mip ${mip_level}.`
      );
    }
    const required_bytes = mip_data.bytes_per_row * mip_data.rows_per_image;
    if (mip_data.data.byteLength < required_bytes) {
      throw new Error(
        `Texture '${this.config.name}' layer ${layer}, mip ${mip_level} needs ${required_bytes} upload bytes but received ${mip_data.data.byteLength}.`
      );
    }

    const target_layer = this.config.pool_key ? this.bindless_handle + layer : layer;
    renderer.device.queue.writeTexture(
      {
        texture: this.image,
        mipLevel: mip_level,
        origin: { x: 0, y: 0, z: target_layer },
      },
      mip_data.data,
      {
        bytesPerRow: mip_data.bytes_per_row,
        rowsPerImage: mip_data.rows_per_image,
      },
      {
        // Compressed copies address whole physical texel blocks. The padded copy
        // extent is valid even for logical 2x2 and 1x1 tail mips.
        width: mip_data.copy_width ?? mip_data.width,
        height: mip_data.copy_height ?? mip_data.height,
        depthOrArrayLayers: 1,
      }
    );
  }

  static get_default_sampler() {
    return TextureSampler.create({
      name: "default_sampler",
      type: "filtering",
    });
  }

  static create(config) {
    let image = ResourceCache.get().fetch(CacheTypes.IMAGE, Name.from(config.name));

    if (image && config.force) {
      image.destroy();
      image = null;
    }

    if (!image) {
      image = new Texture();
      image.init(config);
      ResourceCache.get().store(CacheTypes.IMAGE, Name.from(config.name), image);
    }

    return image;
  }

  static create_from_texture(raw_image, name, config) {
    let cached_image = ResourceCache.get().fetch(CacheTypes.IMAGE, Name.from(name));

    if (cached_image && config && config.force) {
      cached_image.destroy();
      cached_image = null;
    } else if (cached_image) {
      cached_image.set_image(raw_image);
      return cached_image;
    }

    if (!cached_image) {
      cached_image = new Texture();
      cached_image.config = { name: name };
      ResourceCache.get().store(CacheTypes.IMAGE, Name.from(name), cached_image);
    }

    cached_image.set_image(raw_image);

    return cached_image;
  }

  static load(config) {
    let image = ResourceCache.get().fetch(CacheTypes.IMAGE, Name.from(config.name));

    if (image && config.force) {
      image.destroy();
      image = null;
    }

    if (!image) {
      image = new Texture();
      TextureArrayPools.reserve(config, image);
      try {
        image.load(config);
      } catch (error) {
        TextureArrayPools.release_reservation(image);
        throw error;
      }
      ResourceCache.get().store(CacheTypes.IMAGE, Name.from(config.name), image);
    }

    return image;
  }

  static is_compressed_format(format) {
    return /^(bc[1-7]h?|etc2|eac|astc)-/.test(format ?? "");
  }

  static #default = null;
  static default() {
    if (!Texture.#default) {
      Texture.#default = Texture.create({
        name: "default",
        width: 1,
        height: 1,
        format: "rgba8unorm",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.RENDER_ATTACHMENT |
          GPUTextureUsage.SAMPLED,
        clear_value: { r: 0, g: 0, b: 0, a: 0 },
      });
    }
    return Texture.#default;
  }

  static #default_array = null;
  static default_array() {
    if (!Texture.#default_array) {
      Texture.#default_array = Texture.create({
        name: "default",
        width: 1,
        height: 1,
        format: "rgba8unorm",
        dimension: "2d-array",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.RENDER_ATTACHMENT |
          GPUTextureUsage.SAMPLED,
        clear_value: { r: 0, g: 0, b: 0, a: 0 },
      });
    }
    return Texture.#default_array;
  }

  static #default_cube = null;
  static default_cube() {
    if (!Texture.#default_cube) {
      Texture.#default_cube = Texture.create({
        name: "default_cube",
        width: 1,
        height: 1,
        format: "rgba8unorm",
        dimension: "cube",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.RENDER_ATTACHMENT |
          GPUTextureUsage.SAMPLED,
        clear_value: { r: 1, g: 1, b: 1, a: 1 },
      });
    }
    return Texture.#default_cube;
  }

  static #default_blue_noise = null;
  static default_blue_noise() {
    if (!Texture.#default_blue_noise) {
      Texture.#default_blue_noise = Texture.load({
        name: "blue_noise_32x32",
        paths: [
          "engine/textures/noise/blue/LDR_RGB1_0.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_1.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_2.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_3.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_4.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_5.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_6.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_7.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_8.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_9.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_10.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_11.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_12.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_13.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_14.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_15.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_16.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_17.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_18.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_19.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_20.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_21.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_22.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_23.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_24.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_25.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_26.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_27.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_28.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_29.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_30.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_31.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_32.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_33.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_34.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_46.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_47.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_48.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_49.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_50.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_51.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_52.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_53.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_54.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_55.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_56.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_57.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_58.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_59.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_60.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_61.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_62.ktx2",
          "engine/textures/noise/blue/LDR_RGB1_63.ktx2",
        ],
        format: r8unorm_format,
        dimension: "2d-array",
        no_mips: true,
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.SAMPLED,
        force: true,
      });
    }
    return Texture.#default_blue_noise;
  }

  static filter_type_from_format(format) {
    const formatMap = {
      r8unorm: "float",
      r8snorm: "float",
      r8uint: "uint",
      r8sint: "sint",
      r16uint: "uint",
      r16sint: "sint",
      r16float: "float",
      rg8unorm: "float",
      rg8snorm: "float",
      rg8uint: "uint",
      rg8sint: "sint",
      r32uint: "uint",
      r32sint: "sint",
      r32float: "unfilterable-float",
      rg16uint: "uint",
      rg16sint: "sint",
      rg16float: "float",
      rgba8unorm: "float",
      "rgba8unorm-srgb": "float",
      rgba8snorm: "float",
      rgba8uint: "uint",
      rgba8sint: "sint",
      bgra8unorm: "float",
      "bgra8unorm-srgb": "float",
      rgb10a2unorm: "float",
      rg11b10ufloat: "float",
      rgb9e5ufloat: "float",
      rg32uint: "uint",
      rg32sint: "sint",
      rg32float: "unfilterable-float",
      rgba16uint: "uint",
      rgba16sint: "sint",
      rgba16float: "float",
      rgba32uint: "uint",
      rgba32sint: "sint",
      rgba32float: "unfilterable-float",
      // Depth formats
      depth16unorm: "depth",
      depth24plus: "depth",
      "depth24plus-stencil8": "depth",
      depth32float: "unfilterable-float",
      "depth32float-stencil8": "unfilterable-float",
      stencil8: "uint",
    };

    return formatMap[format] || "float"; // Default to 'float' if format is not found
  }

  static filter_type_from_binding_format(format) {
    const formatMap = {
      "f32": "float",
      "u32": "uint",
      "i32": "sint",
    };
    return formatMap[format] || "float";
  }

  static stride_from_format(format) {
    const formatMap = {
      r8unorm: 1,
      r8snorm: 1,
      r8uint: 1,
      r8sint: 1,
      r16uint: 2,
      r16sint: 2,
      r16float: 2,
      rg8unorm: 2,
      rg8snorm: 2,
      rg8uint: 2,
      rg8sint: 2,
      r32uint: 4,
      r32sint: 4,
      r32float: 4,
      rg16uint: 4,
      rg16sint: 4,
      rg16float: 4,
      rgba8unorm: 4,
      "rgba8unorm-srgb": 4,
      rgba8snorm: 4,
      rgba8uint: 4,
      rgba8sint: 4,
      bgra8unorm: 4,
      "bgra8unorm-srgb": 4,
      rgb10a2unorm: 4,
      rg11b10ufloat: 4,
      rgb9e5ufloat: 4,
      rg32uint: 4,
      rg32sint: 4,
      rg32float: 4,
      rgba16uint: 4,
      rgba16sint: 4,
      rgba16float: 4,
      rgba32uint: 4,
      rgba32sint: 4,
      rgba32float: 4,
      // Depth formats
      depth16unorm: 1,
      depth24plus: 1,
      "depth24plus-stencil8": 1,
      depth32float: 1,
      "depth32float-stencil8": 1,
      stencil8: 1,
    };

    return formatMap[format] || 1;
  }

  static dimension_from_type_name(type_name) {
    switch (type_name) {
      case "texture_2d":
      case "texture_2d_depth":
        return "2d";
      case "texture_cube":
        return "cube";
      case "texture_3d":
        return "3d";
      case "texture_array":
      case "texture_2d_array":
        return "2d-array";
      case "texture_cube_array":
        return "cube-array";
      default:
        return "2d";
    }
  }

  static texture_dimension_to_image_dimension(texture_dimension) {
    switch (texture_dimension) {
      case "1d":
        return "1d";
      case "2d":
      case "2d-array":
      case "cube-array":
        return "2d";
      case "3d":
        return "3d";
      default:
        return "2d";
    }
  }
}
