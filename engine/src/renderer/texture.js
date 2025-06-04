import { Renderer } from "./renderer.js";
import { Name } from "../utility/names.js";
import { ResourceCache } from "./resource_cache.js";
import { ImageFlags } from "./renderer_types.js";
import { CacheTypes } from "./renderer_types.js";
import { global_dispatcher } from "../core/dispatcher.js";

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
 * @property {boolean} b_is_bindless - Whether the image is bindless.
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
  b_is_bindless = false;
  flags = ImageFlags.None;
  clear_value = { r: 0, g: 0, b: 0, a: 0 };
  blend = null;
  load_op = "clear";
  store_op = "store";
  b_one_view_per_mip = false;
  b_one_view_per_layer = false;
}

export class Texture {
  config = new TextureConfig();
  image = null;
  views = [];
  current_view = 0;

  // Create a GPU buffer to store the data
  init(config) {
    const renderer = Renderer.get();

    this.config = { ...this.config, ...config };
    this.config.type = config.format.includes("depth") ? "depth" : "color";

    if (this.config.type === "depth") {
      this.config.clear_value = 1.0;
      this.config.load_op = "clear";
    }

    this.image = renderer.device.createTexture({
      label: config.name,
      size: {
        width: config.width,
        height: config.height,
        depthOrArrayLayers: config.dimension === "cube" ? 6 : config.depth,
      },
      mipLevelCount: config.mip_levels,
      sampleCount: config.sample_count,
      format: config.format,
      usage: config.usage,
      dimension: Texture.texture_dimension_to_image_dimension(config.dimension),
    });

    this._setup_views();
  }

  destroy() {
    ResourceCache.get().remove(CacheTypes.IMAGE, Name.from(this.config.name));
    this.image = null;
  }

  async load(paths, config) {
    const renderer = Renderer.get();

    this.config = { ...this.config, ...config };
    this.config.type = config.format.includes("depth") ? "depth" : "color";

    if (this.config.type === "depth") {
      this.config.clear_value = 1.0;
      this.config.load_op = "clear";
    }

    async function load_image_bitmap(path) {
      const resolved_img = await new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = path;
      });

      return await createImageBitmap(resolved_img, {
        colorSpaceConversion: "none",
      });
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
        GPUTextureUsage.SAMPLED,
    });

    this._setup_views();

    const textures = await Promise.all(paths.map(load_image_bitmap));

    // 1) compute how many mip‐levels we want
    const base = textures[0];
    const max_dim = Math.max(base.width, base.height);
    const mip_count = Math.floor(Math.log2(max_dim)) + 1;
    this.config = { ...this.config, ...config, mip_levels: mip_count };
    this.config.width  = base.width;
    this.config.height = base.height;
    this.config.depth  = textures.length;

    // 2) create the GPU texture with multiple mips
    this.image = renderer.device.createTexture({
      label: this.config.name,
      size: {
        width:  base.width,
        height: base.height,
        depthOrArrayLayers: this.config.depth,
      },
      mipLevelCount: this.config.mip_levels,
      sampleCount:   this.config.sample_count,
      format:        this.config.format,
      usage: this.config.usage,
      dimension:
        Texture.texture_dimension_to_image_dimension(
          this.config.dimension
        ),
    });

    for (let layer = 0; layer < textures.length; layer++) {
      const texture = textures[layer];
      // 3) copy the full-res image into mip 0
      renderer.device.queue.copyExternalImageToTexture(
        { source: texture, flipY: true },
        { texture: this.image, mipLevel: 0, origin: { x: 0, y: 0, z: layer } },
        [ texture.width, texture.height ]
      );

      // 4) for each subsequent level, use createImageBitmap to resize
      for (let lvl = 1; lvl < this.config.mip_levels; lvl++) {
        const w = Math.max(1, texture.width  >> lvl);
        const h = Math.max(1, texture.height >> lvl);
        const mip_bitmap = await createImageBitmap(texture, {
        resizeWidth:  w,
        resizeHeight: h,
        resizeQuality: "high",
      });
      renderer.device.queue.copyExternalImageToTexture(
        { source: mip_bitmap, flipY: true },
        { texture: this.image, mipLevel: lvl, origin: { x: 0, y: 0, z: layer } },
        [ w, h ]
      );
      }
    }

    // 5) rebuild all the texture views now that we've got new mips
    this._setup_views();

    if (this.config.material_notifier) {
      global_dispatcher.dispatch(this.config.material_notifier, this);
    }

    Renderer.get().mark_bind_groups_dirty(true /* pass_only */);
  }

  set_image(image) {
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
      this.config.clear_value = 1.0;
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

  set_current_view(index) {
    this.current_view = index;
  }

  get_view(index) {
    return this.views[index];
  }

  copy_buffer(encoder, buffer) {
    encoder.copyBufferToTexture(
      { buffer: buffer.buffer },
      { texture: this.image },
      {
        width: this.config.width,
        height: this.config.height,
        depthOrArrayLayers: this.config.depth,
      }
    );
  }

  copy_texture(encoder, texture) {
    encoder.copyTextureToTexture(
      { texture: texture.image },
      { texture: this.image },
      {
        width: texture.config.width,
        height: texture.config.height,
        depthOrArrayLayers: this.config.depth,
      }
    );
  }

  copy_external(encoder, image, origin = [0, 0, 0], cols = 0, rows = 0, flip_y = false) {
    encoder.copyExternalImageToTexture(
      { source: image, flipY: flip_y },
      { texture: this.image, origin: origin },
      [cols, rows]
    );
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
    const align                  = 256;
    const padded_bytes_per_row   =
        Math.ceil(unpadded_bytes_per_row / align) * align;

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

  static load(paths, config) {
    let image = ResourceCache.get().fetch(CacheTypes.IMAGE, Name.from(config.name));

    if (image && config.force) {
      image.destroy();
      image = null;
    }

    if (!image) {
      image = new Texture();
      image.load(paths, config);
      ResourceCache.get().store(CacheTypes.IMAGE, Name.from(config.name), image);
    }

    return image;
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
