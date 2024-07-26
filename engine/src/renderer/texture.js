import { Name } from "../utility/names.js";
import { ResourceCache, CacheTypes } from "./resource_cache.js";

/**
 * Flags for image resources in the render graph.
 * @enum {number}
 */
export const ImageFlags = Object.freeze({
  /** No flags */
  None: 0,
  /** Indicates a transient image resource */
  Transient: 1,
  /** Indicates the image is loaded locally */
  LocalLoad: 2,
});

/**
 * Configuration for a texture sampler.
 * @typedef {Object} TextureSamplerConfig
 * @property {string|null} name - The name of the sampler.
 * @property {string} mag_filter - The magnification filter. Default is "linear".
 * @property {string} min_filter - The minification filter. Default is "linear".
 * @property {string} mipmap_filter - The mipmap filter. Default is "linear".
 */
class TextureSamplerConfig {
  name = null;
  mag_filter = "linear";
  min_filter = "linear";
  mipmap_filter = "linear";
}

export class TextureSampler {
  config = new TextureSamplerConfig();
  sampler = null;

  init(context, config) {
    this.config = { ...this.config, ...config };

    this.sampler = context.device.createSampler({
      label: this.config.name,
      magFilter: this.config.mag_filter,
      minFilter: this.config.min_filter,
      mipmapFilter: this.config.mipmap_filter,
    });
  }

  static create(context, config) {
    let sampler = ResourceCache.get().fetch(
      CacheTypes.SAMPLER,
      Name.from(config.name)
    );
    if (sampler) {
      return sampler;
    }

    sampler = new TextureSampler();
    sampler.init(context, config);

    ResourceCache.get().store(
      CacheTypes.SAMPLER,
      Name.from(config.name),
      sampler
    );

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
  clear_value = { r: 0, g: 0, b: 0, a: 1 };
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
  init(context, config) {
    this.config = { ...this.config, ...config };
    this.config.type = config.format.includes("depth") ? "depth" : "color";

    if (this.config.type === "depth") {
      this.config.clear_value = 1.0;
      this.config.load_op = "clear";
    }

    this.image = context.device.createTexture({
      label: config.name,
      size: {
        width: config.width,
        height: config.height,
        depthOrArrayLayers: config.depth,
      },
      mipLevelCount: config.mip_levels,
      sampleCount: config.sample_count,
      dimension: config.dimension,
      format: config.format,
      usage: config.usage,
    });

    this._setup_views();
  }

  async load(context, paths, config) {
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

    const textures = await Promise.all(paths.map(load_image_bitmap));

    this.config.width = textures[0].width;
    this.config.height = textures[0].height;
    this.config.depth = textures.length;

    this.image = context.device.createTexture({
      label: this.config.name,
      size: {
        width: this.config.width,
        height: this.config.height,
        depthOrArrayLayers: this.config.depth,
      },
      mipLevelCount: config.mip_levels,
      sampleCount: config.sample_count,
      format: config.format,
      usage: config.usage,
    });

    textures.forEach((texture, layer) => {
      context.device.queue.copyExternalImageToTexture(
        { source: texture, flipY: true },
        { texture: this.image, origin: { x: 0, y: 0, z: layer } },
        [texture.width, texture.height]
      );
    });

    this._setup_views();
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
      this.config.clear_value = { r: 0, g: 0, b: 0, a: 1 };
      this.config.load_op = "clear";
    }

    this._setup_views();
  }

  create_view(view_config = {}) {
    const view_descriptor = {
      label: view_config.label ?? this.config.name,
      format: this.config.format,
      dimension: view_config.dimension ?? this.config.dimension,
      aspect: view_config.aspect ?? "all",
      baseMipLevel: view_config.base_mip_level ?? 0,
      baseArrayLayer: view_config.base_array_layer ?? 0,
      arrayLayerCount: view_config.array_layers ?? this.config.depth,
    }

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
    encoder.copyBufferToImage(
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

  static get_default_sampler(context) {
    return TextureSampler.create(context, {
      name: "default_sampler",
    });
  }

  static create(context, config) {
    let image = ResourceCache.get().fetch(
      CacheTypes.IMAGE,
      Name.from(config.name)
    );

    if (image) {
      return image;
    }

    image = new Texture();

    image.init(context, config);

    ResourceCache.get().store(CacheTypes.IMAGE, Name.from(config.name), image);

    return image;
  }

  static create_from_texture(raw_image, name) {
    let cached_image = ResourceCache.get().fetch(
      CacheTypes.IMAGE,
      Name.from(name)
    );

    if (cached_image) {
      cached_image.set_image(raw_image);
      return cached_image;
    }

    cached_image = new Texture();

    cached_image.config = { name: name };

    ResourceCache.get().store(CacheTypes.IMAGE, Name.from(name), cached_image);

    cached_image.set_image(raw_image);

    return cached_image;
  }

  static async load(context, paths, config) {
    let image = ResourceCache.get().fetch(
      CacheTypes.IMAGE,
      Name.from(config.name)
    );

    if (image) {
      return image;
    }

    image = new Texture();
    await image.load(context, paths, config);

    ResourceCache.get().store(CacheTypes.IMAGE, Name.from(config.name), image);

    return image;
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
        return "2d_array";
      case "texture_cube_array":
        return "cube_array";
      default:
        return "2d";
    }
  }
}
