import { Name } from "@/utility/names.js";
import { ResourceCache, CacheTypes } from "@/renderer/resource_cache.js";

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
 * Configuration for a image resource.
 * @property {string} name - Name of the image.
 * @property {number} width - Width of the image.
 * @property {number} height - Height of the image.
 * @property {number} depth - Depth of the image (for 3D textures).
 * @property {number} array_layers - Number of array layers in the image.
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
class ImageConfig {
  name = null;
  width = 0;
  height = 0;
  depth = 0;
  array_layers = 1;
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
}

export class Image {
  config = new ImageConfig();
  image = null;
  view = null;

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
      loadOp: config.load_op ?? 'clear',
      storeOp: config.store_op ?? 'store',
      clearValue: config.clear_value ?? { r: 0, g: 0, b: 0, a: 1 },
    });

    this.view = this.create_view();
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

    this.view = this.create_view();
  }

  create_view(view_config = {}) {
    return this.image.createView({
      label: this.config.name,
      dimension: this.config.dimension,
      format: this.config.format,
      aspect: view_config.aspect ?? "all",
      baseMipLevel: view_config.baseMipLevel ?? 0,
      mipLevelCount: view_config.mipLevelCount ?? this.config.mip_levels,
      baseArrayLayer: view_config.baseArrayLayer ?? 0,
      arrayLayerCount: view_config.arrayLayerCount ?? this.config.array_layers,
    });
  }

  copy_buffer(encoder, buffer) {
    encoder.copyBufferToImage(
      { buffer: buffer.buffer },
      { texture: this.image },
      {
        width: this.config.width,
        height: this.config.height,
        depthOrArrayLayers: 1,
      }
    );
  }

  copy_texture(encoder, texture) {
    encoder.copyTextureToTexture(
      { texture: texture.image },
      { texture: this.image },
      {
        width: this.config.width,
        height: this.config.height,
        depthOrArrayLayers: 1,
      }
    );
  }

  get physical_id() {
    return Name.from(this.config.name);
  }

  static get_default_sampler(context) {
    let sampler = ResourceCache.get().fetch(
      CacheTypes.SAMPLER,
      Name.from("default_sampler")
    );
    if (sampler) {
      return sampler;
    }

    sampler = context.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      mipmapFilter: "linear",
    });

    ResourceCache.get().store(
      CacheTypes.SAMPLER,
      Name.from("default_sampler"),
      sampler
    );

    return sampler;
  }

  static create(context, config) {
    let image = ResourceCache.get().fetch(
      CacheTypes.IMAGE,
      Name.from(config.name)
    );
    if (!image) {
      image = new Image();
      image.init(context, config);
      ResourceCache.get().store(
        CacheTypes.IMAGE,
        Name.from(config.name),
        image
      );
    }
    return image;
  }

  static create_from_image(raw_image, name) {
    let cached_image = ResourceCache.get().fetch(
      CacheTypes.IMAGE,
      Name.from(name)
    );
    if (!cached_image) {
      cached_image = new Image();
      cached_image.config = { name: name };
      ResourceCache.get().store(
        CacheTypes.IMAGE,
        Name.from(name),
        cached_image
      );
    }
    cached_image.set_image(raw_image);
    return cached_image;
  }
}
