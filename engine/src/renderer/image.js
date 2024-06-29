import Name from "@/utility/names.js";
import { ResourceCache, CacheTypes } from "@/renderer/resource_cache.js";

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
export class Image {
  config = null;
  image = null;

  // Create a GPU buffer to store the data
  init(context, config) {
    this.config = config;
    this.config.type = config.format.includes("depth") ? "depth" : "color";

    this.image = context.device.createImage({
      label: config.name,
      size: { x: config.width, y: config.height, z: config.depth },
      mipLevelCount: config.mip_levels,
      sampleCount: config.sample_count,
      dimension: config.dimension,
      format: config.format,
      usage: config.usage,
    });
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
    this.config.clear_value = { r: 0, g: 0, b: 0, a: 1 };
    this.config.load_op = "load";
    this.config.store_op = "store";
    this.config.type = this.config.format.includes("depth") ? "depth" : "color";
  }

  create_view(context, view_config) {
    return context.device.createImageView(this.image, {
      label: this.config.name,
      dimension: this.config.dimension,
      format: this.config.format,
      aspect: view_config.aspect,
      baseMipLevel: view_config.baseMipLevel,
      mipLevelCount: this.config.mip_levels,
      baseArrayLayer: view_config.baseArrayLayer,
      arrayLayerCount: view_config.arrayLayerCount,
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
