import { Renderer } from "./renderer.js";
import { Texture } from "./texture.js";
import { global_dispatcher } from "../core/dispatcher.js";

export const MIN_TEXTURES_PER_POOL = 8;

class TextureArrayPool {
  constructor(config) {
    this.config = config;
    this.next_index = 0;
    this.pool_key = config.pool_key || "default";
    this.pool_id = TextureArrayPools.next_pool_id++;
    this.capacity = MIN_TEXTURES_PER_POOL;
    this.ping_pong_index = 0;

    this.texture = Texture.create({
      name: `texture_pool_${this.pool_key}`,
      width: this.config.width,
      height: this.config.height,
      depth: this.capacity,
      mip_levels: config.mip_levels,
      sample_count: config.sample_count,
      format: config.format,
      usage: config.usage | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      dimension: "2d-array",
      force: true,
    });

    // Track UV scales per layer (used for writing to texture padding)
    this.uv_scales = new Array(this.capacity).fill(null).map(() => [1.0, 1.0]);
  }

  /**
   * Ensures the pool can accommodate the given dimensions.
   * If incoming dimensions exceed current pool size, resizes to the maximum.
   */
  ensure_dimensions(width, height) {
    const new_width = Math.max(this.config.width, width);
    const new_height = Math.max(this.config.height, height);

    if (new_width > this.config.width || new_height > this.config.height) {
      this._resize_dimensions(new_width, new_height);
    }
  }

  /**
   * Allocates a new layer in the texture pool.
   * @param {number} texture_width - Width of the texture to allocate.
   * @param {number} texture_height - Height of the texture to allocate.
   * @returns {number} The index of the allocated layer.
   */
  allocate(texture_width, texture_height) {
    if (this.next_index >= this.capacity) {
      this._grow();
    }
    const index = this.next_index++;

    // Compute and store UV scale for this layer
    const u_scale = texture_width / this.config.width;
    const v_scale = texture_height / this.config.height;
    this.uv_scales[index] = [u_scale, v_scale];

    // Write UV scale to the last pixel of this layer (padding area)
    // This allows shaders to read the scale without a separate buffer
    this._write_uv_scale_to_texture(index, u_scale, v_scale);

    return index;
  }

  /**
   * Writes UV scale values to the last pixel (width-1, height-1) of a texture layer.
   * Shaders can read this pixel to get the proper UV scaling for textures smaller than the pool.
   * Format: R = u_scale, G = v_scale, B = 0 (sentinel), A = 0 (sentinel)
   * The sentinel values (0, 0) in BA channels help identify this as metadata.
   */
  _write_uv_scale_to_texture(layer_index, u_scale, v_scale) {
    const renderer = Renderer.get();

    // Create pixel data based on texture format
    // For most formats, we encode scale in RG channels with BA as sentinel (0)
    const pixel_data = new Uint8Array([
      Math.round(u_scale * 255),  // R: u_scale
      Math.round(v_scale * 255),  // G: v_scale
      0,                           // B: sentinel
      0,                           // A: sentinel
    ]);

    renderer.device.queue.writeTexture(
      {
        texture: this.texture.image,
        mipLevel: 0,
        origin: { x: this.config.width - 1, y: this.config.height - 1, z: layer_index },
      },
      pixel_data,
      { bytesPerRow: 4 },
      { width: 1, height: 1, depthOrArrayLayers: 1 }
    );
  }

  /**
   * Resizes the pool to new dimensions, preserving existing texture data.
   * Called when an allocation requests dimensions larger than the current pool.
   */
  _resize_dimensions(new_width, new_height) {
    this.ping_pong_index = (this.ping_pong_index + 1) % 2;

    const old_texture = this.texture;
    const old_width = this.config.width;
    const old_height = this.config.height;

    this.config.width = new_width;
    this.config.height = new_height;

    const max_dim = Math.max(this.config.width, this.config.height);
    this.config.mip_levels = !!this.config.no_mips ? 1 : Math.floor(Math.log2(max_dim)) + 1;

    this.texture = Texture.create({
      name: `texture_pool_${this.pool_key}_${this.ping_pong_index}`,
      width: this.config.width,
      height: this.config.height,
      depth: this.capacity,
      mip_levels: this.config.mip_levels,
      sample_count: this.config.sample_count,
      format: this.config.format,
      usage: this.config.usage | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      dimension: "2d-array",
      force: true,
    });

    Renderer.get().enqueue_pre_commands(
      `${this.config.name}_resize`,
      (graph, frame_data, encoder) => {
        this.texture.copy_texture(encoder, old_texture);
      },
      false /*persistent*/
    );

    this.texture.rename(`texture_pool_${this.pool_key}`);

    // Recalculate and rewrite UV scales for all existing layers
    // Original texture size derived from: tex_size = old_scale * old_pool_size
    for (let i = 0; i < this.next_index; i++) {
      const [old_u_scale, old_v_scale] = this.uv_scales[i];
      const texture_width = old_u_scale * old_width;
      const texture_height = old_v_scale * old_height;
      const new_u_scale = texture_width / this.config.width;
      const new_v_scale = texture_height / this.config.height;
      this.uv_scales[i] = [new_u_scale, new_v_scale];
      this._write_uv_scale_to_texture(i, new_u_scale, new_v_scale);
    }

    global_dispatcher.dispatch(`texture_pool_${this.pool_key}`, this.texture);

    Renderer.get().mark_bind_groups_dirty(true);
  }

  _grow() {
    const old_capacity = this.capacity;
    this.capacity *= 2;
    this.ping_pong_index = (this.ping_pong_index + 1) % 2;

    const old_texture = this.texture;
    this.texture = Texture.create({
      name: `texture_pool_${this.pool_key}_${this.ping_pong_index}`,
      width: this.config.width,
      height: this.config.height,
      depth: this.capacity,
      mip_levels: this.config.mip_levels,
      sample_count: this.config.sample_count,
      format: this.config.format,
      usage: this.config.usage | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      dimension: "2d-array",
      force: true,
    });

    Renderer.get().enqueue_pre_commands(
      `${this.config.name}_grow`,
      (graph, frame_data, encoder) => {
        this.texture.copy_texture(encoder, old_texture);
      },
      false /*persistent*/
    );

    this.texture.rename(`texture_pool_${this.pool_key}`);

    // Expand UV scales array and initialize new slots
    const old_scales = this.uv_scales;
    this.uv_scales = new Array(this.capacity).fill(null).map((_, i) => {
      return i < old_capacity ? old_scales[i] : [1.0, 1.0];
    });

    // Rewrite UV scales to new texture (copy doesn't preserve the last pixel correctly
    // since it's now at a different location in the larger texture)
    for (let i = 0; i < this.next_index; i++) {
      const [u_scale, v_scale] = this.uv_scales[i];
      this._write_uv_scale_to_texture(i, u_scale, v_scale);
    }

    global_dispatcher.dispatch(`texture_pool_${this.pool_key}`, this.texture);

    Renderer.get().mark_bind_groups_dirty(true);
  }
}

export class TextureArrayPools {
  static pools = new Map();
  static next_pool_id = 0;
  static fallback_texture = null;
  static fallback_view = null;

  static allocate(config) {
    const key = config.pool_key || "default";

    let pool = this.pools.get(key);
    if (!pool) {
      pool = new TextureArrayPool(config);
      this.pools.set(key, pool);
      Renderer.get().mark_bind_groups_dirty(true);
    } else {
      pool.ensure_dimensions(config.width, config.height);
    }
    const index = pool.allocate(config.width, config.height);

    return {
      texture: pool.texture,
      index: index,
      pool_id: pool.pool_id,
    };
  }

  static get_pool(key) {
    return this.pools.get(key);
  }

  static get_fallback_view() {
    if (this.fallback_view) {
      return this.fallback_view;
    }

    // Create a tiny 1x1 array texture with many layers so any layer index resolves safely
    this.fallback_texture = Texture.create({
      name: "texture_pool_fallback",
      width: 1,
      height: 1,
      depth: MIN_TEXTURES_PER_POOL,
      mip_levels: 1,
      sample_count: 1,
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      dimension: "2d-array",
    });

    return this.fallback_texture;
  }
}
