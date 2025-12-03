import { Renderer } from "./renderer.js";
import { Texture } from "./texture.js";
import { global_dispatcher } from "../core/dispatcher.js";

export const MIN_TEXTURES_PER_POOL = 8;

// ═══════════════════════════════════════════════════════════════════════════════
// Texture Array Pool
// Manages a 2D texture array that can grow in capacity and dimensions.
// ═══════════════════════════════════════════════════════════════════════════════
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
      usage:
        config.usage |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.STORAGE_BINDING,
      dimension: "2d-array",
      force: true,
    });
  }

  /**
   * Allocates a new layer in the texture pool.
   * @returns {number} The index of the allocated layer.
   */
  allocate() {
    if (this.next_index >= this.capacity) {
      this._grow();
    }
    const index = this.next_index++;

    return index;
  }

  /**
   * Resizes the pool to new dimensions.
   */
  _resize_dimensions(new_width, new_height) {
    this.ping_pong_index = (this.ping_pong_index + 1) % 2;
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
      usage:
        this.config.usage |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.STORAGE_BINDING,
      dimension: "2d-array",
      force: true,
    });

    this.texture.rename(`texture_pool_${this.pool_key}`);

    global_dispatcher.dispatch(`texture_pool_${this.pool_key}`, this.texture);

    Renderer.get().mark_bind_groups_dirty(true);
  }

  /**
   * Doubles the pool capacity while preserving existing content.
   * Since dimensions don't change, a simple texture copy is sufficient.
   */
  _grow() {
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
      usage:
        this.config.usage |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.STORAGE_BINDING,
      dimension: "2d-array",
      force: true,
    });

    // ─────────────────────────────────────────────────────────────────────────────
    // Growing capacity doesn't change dimensions, so simple copy works.
    // Uses the pre-commands mechanism to run before the main render passes.
    // ─────────────────────────────────────────────────────────────────────────────
    Renderer.get().enqueue_pre_commands(
      `texture_pool_grow_${this.pool_key}`,
      (graph, frame_data, encoder) => {
        this.texture.copy_texture(encoder, old_texture);
      },
      false /*persistent*/
    );

    this.texture.rename(`texture_pool_${this.pool_key}`);

    global_dispatcher.dispatch(`texture_pool_${this.pool_key}`, this.texture);

    Renderer.get().mark_bind_groups_dirty(true);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Texture Array Pools Static Manager
// ═══════════════════════════════════════════════════════════════════════════════
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
    }
    const index = pool.allocate();

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
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
      dimension: "2d-array",
    });

    return this.fallback_texture;
  }
}
