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
      width: config.width,
      height: config.height,
      depth: this.capacity,
      mip_levels: config.mip_levels,
      sample_count: config.sample_count,
      format: config.format,
      usage: config.usage | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
      dimension: "2d-array",
      force: true,
    });
  }

  allocate() {
    if (this.next_index >= this.capacity) {
      this._grow();
    }
    const index = this.next_index++;
    return index;
  }

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
      usage: this.config.usage | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
      dimension: "2d-array",
      force: true,
    });

    Renderer.get().enqueue_pre_commands(
      this.config.name,
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
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      dimension: "2d-array",
    });

    return this.fallback_texture;
  }
}
