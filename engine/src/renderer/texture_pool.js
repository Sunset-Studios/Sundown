import { Renderer } from "./renderer.js";
import { Texture } from "./texture.js";
import { ResourceCache } from "./resource_cache.js";
import { CacheTypes } from "./renderer_types.js";
import { global_dispatcher } from "../core/dispatcher.js";
import { MAX_BUFFERED_FRAMES } from "../core/minimal.js";
import { Name } from "../utility/names.js";

export const MIN_TEXTURES_PER_POOL = 8;

const pooled_texture_dimension_caps = Object.freeze({
  albedo: 1024,
  normal: 1024,
  roughness: 512,
  metallic: 512,
  ao: 512,
  height: 512,
  specular: 256,
  emission: 256,
  default: 1024,
});

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
    this.members = new Set();

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

  needs_resize(config) {
    return (
      config.width > this.config.width ||
      config.height > this.config.height ||
      (config.mip_levels ?? 1) > (this.config.mip_levels ?? 1)
    );
  }

  register(texture) {
    if (texture) {
      this.members.add(texture);
    }
  }

  /**
   * Allocates a new layer in the texture pool.
   * @returns {number} The index of the allocated layer.
   */
  allocate(texture = null) {
    if (texture?.bindless_handle >= 0) {
      this.register(texture);
      return texture.bindless_handle;
    }

    if (this.next_index >= this.capacity) {
      this._grow();
    }
    const index = this.next_index++;
    if (texture) {
      texture.bindless_handle = index;
      this.register(texture);
    }

    return index;
  }

  /**
   * Doubles the pool capacity while preserving existing content.
   * Since dimensions don't change, a simple texture copy is sufficient.
   */
  _grow() {
    this.capacity *= 2;
    const old_ping_pong_index = this.ping_pong_index;
    this.ping_pong_index = (this.ping_pong_index + 1) % 2;

    const old_texture = this.texture;

    // ─────────────────────────────────────────────────────────────────────────────
    // Remove old texture from cache BEFORE creating new one with the canonical name.
    // This prevents the destroy() call from accidentally removing the new texture.
    // We give it a temporary name that marks it as pending deletion.
    // ─────────────────────────────────────────────────────────────────────────────
    const deletion_name = `texture_pool_${this.pool_key}_delete_${old_ping_pong_index}`;
    old_texture.rename(deletion_name);

    this.texture = Texture.create({
      name: `texture_pool_${this.pool_key}`,
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

    this._sync_members_to_pool();

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

    global_dispatcher.dispatch(`texture_pool_${this.pool_key}`, this.texture);

    Renderer.get().mark_bind_groups_dirty(true);

    // Use the renderer's execution queue which is updated after onSubmittedWorkDone().
    // Wait MAX_BUFFERED_FRAMES + 1 to ensure all in-flight GPU work referencing the
    // old texture has completed before destroying it.
    Renderer.get().execution_queue.push_execution(
      () => old_texture.destroy(),
      Name.from(deletion_name),
      MAX_BUFFERED_FRAMES + 1
    );
  }

  async resize(config) {
    if (!this.needs_resize(config)) {
      return;
    }

    const old_ping_pong_index = this.ping_pong_index;
    this.ping_pong_index = (this.ping_pong_index + 1) % 2;

    const old_texture = this.texture;
    this.config = {
      ...this.config,
      width: Math.max(this.config.width, config.width),
      height: Math.max(this.config.height, config.height),
      mip_levels: Math.max(this.config.mip_levels ?? 1, config.mip_levels ?? 1),
    };

    const deletion_name = `texture_pool_${this.pool_key}_delete_${old_ping_pong_index}`;
    old_texture.rename(deletion_name);

    this.texture = Texture.create({
      name: `texture_pool_${this.pool_key}`,
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

    this._sync_members_to_pool();

    const reloads = [];
    for (const member of this.members) {
      if (member.config?.paths?.length) {
        reloads.push(member.reload_from_source());
      }
    }
    await Promise.all(reloads);

    global_dispatcher.dispatch(`texture_pool_${this.pool_key}`, this.texture);
    Renderer.get().mark_bind_groups_dirty(true);
    Renderer.get().execution_queue.push_execution(
      () => old_texture.destroy(),
      Name.from(deletion_name),
      MAX_BUFFERED_FRAMES + 1
    );
  }

  _sync_members_to_pool() {
    for (const member of this.members) {
      member.image = this.texture.image;
      member.views = this.texture.views;
      member.config.width = this.config.width;
      member.config.height = this.config.height;
      member.config.mip_levels = this.config.mip_levels;
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Texture Array Pools Static Manager
// ═══════════════════════════════════════════════════════════════════════════════
export class TextureArrayPools {
  static next_pool_id = 0;
  static fallback_texture = null;
  static fallback_view = null;

  static get_dimension_cap(pool_key) {
    return pooled_texture_dimension_caps[pool_key] ?? pooled_texture_dimension_caps.default;
  }

  static normalize_pool_config(config) {
    const normalized = { ...config };
    const cap = this.get_dimension_cap(config.pool_key || "default");

    normalized.width = Math.max(1, Math.min(normalized.width || 1, cap));
    normalized.height = Math.max(1, Math.min(normalized.height || 1, cap));

    const max_dim = Math.max(normalized.width, normalized.height);
    normalized.mip_levels = normalized.no_mips ? 1 : Math.floor(Math.log2(max_dim)) + 1;

    return normalized;
  }

  static allocate(config) {
    const normalized = this.normalize_pool_config(config);
    const key = normalized.pool_key || "default";

    let pool = ResourceCache.get().fetch(CacheTypes.IMAGE_POOL, key);
    if (!pool) {
      pool = new TextureArrayPool(normalized);
      ResourceCache.get().store(CacheTypes.IMAGE_POOL, key, pool);
      Renderer.get().mark_bind_groups_dirty(true);
    }
    const index = pool.allocate();

    return {
      texture: pool.texture,
      index: index,
      pool_id: pool.pool_id,
    };
  }

  static async allocate_loaded(config, texture) {
    const normalized = this.normalize_pool_config(config);
    const key = normalized.pool_key || "default";

    let pool = ResourceCache.get().fetch(CacheTypes.IMAGE_POOL, key);
    if (!pool) {
      pool = new TextureArrayPool(normalized);
      ResourceCache.get().store(CacheTypes.IMAGE_POOL, key, pool);
      Renderer.get().mark_bind_groups_dirty(true);
    } else if (pool.needs_resize(normalized)) {
      await pool.resize(normalized);
    }

    const index = pool.allocate(texture);

    return {
      texture: pool.texture,
      index,
      pool_id: pool.pool_id,
    };
  }

  static get_pool(key) {
    return ResourceCache.get().fetch(CacheTypes.IMAGE_POOL, key);
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
