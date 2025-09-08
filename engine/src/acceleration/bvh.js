import { Renderer } from "../renderer/renderer.js";
import { Buffer } from "../renderer/buffer.js";

const MORTON_CODES_BUFFER_NAME = "morton_codes_buffer";
const TEMP_MORTON_CODES_BUFFER_NAME = "temp_morton_codes_buffer";
const SORTED_INDICES_BUFFER_NAME = "sorted_indices_buffer";
const TEMP_SORTED_INDICES_BUFFER_NAME = "temp_sorted_indices_buffer";
const BVH4_NODES_BUFFER_NAME = "bvh4_nodes_buffer";
const PARENT_IDX_BUFFER_NAME = "bvh_parent_idx";
const SCENE_BVH_BUFFER_NAME = "scene_bvh_buffer";
const ONESWEEP_GLOBAL_HIST_BUFFER_NAME = "onesweep_global_histogram";
const ONESWEEP_PASS_HIST_BUFFER_NAME = "onesweep_pass_histogram";
const ONESWEEP_TILE_INDICES_BUFFER_NAME = "onesweep_tile_indices";
const ONESWEEP_ERROR_COUNT_BUFFER_NAME = "onesweep_error_count";
const BVH4_BUILD_STATE_BUFFER_NAME = "bvh4_build_state";
const BVH4_INDEX_PAIRS_BUFFER_NAME = "bvh4_index_pairs";
const BVH4_PRIM_INDICES_BUFFER_NAME = "bvh4_prim_indices";
const BVH4_DEBUG_WATCHDOG_BUFFER_NAME = "bvh4_debug_watchdog";

// Onesweep configuration
const RADIX_BITS = 8;
export const RADIX = 1 << RADIX_BITS; // 256
export const RADIX_PASSES = 32 / RADIX_BITS; // 4
export const WORKGROUP_SIZE = 256;
export const ITEMS_PER_TILE = 16;
export const TILE_SIZE = WORKGROUP_SIZE * ITEMS_PER_TILE;

// float4 min_point (xyz + additional_data as w)
// float4 max_point (xyz + additional_data as w)
const SCENE_BVH_BYTE_SIZE = 32;
const BVH4_NODE_BYTE_SIZE = 48;
const DEFAULT_BVH_SIZE = 1024;

const storage_usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

/**
 * Standalone BVH implementation optimized for cache access patterns
 * and decoupled from entity system
 */
export class BVH {
  // Static properties
  static is_initialized = false;
  static bvh_size = DEFAULT_BVH_SIZE;

  static scene_bounds = new Float32Array([
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY,
    0,
    Number.NEGATIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
    0,
  ]);

  static scene_bounds_buffer = null;
  static morton_codes_buffer = null;
  static temp_morton_codes_buffer = null;
  static sorted_indices_buffer = null;
  static temp_sorted_indices_buffer = null;
  static onesweep_global_hist_buffer = null;
  static onesweep_pass_hist_buffer = null;
  static onesweep_tile_indices_buffer = null;
  static onesweep_error_count_buffer = null;
  static bvh4_nodes_buffer = null;
  static bvh_info_buffer = null;
  static parent_idx_buffer = null;
  static bvh4_build_state_buffer = null;
  static bvh4_index_pairs_buffer = null;
  static bvh4_prim_indices_buffer = null;
  static bvh4_debug_watchdog_buffer = null;
  
  static modified = true;

  /**
   * Initialize the BVH
   */
  static initialize() {
    if (this.is_initialized) return;

    this.scene_bounds_buffer = Buffer.create({
      name: SCENE_BVH_BUFFER_NAME,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      size: SCENE_BVH_BYTE_SIZE,
      force: true,
    });

    // Set up the root node
    this.rebuild_buffers();

    this.is_initialized = true;
  }

  /**
   * Resize the BVH to accommodate more nodes
   * @param {number} new_size - The new size of the tree
   */
  static resize(new_size) {
    if (new_size <= this.bvh_size) return;
    this.bvh_size = new_size;
    this.modified = true;
  }

  /**
   * Rebuild GPU buffers if needed
   */
  static rebuild_buffers() {
    if (!this.modified) return;
    
    const required_primitive_size = this.bvh_size * 4;

    if (
      !this.morton_codes_buffer ||
      this.morton_codes_buffer.config.size < required_primitive_size
    ) {
      this.morton_codes_buffer = Buffer.create({
        name: MORTON_CODES_BUFFER_NAME,
        usage: storage_usage,
        size: required_primitive_size,
        force: true,
      });

      Renderer.get().mark_bind_groups_dirty(true);
    }

    if (
      !this.temp_morton_codes_buffer ||
      this.temp_morton_codes_buffer.config.size < required_primitive_size
    ) {
      this.temp_morton_codes_buffer = Buffer.create({
        name: TEMP_MORTON_CODES_BUFFER_NAME,
        usage: storage_usage,
        size: required_primitive_size,
        force: true,
      });

      Renderer.get().mark_bind_groups_dirty(true);
    }

    if (
      !this.sorted_indices_buffer ||
      this.sorted_indices_buffer.config.size < required_primitive_size
    ) {
      this.sorted_indices_buffer = Buffer.create({
        name: SORTED_INDICES_BUFFER_NAME,
        usage: storage_usage,
        size: required_primitive_size,
        force: true,
      });

      Renderer.get().mark_bind_groups_dirty(true);
    }

    if (
      !this.temp_sorted_indices_buffer ||
      this.temp_sorted_indices_buffer.config.size < required_primitive_size
    ) {
      this.temp_sorted_indices_buffer = Buffer.create({
        name: TEMP_SORTED_INDICES_BUFFER_NAME,
        usage: storage_usage,
        size: required_primitive_size,
        force: true,
      });
    }

    // Onesweep buffers (separate explicit buffers matching WGSL bindings)
    const max_thread_blocks = Math.max(1, Math.ceil(this.bvh_size / TILE_SIZE));
    const pass_hist_count = max_thread_blocks * RADIX * RADIX_PASSES;
    const pass_hist_size_bytes = pass_hist_count * 4;
    if (
      !this.onesweep_pass_hist_buffer ||
      this.onesweep_pass_hist_buffer.config.size < pass_hist_size_bytes
    ) {
      this.onesweep_pass_hist_buffer = Buffer.create({
        name: ONESWEEP_PASS_HIST_BUFFER_NAME,
        usage: storage_usage,
        size: pass_hist_size_bytes,
        force: true,
      });
      Renderer.get().mark_bind_groups_dirty(true);
    }

    const global_hist_count = RADIX * RADIX_PASSES;
    const global_hist_size_bytes = global_hist_count * 4;
    if (
      !this.onesweep_global_hist_buffer ||
      this.onesweep_global_hist_buffer.config.size < global_hist_size_bytes
    ) {
      this.onesweep_global_hist_buffer = Buffer.create({
        name: ONESWEEP_GLOBAL_HIST_BUFFER_NAME,
        usage: storage_usage,
        size: global_hist_size_bytes,
        force: true,
      });
      Renderer.get().mark_bind_groups_dirty(true);
    }

    const tile_indices_size_bytes = RADIX_PASSES * 4;
    if (
      !this.onesweep_tile_indices_buffer ||
      this.onesweep_tile_indices_buffer.config.size < tile_indices_size_bytes
    ) {
      this.onesweep_tile_indices_buffer = Buffer.create({
        name: ONESWEEP_TILE_INDICES_BUFFER_NAME,
        usage: storage_usage,
        size: tile_indices_size_bytes,
        force: true,
      });
      Renderer.get().mark_bind_groups_dirty(true);
    }

    const error_count_size_bytes = 4;
    if (
      !this.onesweep_error_count_buffer ||
      this.onesweep_error_count_buffer.config.size < error_count_size_bytes
    ) {
      this.onesweep_error_count_buffer = Buffer.create({
        name: ONESWEEP_ERROR_COUNT_BUFFER_NAME,
        usage: storage_usage,
        size: error_count_size_bytes,
        force: true,
      });
    }

    const required_bvh4_size = this.bvh_size * BVH4_NODE_BYTE_SIZE;

    if (!this.bvh4_nodes_buffer || this.bvh4_nodes_buffer.config.size < required_bvh4_size) {
      this.bvh4_nodes_buffer = Buffer.create({
        name: BVH4_NODES_BUFFER_NAME,
        usage: storage_usage,
        size: required_bvh4_size,
        force: true,
      });

      Renderer.get().mark_bind_groups_dirty(true);
    }

    // BVH2 info buffer (u32 values: leaf_count, bvh2_count, prim_count)
    if (!this.bvh_info_buffer) {
      this.bvh_info_buffer = Buffer.create({
        name: "bvh_info",
        usage: storage_usage | GPUBufferUsage.UNIFORM,
        size: 16,
        force: true,
      });
    }

    // Parent index buffer (u32 per boundary)
    if (!this.parent_idx_buffer || this.parent_idx_buffer.config.size < required_primitive_size) {
      this.parent_idx_buffer = Buffer.create({
        name: PARENT_IDX_BUFFER_NAME,
        usage: storage_usage,
        size: required_primitive_size,
        force: true,
      });
    }

    // BVH4 build state buffer (5 u32 values: work_counter, node_counter, leaf_counter, work_alloc_counter, prim_count)
    if (!this.bvh4_build_state_buffer) {
      this.bvh4_build_state_buffer = Buffer.create({
        name: BVH4_BUILD_STATE_BUFFER_NAME,
        usage: storage_usage,
        size: 20, // 5 * 4 bytes
        force: true,
      });
    }

    // BVH4 index pairs buffer (u64 per primitive for work queue)
    const index_pairs_size = this.bvh_size * 8; // u64 = 8 bytes
    if (!this.bvh4_index_pairs_buffer || this.bvh4_index_pairs_buffer.config.size < index_pairs_size) {
      this.bvh4_index_pairs_buffer = Buffer.create({
        name: BVH4_INDEX_PAIRS_BUFFER_NAME,
        usage: storage_usage,
        size: index_pairs_size,
        force: true,
      });
    }

    // BVH4 primitive indices buffer (u32 per primitive)
    if (!this.bvh4_prim_indices_buffer || this.bvh4_prim_indices_buffer.config.size < required_primitive_size) {
      this.bvh4_prim_indices_buffer = Buffer.create({
        name: BVH4_PRIM_INDICES_BUFFER_NAME,
        usage: storage_usage,
        size: required_primitive_size,
        force: true,
      });
    }

    // BVH4 debug watchdog buffer (3 u32 counters, padded to 16 bytes)
    if (!this.bvh4_debug_watchdog_buffer) {
      this.bvh4_debug_watchdog_buffer = Buffer.create({
        name: BVH4_DEBUG_WATCHDOG_BUFFER_NAME,
        usage: storage_usage,
        size: 16,
        force: true,
        cpu_readback: true,
        raw_data: new Uint32Array(4),
      });
    }

    this.modified = false;
  }

  /**
   * Clear the scene bounds
   */
  static clear_scene_bounds() {
    this.scene_bounds[0] = Number.POSITIVE_INFINITY;
    this.scene_bounds[1] = Number.POSITIVE_INFINITY;
    this.scene_bounds[2] = Number.POSITIVE_INFINITY;
    this.scene_bounds[3] = 0;
    this.scene_bounds[4] = Number.NEGATIVE_INFINITY;
    this.scene_bounds[5] = Number.NEGATIVE_INFINITY;
    this.scene_bounds[6] = Number.NEGATIVE_INFINITY;
    this.scene_bounds[7] = 0;
    this.scene_bounds_buffer.write_raw(this.scene_bounds);
  }

  /**
   * Get GPU data for binding to shaders
   * @returns {Object} - Object containing GPU buffers
   */
  static #data_buffers = {
    scene_bounds_buffer: null,
    morton_codes_buffer: null,
    temp_morton_codes_buffer: null,
    sorted_indices_buffer: null,
    temp_sorted_indices_buffer: null,
    bvh4_nodes_buffer: null,
    onesweep_global_hist_buffer: null,
    onesweep_pass_hist_buffer: null,
    onesweep_tile_indices_buffer: null,
    onesweep_error_count_buffer: null,
    parent_idx_buffer: null,
    bvh_info_buffer: null,
  };

  /**
   * Get GPU data for binding to shaders (and rebuild buffers if needed)
   * @returns {Object} - Object containing GPU buffers
   */
  static to_gpu_data() {
    this.rebuild_buffers();

    this.#data_buffers.scene_bounds_buffer = this.scene_bounds_buffer;
    this.#data_buffers.morton_codes_buffer = this.morton_codes_buffer;
    this.#data_buffers.temp_morton_codes_buffer = this.temp_morton_codes_buffer;
    this.#data_buffers.sorted_indices_buffer = this.sorted_indices_buffer;
    this.#data_buffers.temp_sorted_indices_buffer = this.temp_sorted_indices_buffer;
    this.#data_buffers.bvh4_nodes_buffer = this.bvh4_nodes_buffer;
    this.#data_buffers.bvh4_parents_buffer = this.bvh4_parents_buffer;
    this.#data_buffers.onesweep_global_hist_buffer = this.onesweep_global_hist_buffer;
    this.#data_buffers.onesweep_pass_hist_buffer = this.onesweep_pass_hist_buffer;
    this.#data_buffers.onesweep_tile_indices_buffer = this.onesweep_tile_indices_buffer;
    this.#data_buffers.onesweep_error_count_buffer = this.onesweep_error_count_buffer;
    this.#data_buffers.bvh_info_buffer = this.bvh_info_buffer;
    this.#data_buffers.parent_idx_buffer = this.parent_idx_buffer;
    this.#data_buffers.bvh4_build_state_buffer = this.bvh4_build_state_buffer;
    this.#data_buffers.bvh4_index_pairs_buffer = this.bvh4_index_pairs_buffer;
    this.#data_buffers.bvh4_prim_indices_buffer = this.bvh4_prim_indices_buffer;
    this.#data_buffers.bvh4_debug_watchdog_buffer = this.bvh4_debug_watchdog_buffer;

    return this.#data_buffers;
  }
}
