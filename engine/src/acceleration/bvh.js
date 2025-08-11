import { Renderer } from "../renderer/renderer.js";
import { Buffer } from "../renderer/buffer.js";
import { ResizableBitArray, TypedStack } from "../memory/container.js";

const BVH_TREE_NODES_BOUNDS_BUFFER_NAME = "bvh_tree_nodes_bounds_buffer";
const MORTON_CODES_BUFFER_NAME = "morton_codes_buffer";
const TEMP_MORTON_CODES_BUFFER_NAME = "temp_morton_codes_buffer";
const SORTED_INDICES_BUFFER_NAME = "sorted_indices_buffer";
const TEMP_SORTED_INDICES_BUFFER_NAME = "temp_sorted_indices_buffer";
const BVH2_NODES_BUFFER_NAME = "bvh2_nodes_buffer";
const BVH4_NODES_BUFFER_NAME = "bvh4_nodes_buffer";
const SCENE_BVH_BUFFER_NAME = "scene_bvh_buffer";
const CLUSTERS_IN_BUFFER_NAME = "bvh_clusters_in";
const CLUSTERS_OUT_BUFFER_NAME = "bvh_clusters_out";
const USER_DATA_BUFFER_NAME = "user_data_buffer";

// float4 min_point (xyz + additional_data as w)
// float4 max_point (xyz + additional_data as w)
const NODE_BOUNDS_SIZE = 8; // Size in float32 elements
const SCENE_BVH_BYTE_SIZE = 32;
const BVH2_NODE_BYTE_SIZE = 56;
const BVH4_NODE_BYTE_SIZE = 72;
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
  static allocated_count = 0;
  static free_nodes = new TypedStack(DEFAULT_BVH_SIZE, Uint32Array);
  static free_nodes_bitmask = new ResizableBitArray(DEFAULT_BVH_SIZE);

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

  static bounds_buffer = null;
  static user_data_buffer = null;
  static scene_bounds_buffer = null;
  static morton_codes_buffer = null;
  static temp_morton_codes_buffer = null;
  static sorted_indices_buffer = null;
  static temp_sorted_indices_buffer = null;
  static bvh2_nodes_buffer = null;
  static bvh4_nodes_buffer = null;
  static histogram_buffer = null;
  static node_counters_buffer = null;
  static clusters_in_buffer = null;
  static clusters_out_buffer = null;

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

    this.free_nodes.resize(this.bvh_size);

    // Initialize new nodes as free
    for (let i = 0; i < this.bvh_size; i++) {
      this.free_nodes.push(i);
      this.free_nodes_bitmask.set(i, true);
    }

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

    const old_size = this.bvh_size; // old_size was this.size, it should be this.size before update
    this.bvh_size = new_size;

    this.free_nodes.resize(this.bvh_size);

    // Initialize new nodes as free
    for (let i = old_size; i < this.bvh_size; i++) {
      this.free_nodes.push(i);
      this.free_nodes_bitmask.set(i, true);
    }

    this.modified = true;
  }

  /**
   * Allocate a new node
   * @returns {number} - The index of the new node
   */
  static allocate_node() {
    if (this.free_nodes.length === 0) {
      this.resize(this.bvh_size * 2 + 1);
    }

    const node_index = this.free_nodes.pop();
    this.free_nodes_bitmask.set(node_index, false);

    ++this.allocated_count;

    this.modified = true;

    return node_index;
  }

  /**
   * Free a node
   * @param {number} node_index - The index of the node to free
   */
  static free_node(node_index) {
    if (node_index <= 0 || node_index >= this.bvh_size) return;

    // Check if the node is already free
    if (this.free_nodes_bitmask.get(node_index)) {
      return;
    }

    this.free_nodes.push(node_index);
    this.free_nodes_bitmask.set(node_index, true);

    --this.allocated_count;

    this.modified = true;
  }

  /**
   * Rebuild GPU buffers if needed
   */
  static rebuild_buffers() {
    if (!this.modified) return;

    const required_node_bounds_size = this.bvh_size * NODE_BOUNDS_SIZE;

    if (!this.bounds_buffer || this.bounds_buffer.config.size < required_node_bounds_size) {
      this.bounds_buffer = Buffer.create({
        name: BVH_TREE_NODES_BOUNDS_BUFFER_NAME,
        usage: storage_usage,
        size: required_node_bounds_size,
        force: true,
      });

      Renderer.get().mark_bind_groups_dirty(true);
    }

    const required_primitive_size = this.bvh_size * 4;

    if (!this.user_data_buffer || this.user_data_buffer.config.size < required_primitive_size) {
      this.user_data_buffer = Buffer.create({
        name: USER_DATA_BUFFER_NAME,
        usage: storage_usage,
        size: required_primitive_size,
        force: true,
      });

      Renderer.get().mark_bind_groups_dirty(true);
    }

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

    const histogram_size = 256 * 4;
    if (!this.histogram_buffer || this.histogram_buffer.config.size < histogram_size) {
      this.histogram_buffer = Buffer.create({
        name: "bvh_histogram_buffer",
        usage: storage_usage,
        size: histogram_size,
        force: true,
      });
      Renderer.get().mark_bind_groups_dirty(true);
    }

    const required_bvh2_size = this.bvh_size * 2 * BVH2_NODE_BYTE_SIZE;

    if (!this.bvh2_nodes_buffer || this.bvh2_nodes_buffer.config.size < required_bvh2_size) {
      this.bvh2_nodes_buffer = Buffer.create({
        name: BVH2_NODES_BUFFER_NAME,
        usage: storage_usage,
        size: required_bvh2_size,
        force: true,
      });

      Renderer.get().mark_bind_groups_dirty(true);
    }

    const required_bvh4_size = this.bvh_size * 2 * BVH4_NODE_BYTE_SIZE;

    if (!this.bvh4_nodes_buffer || this.bvh4_nodes_buffer.config.size < required_bvh4_size) {
      this.bvh4_nodes_buffer = Buffer.create({
        name: BVH4_NODES_BUFFER_NAME,
        usage: storage_usage,
        size: required_bvh4_size,
        force: true,
      });

      Renderer.get().mark_bind_groups_dirty(true);
    }

    // H-PLOC counters and cluster buffers
    // Combined counters: two u32 values (bvh2_count, bvh4_count)
    if (!this.node_counters_buffer) {
      this.node_counters_buffer = Buffer.create({
        name: "bvh_node_counters",
        usage: storage_usage | GPUBufferUsage.UNIFORM,
        size: 8,
        force: true,
      });
    }

    const cluster_stride = 8 * 4; // 2x vec4<f32>
    const clusters_required_size = Math.max(this.bvh_size, DEFAULT_BVH_SIZE) * cluster_stride;
    if (!this.clusters_in_buffer || this.clusters_in_buffer.config.size < clusters_required_size) {
      this.clusters_in_buffer = Buffer.create({
        name: CLUSTERS_IN_BUFFER_NAME,
        usage: storage_usage,
        size: clusters_required_size,
        force: true,
      });
    }
    if (!this.clusters_out_buffer || this.clusters_out_buffer.config.size < clusters_required_size) {
      this.clusters_out_buffer = Buffer.create({
        name: CLUSTERS_OUT_BUFFER_NAME,
        usage: storage_usage,
        size: clusters_required_size,
        force: true,
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
    bounds_buffer: null,
    user_data_buffer: null,
    scene_bounds_buffer: null,
    morton_codes_buffer: null,
    temp_morton_codes_buffer: null,
    sorted_indices_buffer: null,
    temp_sorted_indices_buffer: null,
    bvh2_nodes_buffer: null,
    bvh4_nodes_buffer: null,
    histogram_buffer: null,
  };

  /**
   * Get GPU data for binding to shaders (and rebuild buffers if needed)
   * @returns {Object} - Object containing GPU buffers
   */
  static to_gpu_data() {
    this.rebuild_buffers();

    this.#data_buffers.bounds_buffer = this.bounds_buffer;
    this.#data_buffers.user_data_buffer = this.user_data_buffer;
    this.#data_buffers.scene_bounds_buffer = this.scene_bounds_buffer;
    this.#data_buffers.morton_codes_buffer = this.morton_codes_buffer;
    this.#data_buffers.temp_morton_codes_buffer = this.temp_morton_codes_buffer;
    this.#data_buffers.sorted_indices_buffer = this.sorted_indices_buffer;
    this.#data_buffers.temp_sorted_indices_buffer = this.temp_sorted_indices_buffer;
    this.#data_buffers.bvh2_nodes_buffer = this.bvh2_nodes_buffer;
    this.#data_buffers.bvh4_nodes_buffer = this.bvh4_nodes_buffer;
    this.#data_buffers.histogram_buffer = this.histogram_buffer;
    this.#data_buffers.node_counters_buffer = this.node_counters_buffer;
    this.#data_buffers.clusters_in_buffer = this.clusters_in_buffer;
    this.#data_buffers.clusters_out_buffer = this.clusters_out_buffer;

    return this.#data_buffers;
  }
}
