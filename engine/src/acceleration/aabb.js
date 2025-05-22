import { MAX_BUFFERED_FRAMES } from "../core/minimal.js";
import { Renderer } from "../renderer/renderer.js";
import { Buffer, BufferSync } from "../renderer/buffer.js";
import { global_dispatcher } from "../core/dispatcher.js";
import { RingBufferAllocator } from "../memory/allocator.js";
import { TypedQueue, TypedStack } from "../memory/container.js";
import { npot } from "../utility/math.js";

// Buffer names for GPU access
const AABB_TREE_NODES_BUFFER_NAME = "aabb_tree_nodes_buffer";
const AABB_TREE_NODES_BOUNDS_BUFFER_NAME = "aabb_tree_nodes_bounds_buffer";

// Events for notifying systems about buffer updates
const AABB_TREE_NODES_EVENT = "aabb_tree_nodes";
const AABB_TREE_NODES_UPDATE_EVENT = "aabb_tree_nodes_update";
const AABB_NODE_BOUNDS_UPDATE_EVENT = "aabb_node_bounds_update";

// Node structure in memory:
// uint4 flags_and_node_data (flags, node_type, padding, padding)
// uint4 node_data (left, right, parent, user_data)
const NODE_SIZE = 8; // Size in float32 elements

// float4 min_point (xyz + additional_data as w)
// float4 max_point (xyz + additional_data as w)
const NODE_BOUNDS_SIZE = 8; // Size in float32 elements

// float4 min_point (xyz + additional_data as w)
// float4 max_point (xyz + additional_data as w)
const NODE_FAT_BOUNDS_SIZE = 8; // Size in float32 elements

// Node types
export const AABB_NODE_TYPE = {
  INTERNAL: 0,
  LEAF: 1,
};

// Node flags
export const AABB_NODE_FLAGS = {
  FREE: 1 << 0,
};

/**
 * View class for accessing AABB node data
 */
class AABBNodeDataView {
  current_node = -1;

  constructor() {}

  get min_point() {
    return [
      AABB.node_bounds[this.current_node * NODE_BOUNDS_SIZE],
      AABB.node_bounds[this.current_node * NODE_BOUNDS_SIZE + 1],
      AABB.node_bounds[this.current_node * NODE_BOUNDS_SIZE + 2],
    ];
  }

  set min_point(value) {
    AABB.node_bounds[this.current_node * NODE_BOUNDS_SIZE] = value[0];
    AABB.node_bounds[this.current_node * NODE_BOUNDS_SIZE + 1] = value[1];
    AABB.node_bounds[this.current_node * NODE_BOUNDS_SIZE + 2] = value[2];
    AABB.modified = true;
  }

  get max_point() {
    return [
      AABB.node_bounds[this.current_node * NODE_BOUNDS_SIZE + 4],
      AABB.node_bounds[this.current_node * NODE_BOUNDS_SIZE + 5],
      AABB.node_bounds[this.current_node * NODE_BOUNDS_SIZE + 6],
    ];
  }

  set max_point(value) {
    AABB.node_bounds[this.current_node * NODE_BOUNDS_SIZE + 4] = value[0];
    AABB.node_bounds[this.current_node * NODE_BOUNDS_SIZE + 5] = value[1];
    AABB.node_bounds[this.current_node * NODE_BOUNDS_SIZE + 6] = value[2];
    AABB.modified = true;
  }

  get flags() {
    return AABB.node_data[this.current_node * NODE_SIZE + 0];
  }

  set flags(value) {
    AABB.node_data[this.current_node * NODE_SIZE + 0] = value;
    AABB.modified = true;
  }

  get node_type() {
    return AABB.node_data[this.current_node * NODE_SIZE + 1];
  }

  set node_type(value) {
    AABB.node_data[this.current_node * NODE_SIZE + 1] = value;
    AABB.modified = true;
  }

  get left() {
    return AABB.node_data[this.current_node * NODE_SIZE + 4];
  }

  set left(value) {
    AABB.node_data[this.current_node * NODE_SIZE + 4] = value;
    AABB.modified = true;
  }

  get right() {
    return AABB.node_data[this.current_node * NODE_SIZE + 5];
  }

  set right(value) {
    AABB.node_data[this.current_node * NODE_SIZE + 5] = value;
    AABB.modified = true;
  }

  get parent() {
    return AABB.node_data[this.current_node * NODE_SIZE + 6];
  }

  set parent(value) {
    AABB.node_data[this.current_node * NODE_SIZE + 6] = value;
    AABB.modified = true;
  }

  get user_data() {
    return AABB.node_data[this.current_node * NODE_SIZE + 7];
  }

  set user_data(value) {
    AABB.node_data[this.current_node * NODE_SIZE + 7] = value;
    AABB.modified = true;
  }

  get height() {
    return AABB.node_heights[this.current_node];
  }

  set height(value) {
    AABB.node_heights[this.current_node] = value;
    AABB.modified = true;
  }

  get fat_min_point() {
    return [
      AABB.node_fat_bounds[this.current_node * NODE_FAT_BOUNDS_SIZE],
      AABB.node_fat_bounds[this.current_node * NODE_FAT_BOUNDS_SIZE + 1],
      AABB.node_fat_bounds[this.current_node * NODE_FAT_BOUNDS_SIZE + 2],
    ];
  }

  set fat_min_point(value) {
    AABB.node_fat_bounds[this.current_node * NODE_FAT_BOUNDS_SIZE] = value[0];
    AABB.node_fat_bounds[this.current_node * NODE_FAT_BOUNDS_SIZE + 1] = value[1];
    AABB.node_fat_bounds[this.current_node * NODE_FAT_BOUNDS_SIZE + 2] = value[2];
    AABB.modified = true;
  }

  get fat_max_point() {
    return [
      AABB.node_fat_bounds[this.current_node * NODE_FAT_BOUNDS_SIZE + 4],
      AABB.node_fat_bounds[this.current_node * NODE_FAT_BOUNDS_SIZE + 5],
      AABB.node_fat_bounds[this.current_node * NODE_FAT_BOUNDS_SIZE + 6],
    ];
  }

  set fat_max_point(value) {
    AABB.node_fat_bounds[this.current_node * NODE_FAT_BOUNDS_SIZE + 4] = value[0];
    AABB.node_fat_bounds[this.current_node * NODE_FAT_BOUNDS_SIZE + 5] = value[1];
    AABB.node_fat_bounds[this.current_node * NODE_FAT_BOUNDS_SIZE + 6] = value[2];
    AABB.modified = true;
  }

  view_node(node_index) {
    this.current_node = node_index;
    return this;
  }
}

const unmapped_state = "unmapped";
const default_aabb_size = 1024;

/**
 * Standalone AABB tree implementation optimized for cache access patterns
 * and decoupled from entity system
 */
export class AABB {
  // Static properties
  static data_view_allocator = new RingBufferAllocator(1024, AABBNodeDataView);
  static is_initialized = false;
  static size = default_aabb_size;
  static root_node = 0;
  static allocated_count = 0;
  static free_nodes = new TypedStack(default_aabb_size, Uint32Array);
  static dirty_nodes = new TypedQueue(default_aabb_size, 0, Uint32Array);
  static node_data = new Uint32Array(NODE_SIZE * default_aabb_size);
  static node_bounds = new Float32Array(NODE_BOUNDS_SIZE * default_aabb_size);
  static node_fat_bounds = new Float32Array(NODE_FAT_BOUNDS_SIZE * default_aabb_size);
  static node_heights = new Float32Array(default_aabb_size);

  static node_data_buffer = null;
  static node_bounds_buffer = null;
  static node_bounds_cpu_buffer = Array(MAX_BUFFERED_FRAMES).fill(null);

  static modified = true;

  // Maximum number of bytes to read back per frame
  static sync_max_bytes_per_frame = npot(1000 * 8 * 4);
  // Current byte offset into the node_bounds buffer
  static sync_read_byte_offset = 0;
  // Track sync status of node_bounds on CPU
  static node_bounds_cpu_sync_status = new Uint8Array(default_aabb_size);

  /**
   * Initialize the AABB tree
   */
  static initialize() {
    if (this.is_initialized) return;

    Renderer.get().on_post_render(this.on_post_render.bind(this));

    this.free_nodes.resize(this.size);

    // Initialize new nodes as free
    for (let i = 0; i < this.size; i++) {
      const node_view = this.get_node_data(i);
      node_view.min_point = [Infinity, Infinity, Infinity];
      node_view.max_point = [-Infinity, -Infinity, -Infinity];
      node_view.fat_min_point = [Infinity, Infinity, Infinity];
      node_view.fat_max_point = [-Infinity, -Infinity, -Infinity];
      node_view.node_type = AABB_NODE_TYPE.INTERNAL;
      node_view.flags = AABB_NODE_FLAGS.FREE;
      node_view.left = 0;
      node_view.right = 0;
      node_view.parent = 0;
      node_view.user_data = 0xffffffff;
      node_view.height = 0; // Initialize height to 0 for free nodes

      this.free_nodes.push(i);
    }

    // Set up the root node
    this.reinitialize_root_node();
    this.rebuild_buffers();

    this.is_initialized = true;
  }

  /**
   * Reinitialize the root node
   */
  static reinitialize_root_node() {
    // Initialize the root node with "infinite" bounds
    const root_view = this.get_node_data(0);
    root_view.min_point = [Infinity, Infinity, Infinity];
    root_view.max_point = [-Infinity, -Infinity, -Infinity];
    root_view.fat_min_point = [Infinity, Infinity, Infinity];
    root_view.fat_max_point = [-Infinity, -Infinity, -Infinity];
    root_view.node_type = AABB_NODE_TYPE.INTERNAL;
    root_view.flags = 0;
    root_view.left = 0;
    root_view.right = 0;
    root_view.parent = 0;
    root_view.user_data = 0xffffffff; // Invalid user data
    root_view.height = 0;
  }

  /**
   * Resize the AABB tree to accommodate more nodes
   * @param {number} new_size - The new size of the tree
   */
  static resize(new_size) {
    if (new_size <= this.size) return;

    const old_size = this.size + 1; // old_size was this.size, it should be this.size before update
    // const old_size = this.size; // Correct variable to use for slicing old arrays

    this.size = new_size + 1;

    {
      const new_node_data = new Uint32Array(this.size * NODE_SIZE);

      if (this.node_data) {
        new_node_data.set(this.node_data);
      }

      this.node_data = new_node_data;
    }

    {
      const new_node_bounds = new Float32Array(this.size * NODE_BOUNDS_SIZE);

      if (this.node_bounds) {
        new_node_bounds.set(this.node_bounds);
      }

      this.node_bounds = new_node_bounds;
    }

    {
      const new_node_fat_bounds = new Float32Array(this.size * NODE_FAT_BOUNDS_SIZE);

      if (this.node_fat_bounds) {
        new_node_fat_bounds.set(this.node_fat_bounds);
      }

      this.node_fat_bounds = new_node_fat_bounds;
    }

    {
      const new_node_heights = new Float32Array(this.size);

      if (this.node_heights) {
        this.node_heights.set(this.node_heights);
      }

      this.node_heights = new_node_heights;
    }

    {
      const new_sync_status = new Uint8Array(this.size);

      if (this.node_bounds_cpu_sync_status) {
        new_sync_status.set(this.node_bounds_cpu_sync_status);
      }

      this.node_bounds_cpu_sync_status = new_sync_status;
    }

    this.free_nodes.resize(this.size);

    // Initialize new nodes as free
    for (let i = old_size; i < this.size; i++) {
      const node_view = this.get_node_data(i);
      node_view.min_point = [Infinity, Infinity, Infinity];
      node_view.max_point = [-Infinity, -Infinity, -Infinity];
      node_view.fat_min_point = [Infinity, Infinity, Infinity];
      node_view.fat_max_point = [-Infinity, -Infinity, -Infinity];
      node_view.node_type = AABB_NODE_TYPE.INTERNAL;
      node_view.flags = AABB_NODE_FLAGS.FREE;
      node_view.left = 0;
      node_view.right = 0;
      node_view.parent = 0;
      node_view.user_data = 0xffffffff;
      node_view.height = 0; // Initialize height to 0 for free nodes

      this.free_nodes.push(i);
    }

    this.modified = true;
  }

  /**
   * Allocate a new node
   * @param {number} user_data - Optional user data to associate with the node
   * @returns {number} - The index of the new node
   */
  static allocate_node(user_data = 0xffffffff) {
    if (this.free_nodes.length === 0) {
      this.resize(this.size * 2 + 1);
    }

    const node_index = this.free_nodes.pop();
    const node_view = this.get_node_data(node_index);

    node_view.min_point = [Infinity, Infinity, Infinity];
    node_view.max_point = [-Infinity, -Infinity, -Infinity];
    node_view.fat_min_point = [Infinity, Infinity, Infinity];
    node_view.fat_max_point = [-Infinity, -Infinity, -Infinity];
    node_view.node_type = AABB_NODE_TYPE.LEAF;
    node_view.flags = 0;
    node_view.left = 0;
    node_view.right = 0;
    node_view.parent = 0;
    node_view.user_data = user_data;
    node_view.height = 1; // Initialize height to 1 for leaf nodes

    ++this.allocated_count;

    this.modified = true;

    if (node_index < this.node_bounds_cpu_sync_status.length) {
      this.node_bounds_cpu_sync_status[node_index] = 0; // New node, bounds not from GPU yet
    }

    return node_index;
  }

  /**
   * Free a node
   * @param {number} node_index - The index of the node to free
   */
  static free_node(node_index) {
    if (node_index <= 0 || node_index >= this.size) return;

    const node_view = this.get_node_data(node_index);

    // Check if the node is already free
    if ((node_view.flags & AABB_NODE_FLAGS.FREE) !== 0) {
      return; // Node is already free
    }

    node_view.min_point = [Infinity, Infinity, Infinity];
    node_view.max_point = [-Infinity, -Infinity, -Infinity];
    node_view.fat_min_point = [Infinity, Infinity, Infinity];
    node_view.fat_max_point = [-Infinity, -Infinity, -Infinity];
    node_view.node_type = AABB_NODE_TYPE.INTERNAL;
    node_view.flags = AABB_NODE_FLAGS.FREE;
    node_view.left = 0;
    node_view.right = 0;
    node_view.parent = 0;
    node_view.user_data = 0xffffffff;
    node_view.height = 0; // Reset height to 0 for free nodes

    this.free_nodes.push(node_index);

    --this.allocated_count;

    this.modified = true;
  }

  /**
   * Get a data view for a node
   * @param {number} node_index - The index of the node
   * @returns {AABBNodeDataView} - A data view for the node
   */
  static get_node_data(node_index) {
    const data_view = this.data_view_allocator.allocate();
    data_view.view_node(node_index);
    return data_view;
  }

  /**
   * Update a node's bounds
   * @param {number} node_index - The index of the node
   * @param {Array<number>} min_point - The minimum point of the AABB
   * @param {Array<number>} max_point - The maximum point of the AABB
   */
  static update_node_bounds(node_index, min_point, max_point) {
    if (node_index <= 0 || node_index >= this.size) return;

    const node_view = this.get_node_data(node_index);
    if ((node_view.flags & AABB_NODE_FLAGS.FREE) === 0) {
      node_view.min_point = [min_point[0], min_point[1], min_point[2]];
      node_view.max_point = [max_point[0], max_point[1], max_point[2]];
    }
  }

  /**
   * Rebuild GPU buffers if needed
   */
  static rebuild_buffers() {
    if (!this.modified) return;

    if (!this.node_data_buffer || this.node_data_buffer.config.size < this.node_data.byteLength) {
      this.node_data_buffer = Buffer.create({
        name: AABB_TREE_NODES_BUFFER_NAME,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        raw_data: this.node_data,
        force: true,
      });

      global_dispatcher.dispatch(AABB_TREE_NODES_EVENT, this.node_data_buffer);

      Renderer.get().mark_bind_groups_dirty(true);
    } else {
      this.node_data_buffer.write(this.node_data);
    }

    global_dispatcher.dispatch(AABB_TREE_NODES_UPDATE_EVENT);

    if (
      !this.node_bounds_buffer ||
      this.node_bounds_buffer.config.size < this.node_bounds.byteLength
    ) {
      this.node_bounds_buffer = Buffer.create({
        name: AABB_TREE_NODES_BOUNDS_BUFFER_NAME,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        raw_data: this.node_bounds,
        force: true,
      });

      for (let i = 0; i < MAX_BUFFERED_FRAMES; i++) {
        this.node_bounds_cpu_buffer[i] = Buffer.create({
          name: `AABB_TREE_NODES_CPU_BOUNDS_BUFFER_${i}`,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
          raw_data: this.node_bounds,
          force: true,
        });
      }

      global_dispatcher.dispatch(AABB_TREE_NODES_UPDATE_EVENT);
    } else {
      this.node_bounds_buffer.write(this.node_bounds);
    }

    global_dispatcher.dispatch(AABB_NODE_BOUNDS_UPDATE_EVENT);

    this.modified = false;
  }

  /**
   * Get GPU data for binding to shaders
   * @returns {Object} - Object containing GPU buffers
   */
  static #data_buffers = {
    node_data_buffer: null,
    node_bounds_buffer: null,
  };

  static to_gpu_data() {
    this.rebuild_buffers();

    this.#data_buffers.node_data_buffer = this.node_data_buffer;
    this.#data_buffers.node_bounds_buffer = this.node_bounds_buffer;

    return this.#data_buffers;
  }

  /**
   * Sync the AABB tree buffers (throttled)
   */
  static async sync_buffers() {
    const buffered_frame = Renderer.get().get_buffered_frame_number();
    const cpu_buffer = AABB.node_bounds_cpu_buffer[buffered_frame];
    if (!cpu_buffer || cpu_buffer.buffer.mapState !== unmapped_state) {
      return;
    }

    const total_bytes_in_buffer = cpu_buffer.config.size; // Total size of AABB.node_bounds buffer
    const current_byte_offset = AABB.sync_read_byte_offset;

    if (current_byte_offset >= total_bytes_in_buffer && total_bytes_in_buffer > 0) {
      // Ensure offset resets if it went past
      AABB.sync_read_byte_offset = 0;
      // current_byte_offset = 0; // No, use the reset value in next iteration. For now, just return if fully synced this cycle.
      return; // Or handle wrap-around logic more gracefully if needed for continuous sync
    }

    const element_offset_in_cpu_array = current_byte_offset / Float32Array.BYTES_PER_ELEMENT;
    const bytes_to_read_this_call = Math.min(
      AABB.sync_max_bytes_per_frame,
      total_bytes_in_buffer - current_byte_offset
    );

    if (bytes_to_read_this_call <= 0) return;

    await cpu_buffer.read(
      AABB.node_bounds, // Target CPU array
      bytes_to_read_this_call,
      current_byte_offset, // Source offset in GPU buffer (implicit for copy_buffer, explicit for read command)
      element_offset_in_cpu_array // Target offset in CPU array AABB.node_bounds
    );

    // Update sync status for the copied region
    const num_elements_read = bytes_to_read_this_call / Float32Array.BYTES_PER_ELEMENT;
    const start_node_index_synced = element_offset_in_cpu_array / NODE_BOUNDS_SIZE;
    const num_nodes_synced = num_elements_read / NODE_BOUNDS_SIZE;

    for (let i = 0; i < num_nodes_synced; i++) {
      const synced_node_idx = Math.floor(start_node_index_synced) + i;
      if (
        synced_node_idx < AABB.size &&
        synced_node_idx < AABB.node_bounds_cpu_sync_status.length
      ) {
        AABB.node_bounds_cpu_sync_status[synced_node_idx] = 1;
      }
    }

    AABB.sync_read_byte_offset = (current_byte_offset + bytes_to_read_this_call) % total_bytes_in_buffer;
  }

  static async on_post_render() {
    BufferSync.request_sync(this);
  }

  static is_node_free(node_index) {
    if (node_index <= 0 || node_index >= this.size) return false;
    const node_view = this.get_node_data(node_index);
    return (node_view.flags & AABB_NODE_FLAGS.FREE) !== 0;
  }

  /**
   * Calculate the surface area of an AABB
   * @param {Array<number>} min_point - The minimum point of the AABB
   * @param {Array<number>} max_point - The maximum point of the AABB
   * @returns {number} - The surface area of the AABB
   */
  static calculate_aabb_surface_area(min_point, max_point) {
    const width = max_point[0] - min_point[0];
    const height = max_point[1] - min_point[1];
    const depth = max_point[2] - min_point[2];

    return 2.0 * (width * height + width * depth + height * depth);
  }

  /**
   * Merge two AABBs
   * @param {Array<number>} a_min - The minimum point of the first AABB
   * @param {Array<number>} a_max - The maximum point of the first AABB
   * @param {Array<number>} b_min - The minimum point of the second AABB
   * @param {Array<number>} b_max - The maximum point of the second AABB
   * @returns {Object} - The merged AABB with min and max properties
   */
  static #temp_merged_aabbs = { min: [0.0, 0.0, 0.0], max: [0.0, 0.0, 0.0] };
  static merge_aabbs(a_min, a_max, b_min, b_max) {
    this.#temp_merged_aabbs.min = [
      Math.min(a_min[0], b_min[0]),
      Math.min(a_min[1], b_min[1]),
      Math.min(a_min[2], b_min[2]),
    ];
    this.#temp_merged_aabbs.max = [
      Math.max(a_max[0], b_max[0]),
      Math.max(a_max[1], b_max[1]),
      Math.max(a_max[2], b_max[2]),
    ];
    return this.#temp_merged_aabbs;
  }

  /**
   * Calculate a fat margin for an AABB
   * @param {Array<number>} min_point - The minimum point of the AABB
   * @param {Array<number>} max_point - The maximum point of the AABB
   * @param {number} fat_margin_factor - The factor to use for the fat margin
   * @returns {Object} - The fat AABB with min and max properties
   */
  static calculate_fat_margin(min_point, max_point, fat_margin_factor = 0.1) {
    const width = max_point[0] - min_point[0];
    const height = max_point[1] - min_point[1];
    const depth = max_point[2] - min_point[2];

    const margin_x = width * fat_margin_factor;
    const margin_y = height * fat_margin_factor;
    const margin_z = depth * fat_margin_factor;

    return {
      min: [min_point[0] - margin_x, min_point[1] - margin_y, min_point[2] - margin_z],
      max: [max_point[0] + margin_x, max_point[1] + margin_y, max_point[2] + margin_z],
    };
  }
}
