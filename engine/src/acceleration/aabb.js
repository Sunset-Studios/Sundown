import { Renderer } from "../renderer/renderer.js";
import { Buffer } from "../renderer/buffer.js";
import { TransformFragment } from "../core/ecs/fragments/transform_fragment.js";
import { global_dispatcher } from "../core/dispatcher.js";
import { RingBufferAllocator } from "../memory/allocator.js";
import { TypedQueue, TypedStack } from "../memory/container.js";

// Buffer names for GPU access
const AABB_TREE_NODES_BUFFER_NAME = "aabb_tree_nodes_buffer";
const AABB_TREE_NODES_BOUNDS_BUFFER_NAME = "aabb_tree_nodes_bounds_buffer";
const AABB_TREE_NODES_CPU_BOUNDS_BUFFER_NAME = "aabb_tree_nodes_cpu_bounds_buffer";

// Events for notifying systems about buffer updates
const AABB_TREE_NODES_EVENT = "aabb_tree_nodes";
const AABB_TREE_NODES_UPDATE_EVENT = "aabb_tree_nodes_update";
const AABB_NODE_BOUNDS_UPDATE_EVENT = "aabb_node_bounds_update";

// Node structure in memory:
// float4 flags_and_node_data (flags, node_type, padding, padding)
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
  MOVED: 1 << 0,
  FREE: 1 << 1,
};

/**
 * View class for accessing AABB node data
 */
class AABBNodeDataView {
  current_node = -1;

  constructor() {}

  get min_point() {
    return [
      AABB.data.node_bounds[this.current_node * NODE_BOUNDS_SIZE],
      AABB.data.node_bounds[this.current_node * NODE_BOUNDS_SIZE + 1],
      AABB.data.node_bounds[this.current_node * NODE_BOUNDS_SIZE + 2],
    ];
  }

  set min_point(value) {
    AABB.data.node_bounds[this.current_node * NODE_BOUNDS_SIZE] = value[0];
    AABB.data.node_bounds[this.current_node * NODE_BOUNDS_SIZE + 1] = value[1];
    AABB.data.node_bounds[this.current_node * NODE_BOUNDS_SIZE + 2] = value[2];
    AABB.data.modified = true;
  }

  get max_point() {
    return [
      AABB.data.node_bounds[this.current_node * NODE_BOUNDS_SIZE + 4],
      AABB.data.node_bounds[this.current_node * NODE_BOUNDS_SIZE + 5],
      AABB.data.node_bounds[this.current_node * NODE_BOUNDS_SIZE + 6],
    ];
  }

  set max_point(value) {
    AABB.data.node_bounds[this.current_node * NODE_BOUNDS_SIZE + 4] = value[0];
    AABB.data.node_bounds[this.current_node * NODE_BOUNDS_SIZE + 5] = value[1];
    AABB.data.node_bounds[this.current_node * NODE_BOUNDS_SIZE + 6] = value[2];
    AABB.data.modified = true;
  }

  get flags() {
    return AABB.data.node_data[this.current_node * NODE_SIZE + 0];
  }

  set flags(value) {
    AABB.data.node_data[this.current_node * NODE_SIZE + 0] = value;
    AABB.data.modified = true;
  }

  get node_type() {
    return AABB.data.node_data[this.current_node * NODE_SIZE + 1];
  }

  set node_type(value) {
    AABB.data.node_data[this.current_node * NODE_SIZE + 1] = value;
    AABB.data.modified = true;
  }

  get left() {
    return AABB.data.node_data[this.current_node * NODE_SIZE + 4];
  }

  set left(value) {
    AABB.data.node_data[this.current_node * NODE_SIZE + 4] = value;
    AABB.data.modified = true;
  }

  get right() {
    return AABB.data.node_data[this.current_node * NODE_SIZE + 5];
  }

  set right(value) {
    AABB.data.node_data[this.current_node * NODE_SIZE + 5] = value;
    AABB.data.modified = true;
  }

  get parent() {
    return AABB.data.node_data[this.current_node * NODE_SIZE + 6];
  }

  set parent(value) {
    AABB.data.node_data[this.current_node * NODE_SIZE + 6] = value;
    AABB.data.modified = true;
  }

  get user_data() {
    return AABB.data.node_data[this.current_node * NODE_SIZE + 7];
  }

  set user_data(value) {
    AABB.data.node_data[this.current_node * NODE_SIZE + 7] = value;
    AABB.data.modified = true;
  }

  get height() {
    return AABB.data.node_heights[this.current_node];
  }

  set height(value) {
    AABB.data.node_heights[this.current_node] = value;
    AABB.data.modified = true;
  }

  get fat_min_point() {
    return [
      AABB.data.node_fat_bounds[this.current_node * NODE_FAT_BOUNDS_SIZE],
      AABB.data.node_fat_bounds[this.current_node * NODE_FAT_BOUNDS_SIZE + 1],
      AABB.data.node_fat_bounds[this.current_node * NODE_FAT_BOUNDS_SIZE + 2],
    ];
  }

  set fat_min_point(value) {
    AABB.data.node_fat_bounds[this.current_node * NODE_FAT_BOUNDS_SIZE] = value[0];
    AABB.data.node_fat_bounds[this.current_node * NODE_FAT_BOUNDS_SIZE + 1] = value[1];
    AABB.data.node_fat_bounds[this.current_node * NODE_FAT_BOUNDS_SIZE + 2] = value[2];
    AABB.data.modified = true;
  }

  get fat_max_point() {
    return [
      AABB.data.node_fat_bounds[this.current_node * NODE_FAT_BOUNDS_SIZE + 4],
      AABB.data.node_fat_bounds[this.current_node * NODE_FAT_BOUNDS_SIZE + 5],
      AABB.data.node_fat_bounds[this.current_node * NODE_FAT_BOUNDS_SIZE + 6],
    ];
  }

  set fat_max_point(value) {
    AABB.data.node_fat_bounds[this.current_node * NODE_FAT_BOUNDS_SIZE + 4] = value[0];
    AABB.data.node_fat_bounds[this.current_node * NODE_FAT_BOUNDS_SIZE + 5] = value[1];
    AABB.data.node_fat_bounds[this.current_node * NODE_FAT_BOUNDS_SIZE + 6] = value[2];
    AABB.data.modified = true;
  }

  view_node(node_index) {
    this.current_node = node_index;
    return this;
  }
}

const unmapped_state = "unmapped";

/**
 * Standalone AABB tree implementation optimized for cache access patterns
 * and decoupled from entity system
 */
export class AABB {
  // Static properties
  static data_view_allocator = new RingBufferAllocator(1024, AABBNodeDataView);
  static size = 1024;
  static data = null;
  static root_node = 0;
  static free_nodes = new TypedStack(1024, Uint32Array);
  static dirty_nodes = new TypedQueue(1024, 0, Uint32Array);
  static allocated_count = 0;

  /**
   * Initialize the AABB tree
   */
  static initialize() {
    this.data = {
      node_data: new Float32Array(NODE_SIZE),
      node_bounds: new Float32Array(NODE_BOUNDS_SIZE),
      node_fat_bounds: new Float32Array(NODE_FAT_BOUNDS_SIZE),
      node_heights: new Float32Array(1),
      node_data_buffer: null,
      node_bounds_buffer: null,
      node_bounds_cpu_buffer: null,
      modified: true,
    };

    Renderer.get().on_post_render(this.on_post_render.bind(this));

    this.free_nodes.resize(this.size);

    // Initialize new nodes as free
    for (let i = 0; i < this.size; i++) {
      const node_view = this.get_node_data(i);
      node_view.min_point = [Infinity, Infinity, Infinity];
      node_view.max_point = [-Infinity, -Infinity, -Infinity];
      node_view.fat_min_point = [Infinity, Infinity, Infinity];
      node_view.fat_max_point = [-Infinity, -Infinity, -Infinity];
      node_view.node_type = AABB_NODE_TYPE.LEAF;
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
  }

  /**
   * Reinitialize the root node
   */
  static reinitialize_root_node() {
    if (!this.data) this.initialize();

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
    if (!this.data) this.initialize();

    if (new_size <= this.size) return;

    const old_size = this.size + 1;
    this.size = new_size + 1;

    {
      // Create a new array with the new size
      const new_node_data = new Float32Array(this.size * NODE_SIZE);

      // Copy existing data
      if (this.data.node_data) {
        new_node_data.set(this.data.node_data);
      }

      // Replace the old array
      this.data.node_data = new_node_data;
    }

    {
      // Create a new array with the new size
      const new_node_bounds = new Float32Array(this.size * NODE_BOUNDS_SIZE);

      // Copy existing bounds
      if (this.data.node_bounds) {
        new_node_bounds.set(this.data.node_bounds);
      }

      // Replace the old array
      this.data.node_bounds = new_node_bounds;
    }

    {
      // Create a new array with the new size
      const new_node_fat_bounds = new Float32Array(this.size * NODE_FAT_BOUNDS_SIZE);

      // Copy existing bounds
      if (this.data.node_fat_bounds) {
        new_node_fat_bounds.set(this.data.node_fat_bounds);
      }

      // Replace the old array
      this.data.node_fat_bounds = new_node_fat_bounds;
    }

    {
      // Create a new array with the new size
      const new_node_heights = new Float32Array(this.size);

      // Copy existing heights
      if (this.data.node_heights) {
        this.data.node_heights.set(this.data.node_heights);
      }

      // Replace the old array
      this.data.node_heights = new_node_heights;
    }

    {
      // Create a new array with the new size
      const new_temp_sync_buffer = new Float32Array(this.size * NODE_BOUNDS_SIZE);

      // Copy existing data
      if (this.data.node_bounds) {
        new_temp_sync_buffer.set(this.data.node_bounds);
      }

      // Replace the old array
      this.#temp_sync_buffer = new_temp_sync_buffer;
    }

    this.free_nodes.resize(this.size);

    // Initialize new nodes as free
    for (let i = old_size; i < this.size; i++) {
      const node_view = this.get_node_data(i);
      node_view.min_point = [Infinity, Infinity, Infinity];
      node_view.max_point = [-Infinity, -Infinity, -Infinity];
      node_view.fat_min_point = [Infinity, Infinity, Infinity];
      node_view.fat_max_point = [-Infinity, -Infinity, -Infinity];
      node_view.node_type = AABB_NODE_TYPE.LEAF;
      node_view.flags = AABB_NODE_FLAGS.FREE;
      node_view.left = 0;
      node_view.right = 0;
      node_view.parent = 0;
      node_view.user_data = 0xffffffff;
      node_view.height = 0; // Initialize height to 0 for free nodes

      this.free_nodes.push(i);
    }

    this.data.modified = true;
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

    this.data.modified = true;

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
    node_view.node_type = AABB_NODE_TYPE.LEAF;
    node_view.flags = AABB_NODE_FLAGS.FREE;
    node_view.left = 0;
    node_view.right = 0;
    node_view.parent = 0;
    node_view.user_data = 0xffffffff;
    node_view.height = 0; // Reset height to 0 for free nodes

    this.free_nodes.push(node_index);

    --this.allocated_count;

    this.data.modified = true;
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
      let flags = node_view.flags;
      flags |= AABB_NODE_FLAGS.MOVED;
      node_view.flags = flags;
    }
  }

  /**
   * Rebuild GPU buffers if needed
   */
  static rebuild_buffers() {
    if (!this.data.modified) return;

    let retry = this.#synching;

    if (
      !this.data.node_data_buffer ||
      this.data.node_data_buffer.config.size < this.data.node_data.byteLength
    ) {
      this.data.node_data_buffer = Buffer.create({
        name: AABB_TREE_NODES_BUFFER_NAME,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        raw_data: this.data.node_data,
        force: true,
      });

      global_dispatcher.dispatch(AABB_TREE_NODES_EVENT, this.data.node_data_buffer);

      Renderer.get().mark_bind_groups_dirty(true);
    } else {
      this.data.node_data_buffer.write(this.data.node_data);
    }

    global_dispatcher.dispatch(AABB_TREE_NODES_UPDATE_EVENT);

    if (
      !this.data.node_bounds_buffer ||
      this.data.node_bounds_buffer.config.size < this.data.node_bounds.byteLength
    ) {
      this.data.node_bounds_buffer = Buffer.create({
        name: AABB_TREE_NODES_BOUNDS_BUFFER_NAME,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        raw_data: this.data.node_bounds,
        force: true,
      });

      this.data.node_bounds_cpu_buffer = Buffer.create({
        name: AABB_TREE_NODES_CPU_BOUNDS_BUFFER_NAME,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        raw_data: this.data.node_bounds,
        force: true,
      });

      global_dispatcher.dispatch(AABB_TREE_NODES_UPDATE_EVENT);
    } else if (!retry) {
      this.data.node_bounds_buffer.write(this.data.node_bounds);
    } 

    global_dispatcher.dispatch(AABB_NODE_BOUNDS_UPDATE_EVENT);

    this.data.modified = retry;
  }

  /**
   * Get GPU data for binding to shaders
   * @returns {Object} - Object containing GPU buffers
   */
  static #data_buffers = {
    node_data_buffer: null,
    node_bounds_buffer: null,
  }
  static to_gpu_data() {
    if (!this.data) {
      this.initialize();
    }

    this.rebuild_buffers();

    this.#data_buffers.node_data_buffer = this.data.node_data_buffer;
    this.#data_buffers.node_bounds_buffer = this.data.node_bounds_buffer;

    return this.#data_buffers;
  }

  /**
   * Sync the AABB tree buffers
   */
  static #temp_sync_buffer = new Float32Array(NODE_BOUNDS_SIZE);
  static #synching = false;
  static async sync_buffers() {
    if (this.#synching) return;
    // Only do readbacks if the buffer is ready and not being modified
    if (this.data.node_bounds_cpu_buffer?.buffer.mapState === unmapped_state) {
      try {
        this.#synching = true;
        // Do the readback
        await this.data.node_bounds_cpu_buffer.read(
          this.#temp_sync_buffer,
          this.#temp_sync_buffer.byteLength,
          0,
          0,
          Float32Array
        );
      } finally {
        this.#synching = false;
        for (let i = 0; i < this.size; i++) {
          if (this.data.node_data[i * NODE_SIZE + 1] === AABB_NODE_TYPE.LEAF) {
            this.data.node_bounds[i * NODE_BOUNDS_SIZE + 0] = this.#temp_sync_buffer[i * NODE_BOUNDS_SIZE + 0];
            this.data.node_bounds[i * NODE_BOUNDS_SIZE + 1] = this.#temp_sync_buffer[i * NODE_BOUNDS_SIZE + 1];
            this.data.node_bounds[i * NODE_BOUNDS_SIZE + 2] = this.#temp_sync_buffer[i * NODE_BOUNDS_SIZE + 2];
            this.data.node_bounds[i * NODE_BOUNDS_SIZE + 3] = this.#temp_sync_buffer[i * NODE_BOUNDS_SIZE + 3];
            this.data.node_bounds[i * NODE_BOUNDS_SIZE + 4] = this.#temp_sync_buffer[i * NODE_BOUNDS_SIZE + 4];
            this.data.node_bounds[i * NODE_BOUNDS_SIZE + 5] = this.#temp_sync_buffer[i * NODE_BOUNDS_SIZE + 5];
            this.data.node_bounds[i * NODE_BOUNDS_SIZE + 6] = this.#temp_sync_buffer[i * NODE_BOUNDS_SIZE + 6];
            this.data.node_bounds[i * NODE_BOUNDS_SIZE + 7] = this.#temp_sync_buffer[i * NODE_BOUNDS_SIZE + 7];
          }
        }
      }
    }
  }

  static async on_post_render() {
    if (!this.data) return;
    await this.sync_buffers();
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
   * Calculate the volume of an AABB
   * @param {Array<number>} min_point - The minimum point of the AABB
   * @param {Array<number>} max_point - The maximum point of the AABB
   * @returns {number} - The volume of the AABB
   */
  static calculate_aabb_volume(min_point, max_point) {
    const width = max_point[0] - min_point[0];
    const height = max_point[1] - min_point[1];
    const depth = max_point[2] - min_point[2];

    return width * height * depth;
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

  /**
   * Compute the bounds of an entity
   * @param {bigint} entity - The entity to compute bounds for
   * @returns {Object} - The entity's bounds with min and max properties
   */
  static #bounds = { min: [0, 0, 0], max: [0, 0, 0] };
  static compute_entity_bounds(entity, instance_index, bounds_padding = 0.0) {
    // Get the entity's world transform
    const position = TransformFragment.get_world_position(entity, instance_index);
    const scale = TransformFragment.get_world_scale(entity, instance_index);

    // Calculate bounds based on position and scale
    // This is a simple axis-aligned box, but could be more sophisticated
    // based on the entity's mesh or collider
    const half_size = [
      Math.abs(scale[0]) * 0.5,
      Math.abs(scale[1]) * 0.5,
      Math.abs(scale[2]) * 0.5,
    ];

    // Add padding
    const padding = [
      half_size[0] * bounds_padding,
      half_size[1] * bounds_padding,
      half_size[2] * bounds_padding,
    ];

    this.#bounds.min = [
      position[0] - half_size[0] - padding[0],
      position[1] - half_size[1] - padding[1],
      position[2] - half_size[2] - padding[2],
    ];

    this.#bounds.max = [
      position[0] + half_size[0] + padding[0],
      position[1] + half_size[1] + padding[1],
      position[2] + half_size[2] + padding[2],
    ];

    return this.#bounds;
  }

  /**
   * Calculate the cost of combining two AABBs
   * @param {Array<number>} node_min_point - The min point of the node AABB
   * @param {Array<number>} node_max_point - The max point of the node AABB
   * @param {Array<number>} other_min_point - The min point of the other AABB
   * @param {Array<number>} other_max_point - The max point of the other AABB
   * @returns {number} - The cost of combining the AABBs
   */
  static calculate_combination_cost(
    node_min_point,
    node_max_point,
    other_min_point,
    other_max_point
  ) {
    const combined = AABB.merge_aabbs(
      node_min_point,
      node_max_point,
      other_min_point,
      other_max_point
    );

    return AABB.calculate_aabb_surface_area(combined.min, combined.max);
  }
}
