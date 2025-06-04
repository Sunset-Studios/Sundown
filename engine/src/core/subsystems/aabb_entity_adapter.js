import { Renderer } from "../../renderer/renderer.js";
import { DEFAULT_CHUNK_CAPACITY, ROW_MASK } from "../ecs/solar/types.js";
import { FragmentGpuBuffer } from "../ecs/solar/memory.js";
import { ComputeTaskQueue } from "../../renderer/compute_task_queue.js";
import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager } from "../ecs/entity.js";
import { EntityFlags } from "../minimal.js";
import { TransformFragment } from "../ecs/fragments/transform_fragment.js";
import { AABB, AABB_NODE_FLAGS } from "../../acceleration/aabb.js";
import { AABBTreeProcessor } from "../../acceleration/aabb_tree_processor.js";
import { profile_scope } from "../../utility/performance.js";

const transforms_buffer_name = "transforms";
const aabb_node_index_buffer_name = "aabb_node_index";
const entity_adapter_update_name = "aabb_entity_adapter.update";
const entity_bounds_update_task_name = "entity_bounds_update";
const entity_bounds_update_wgsl_path = "system_compute/bounds_processing.wgsl";

/**
 * Adapter that connects the entity system to the AABB tree
 * This allows entities to use the AABB tree without tightly coupling them
 */
export class AABBEntityAdapter extends SimulationLayer {
  // Entity query for objects with transforms
  entity_query = null;
  // AABB tree processor
  tree_processor = new AABBTreeProcessor();
  // Input and output lists for the entity bounds update task
  bounds_update_input_list = new Array(4);
  bounds_update_output_list = new Array(1);

  constructor() {
    super();
    this.name = "aabb_entity_adapter";
  }

  /**
   * Initialize the adapter
   */
  init() {
    // Set up entity query for objects with transforms
    this.entity_query = EntityManager.create_query([TransformFragment]);
    this._process_entity_changes_chunk_iter = this._process_entity_changes_chunk_iter.bind(this);
    EntityManager.on_delete(this._on_delete.bind(this));
  }

  /**
   * Pre-update the adapter and do any frame-specific setup in the processor
   */
  pre_update() {
    this.tree_processor.pre_update();
  }

  /**
   * Update the adapter
   * @param {number} delta_time - Time since last update
   */
  update(delta_time) {
    profile_scope(entity_adapter_update_name, async () => {
      this._process_entity_changes();
      {
        const processed_node_indices = this.tree_processor.update(delta_time);
        this._update_processed_node_flags(processed_node_indices);
      }
      this._dispatch_bounds_update();
    });
  }

  /**
   * Post-update the adapter and do any frame-specific cleanup in the processor
   */
  post_update() {
    this.tree_processor.post_update();
  }

  /**
   * Mark an entity as dirty (needs update)
   * @param {bigint} entity - The entity to mark as dirty
   */
  mark_entity_dirty(entity) {
    const entity_instances = EntityManager.get_instance_count(entity);
    for (let i = 0; i < entity_instances; i++) {
      const transform_fragment = EntityManager.get_fragment(entity, TransformFragment, i);
      if (!transform_fragment) continue;
      this.tree_processor.mark_node_dirty(transform_fragment.aabb_node_index);
    }
  }

  /**
   * Get the AABB node index for an entity
   * @param {bigint} entity - The entity to get the node for
   * @returns {number|undefined} - The node index or undefined if not found
   */
  get_entity_node_index(entity) {
    const transform_fragment = EntityManager.get_fragment(entity, TransformFragment);
    if (!transform_fragment) return null;

    return transform_fragment.aabb_node_index;
  }

  /**
   * Get the AABB bounds for an entity
   * @param {bigint} entity - The entity to get bounds for
   * @returns {Object|null} - The entity's bounds or null if not found
   */
  get_entity_bounds(entity) {
    const transform_fragment = EntityManager.get_fragment(entity, TransformFragment);
    if (!transform_fragment) return null;

    const node_index = transform_fragment.aabb_node_index;

    if (!node_index || node_index === 0) return null;

    const node_view = AABB.get_node_data(node_index);

    // Check if the node is valid
    if (!node_view || (node_view.flags & AABB_NODE_FLAGS.FREE) !== 0) {
      return null;
    }

    return {
      min: node_view.min_point,
      max: node_view.max_point,
    };
  }

  /**
   * Get statistics about the AABB tree
   * @returns {Object} - Tree statistics
   */
  get_stats() {
    return this.tree_processor.get_stats();
  }

  /**
   * Dispatch bounds update
   */
  _dispatch_bounds_update() {
    const transforms = EntityManager.get_fragment_gpu_buffer(
      TransformFragment,
      transforms_buffer_name
    );
    const aabb_node_index = EntityManager.get_fragment_gpu_buffer(
      TransformFragment,
      aabb_node_index_buffer_name
    );
    const entity_flags = FragmentGpuBuffer.entity_flags_buffer;

    this.bounds_update_input_list[0] = transforms.buffer;
    this.bounds_update_input_list[1] = entity_flags.buffer;
    this.bounds_update_input_list[2] = AABB.node_bounds_buffer;
    this.bounds_update_input_list[3] = aabb_node_index.buffer;

    this.bounds_update_output_list[0] = FragmentGpuBuffer.entity_flags_buffer.buffer;
    this.bounds_update_output_list[1] = AABB.node_bounds_buffer;

    const total_rows = EntityManager.get_max_rows();

    ComputeTaskQueue.new_task(
      entity_bounds_update_task_name,
      entity_bounds_update_wgsl_path,
      this.bounds_update_input_list,
      this.bounds_update_output_list,
      Math.max(1, Math.floor((total_rows + 255) / 256))
    );
  }

  /**
   * Process changes to entities, handling updates and pending deletions.
   */
  _process_entity_changes() {
    this.entity_query.for_each_chunk(this._process_entity_changes_chunk_iter);
  }

  _process_entity_changes_chunk_iter(chunk, flags, counts, archetype) {
    const alive_flag = EntityFlags.ALIVE;
    const no_aabb_update_flag = EntityFlags.NO_AABB_UPDATE;
    const transform_views_for_chunk = chunk.get_fragment_view(TransformFragment);
    const aabb_node_index_typed_array = transform_views_for_chunk.aabb_node_index;

    let should_dirty_chunk = false;
    let slot = 0;
    while (slot < DEFAULT_CHUNK_CAPACITY) {
      const instance_count = counts[slot] || 1;
      const entity_current_flags = flags[slot];
      const no_aabb_update = (entity_current_flags & no_aabb_update_flag) !== 0;
      const is_alive = (entity_current_flags & alive_flag) !== 0;

      // active entities: allocate or mark dirty
      if (is_alive) {
        should_dirty_chunk =
          this._handle_active_entity(
            entity_current_flags,
            aabb_node_index_typed_array,
            chunk,
            slot,
            instance_count,
            no_aabb_update
          ) || should_dirty_chunk;
      }

      slot += instance_count;
    }

    if (should_dirty_chunk) {
      chunk.mark_dirty();
    }
  }

  /**
   * Handles logic for entities marked as pending deletione.
   * @param {bigint} entity
   * @param {TypedArray} aabb_node_index_typed_array
   * @param {number} slot
   * @param {number} instance_count
   * @param {boolean} no_aabb_update
   */
  _handle_pending_deletion(aabb_node_index_typed_array, slot, instance_count, no_aabb_update) {
    const base_node_idx_val = aabb_node_index_typed_array[slot];
    const valid_aabb_update = base_node_idx_val > 0 && !no_aabb_update;

    // Only process if it had a valid node and wasn't flagged to skip AABB updates
    // Remove all instances from the tree
    let modified = false;
    for (let i = 0; i < instance_count; i++) {
      const current_instance_offset = slot + i;

      if (valid_aabb_update) {
        const aabb_instance_index = aabb_node_index_typed_array[current_instance_offset];
        if (aabb_instance_index > 0 && !AABB.is_node_free(aabb_instance_index)) {
          this.tree_processor.remove_node_from_tree(aabb_instance_index);
          AABB.free_node(aabb_instance_index);
          modified = true;
        }
      }

      // Clear the index in the fragment data *after* processing
      aabb_node_index_typed_array[current_instance_offset] = 0;
    }

    return modified;
  }

  /**
   * Handles logic for active entities (new node allocation or existing dirty).
   * @param {bigint} entity
   * @param {number} entity_current_flags
   * @param {TypedArray} aabb_node_index_typed_array
   * @param {number} slot
   * @param {number} instance_count
   * @param {boolean} no_aabb_update
   */
  _handle_active_entity(
    entity_current_flags,
    aabb_node_index_typed_array,
    chunk,
    slot,
    instance_count,
    no_aabb_update
  ) {
    let modifiable_entity_current_flags = entity_current_flags;

    let new_node_allocated_in_slot = false;
    for (let i = 0; i < instance_count; i++) {
      const current_instance_offset = slot + i;
      let node_idx_for_instance = aabb_node_index_typed_array[current_instance_offset];

      if (node_idx_for_instance <= 0) {
        // Added !no_aabb_update here too
        const entity = EntityManager.get_entity_for(chunk, current_instance_offset);
        node_idx_for_instance = AABB.allocate_node(entity.id); // entity.id needs to be storable/retrievable
        aabb_node_index_typed_array[current_instance_offset] = node_idx_for_instance;
        // Ensuring EntityFlags.DIRTY is set makes bounds_processing.wgsl run for this new node.
        modifiable_entity_current_flags |= EntityFlags.DIRTY;
        new_node_allocated_in_slot = true;
      }

      // We use aabb_dirty_from_gpu here. If the GPU said it's dirty, we tell the tree processor.
      // The tree processor will then gate on whether the bounds are actually synced.
      if (new_node_allocated_in_slot && node_idx_for_instance > 0 && !no_aabb_update) {
        this.tree_processor.mark_node_dirty(node_idx_for_instance);
      }
    }

    // Write back flags to the chunk.
    // If a new node was allocated, 'modifiable_entity_current_flags' will have the DIRTY bit set.
    // AABB_DIRTY flag (if it was present in entity_current_flags) is preserved.
    // It will be cleared by AABBTreeProcessor later.
    chunk.flags_meta[slot] = modifiable_entity_current_flags;

    return new_node_allocated_in_slot; // Indicate if chunk might need saving due to new DIRTY or AABB_DIRTY processing
  }

  _update_processed_node_flags(processed_node_indices) {
    for (let i = 0; i < processed_node_indices.length; i++) {
      const node_index = processed_node_indices[i];
      if (node_index <= 0 || node_index >= AABB.size) continue;

      const node_view = AABB.get_node_data(node_index); // This is cheap, views over shared array
      // The user_data field in AABB.node_data is a float.
      // Entity IDs are often integers or bigints. This mapping needs to be robust.
      // For this example, assuming user_data can be resolved to entity_id.
      const entity_id_float = node_view.user_data;
      if (entity_id_float !== 0xffffffff && !isNaN(entity_id_float)) {
        // 0xffffffff is often a sentinel for invalid/unused
        const entity_id = Math.floor(entity_id_float);
        const { chunk, slot } = EntityManager.get_chunk_and_slot(entity_id);
        if (chunk) {
          chunk.flags_meta[slot] &= ~EntityFlags.AABB_DIRTY;
        }
      }
    }
  }

  _on_delete(entity) {
    if (!entity || !entity.segments) return;

    const total_instance_count = entity.instance_count;
    let instance = 0;

    for (let i = 0; i < entity.segments.length; i++) {
      const segment = entity.segments[i];

      const chunk = segment.chunk;
      const counts = chunk.icnt_meta;
      const flags = chunk.flags_meta;
      const transform_views_for_chunk = chunk.get_fragment_view(TransformFragment);
      if (!transform_views_for_chunk) continue;

      const aabb_node_index_typed_array = transform_views_for_chunk.aabb_node_index;

      let slot = segment.slot;
      let should_dirty_chunk = false;
      while (slot < DEFAULT_CHUNK_CAPACITY && instance < total_instance_count) {
        const no_aabb_update = (flags[slot] & EntityFlags.NO_AABB_UPDATE) !== 0;
        const instance_count = counts[slot] || 1;

        should_dirty_chunk =
          this._handle_pending_deletion(
            aabb_node_index_typed_array,
            slot,
            instance_count,
            no_aabb_update
          ) || should_dirty_chunk;

        slot += instance_count;
        instance += instance_count;
      }

      if (should_dirty_chunk) {
        chunk.mark_dirty();
      }
    }
  }
}
