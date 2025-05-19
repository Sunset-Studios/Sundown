import { Renderer } from "../../renderer/renderer.js";
import { FragmentGpuBuffer } from "../ecs/solar/memory.js";
import { ComputeTaskQueue } from "../../renderer/compute_task_queue.js";
import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager } from "../ecs/entity.js";
import { EntityFlags } from "../minimal.js";
import { TransformFragment } from "../ecs/fragments/transform_fragment.js";
import { AABB, AABB_NODE_FLAGS } from "../../acceleration/aabb.js";
import { AABBTreeProcessor } from "../../acceleration/aabb_tree_processor.js";
import { profile_scope } from "../../utility/performance.js";
import { warn } from "../../utility/logging.js";

const transforms_buffer_name = "transforms";
const aabb_node_index_buffer_name = "aabb_node_index";
const copy_aabb_data_to_buffer_name = "copy_aabb_data_to_buffer";
const entity_adapter_update_name = "aabb_entity_adapter.update";
const entity_bounds_update_task_name = "entity_bounds_update";
const entity_bounds_update_wgsl_path = "system_compute/bounds_processing.wgsl";
const unmapped_state = "unmapped";

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
    this.on_post_render_callback = this._on_post_render.bind(this);
    this._process_entity_changes_iter = this._process_entity_changes_iter.bind(this);
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
      this.tree_processor.update(delta_time);
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

    Renderer.get().enqueue_post_commands(
      copy_aabb_data_to_buffer_name,
      this.on_post_render_callback
    );
  }

  /**
   * Process changes to entities, handling updates and pending deletions.
   */
  _process_entity_changes() {
    this.entity_query.for_each(this._process_entity_changes_iter, true /* dirty_only */);
  }

  _process_entity_changes_iter(chunk, slot, instance_count, archetype) {
    // Access fragment views and metadata
      const entity_current_flags = chunk.flags_meta[slot];
      const no_aabb_update = (entity_current_flags & EntityFlags.NO_AABB_UPDATE) !== 0;
      const is_pending_delete = (entity_current_flags & EntityFlags.PENDING_DELETE) !== 0;

      const transform_views_for_chunk = chunk.get_fragment_view(TransformFragment);
      const aabb_node_index_typed_array = transform_views_for_chunk.aabb_node_index;

      if (is_pending_delete) {
        this._handle_pending_deletion(
          aabb_node_index_typed_array,
          slot,
          instance_count,
          no_aabb_update
        );
      } else {
        this._handle_active_entity(
          entity_current_flags,
          aabb_node_index_typed_array,
          chunk,
          slot,
          instance_count,
          no_aabb_update
        );
      }
  }

  /**
   * Handles logic for entities marked as PENDING_DELETE.
   * @param {bigint} entity
   * @param {TypedArray} aabb_node_index_typed_array
   * @param {number} slot
   * @param {number} instance_count
   * @param {boolean} no_aabb_update
   */
  _handle_pending_deletion(aabb_node_index_typed_array, slot, instance_count, no_aabb_update) {
    const base_node_idx_val = aabb_node_index_typed_array[slot];

    // Only process if it had a valid node and wasn't flagged to skip AABB updates
    if (base_node_idx_val > 0 && !no_aabb_update) {
      // Remove all instances from the tree
      for (let i = 0; i < instance_count; i++) {
        const current_instance_offset = slot + i;
        const aabb_instance_index = aabb_node_index_typed_array[current_instance_offset];

        if (aabb_instance_index > 0) {
          const is_free = AABB.is_node_free(aabb_instance_index);
          if (!is_free) {
            this.tree_processor.remove_node_from_tree(aabb_instance_index);
            AABB.free_node(aabb_instance_index);
          }
          // Clear the index in the fragment data *after* processing
          aabb_node_index_typed_array[current_instance_offset] = 0;
        }
      }
    } else {
      // Even if skipped or no node, clear the fragment data for all instances to be safe during deletion
      for (let i = 0; i < instance_count; i++) {
        const current_instance_offset = slot + i;
        aabb_node_index_typed_array[current_instance_offset] = 0;
      }
    }
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
    let entity_flags_were_updated = false; // To batch EntityManager.set_entity_flags calls
    let modifiable_entity_current_flags = entity_current_flags;

    for (let i = 0; i < instance_count; i++) {
      const current_instance_offset = slot + i;
      let node_idx_for_instance = aabb_node_index_typed_array[current_instance_offset];

      // 1. Allocate new AABB node if needed (and not skipping updates)
      if (node_idx_for_instance === 0) {
        const new_aabb_node = AABB.allocate_node(); // Using default user_data
        aabb_node_index_typed_array[current_instance_offset] = new_aabb_node;
        node_idx_for_instance = new_aabb_node;

        // Entity needs its bounds calculated by GPU.
        // Ensuring EntityFlags.DIRTY is set makes bounds_processing.wgsl run.
        if ((modifiable_entity_current_flags & EntityFlags.DIRTY) === 0) {
          modifiable_entity_current_flags |= EntityFlags.DIRTY;
          entity_flags_were_updated = true;
        }
      }

      // 2. Process if GPU indicated AABB data changed (via modifiable_entity_current_flags)
      // This means bounds_processing.wgsl updated the AABB data for this node_idx_for_instance on GPU.
      // The CPU-side tree_processor needs to be aware of this potential change.
      if (
        node_idx_for_instance > 0 &&
        modifiable_entity_current_flags & EntityFlags.AABB_DIRTY &&
        !no_aabb_update
      ) {
        this.tree_processor.mark_node_dirty(node_idx_for_instance);
      }
    }

    // Apply any accumulated flag changes to the entity
    if (entity_flags_were_updated) {
      chunk.flags_meta[slot] = modifiable_entity_current_flags;
    }

    // 3. Clear EntityFlags.AABB_DIRTY if it was set at the start of this function call and processed.
    // This flag is set by bounds_processing.wgsl after it updates GPU AABB data. We consume it here.
    if (modifiable_entity_current_flags & EntityFlags.AABB_DIRTY) {
      // Re-fetch in case changed by set_entity_flags above or if no update was made initially
      const latest_entity_flags = entity_flags_were_updated
        ? modifiable_entity_current_flags
        : chunk.flags_meta[slot];

      if (!no_aabb_update) {
        const base_node_idx_val = aabb_node_index_typed_array[slot]; // Check base slot's node index
        if (base_node_idx_val > 0) {
          // Successfully processed or had a node, clear the flag
          chunk.flags_meta[slot] = latest_entity_flags & ~EntityFlags.AABB_DIRTY;
        } else {
          // Was AABB_DIRTY but no valid base node (even after potential alloc attempt). Clear to avoid re-processing.
          chunk.flags_meta[slot] = latest_entity_flags & ~EntityFlags.AABB_DIRTY;
          warn(
            `AABBEntityAdapter: Entity in slot ${slot} was AABB_DIRTY, but its base_node_index is still 0 after processing. Flag cleared.`
          );
        }
      } else {
        // AABB_DIRTY was set, but we're skipping AABB updates for this entity. Clear the flag.
        chunk.flags_meta[slot] = latest_entity_flags & ~EntityFlags.AABB_DIRTY;
      }
    }
  }

  _on_post_render(graph, frame_data, encoder) {
    const buffered_frame = Renderer.get().get_buffered_frame_number();
    if (AABB.node_bounds_cpu_buffer[buffered_frame]?.buffer.mapState === unmapped_state) {
      AABB.node_bounds_buffer.copy_buffer(encoder, 0, AABB.node_bounds_cpu_buffer[buffered_frame]);
    }
  }
}
