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

    this.bounds_update_input_list[0] = transforms.buffer;
    this.bounds_update_input_list[3] = AABB.node_bounds_buffer;
    this.bounds_update_input_list[4] = aabb_node_index.buffer;

    this.bounds_update_output_list[0] = FragmentGpuBuffer.entity_flags_buffer.buffer;
    this.bounds_update_output_list[1] = AABB.node_bounds_buffer;

    const total_transforms = EntityManager.get_total_subscribed(TransformFragment);

    ComputeTaskQueue.get().new_task(
      entity_bounds_update_task_name,
      entity_bounds_update_wgsl_path,
      this.bounds_update_input_list,
      this.bounds_update_output_list,
      Math.max(1, Math.floor((total_transforms + 255) / 256))
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
    this.entity_query.for_each((chunk, slot, instance_count, archetype) => {
      // Access fragment views and metadata
      const frag_view = chunk.get_fragment_view(TransformFragment);
      // Check if the fragment view exists (it should for this query)
      if (!frag_view) return;

      const entity = EntityManager.get_entity_for(chunk, slot);
      const entity_flags = EntityManager.get_entity_flags(entity);

      const no_aabb_update = (entity_flags & EntityFlags.NO_AABB_UPDATE) !== 0;
      const is_pending_delete = (entity_flags & EntityFlags.PENDING_DELETE) !== 0;
      const is_aabb_dirty = (entity_flags & EntityFlags.AABB_DIRTY) !== 0;

      const aabb_node_index_fragment = frag_view.aabb_node_index;

      // --- Handle Pending Deletion ---
      if (is_pending_delete) {
        const base_node_index = aabb_node_index_fragment[slot];

        // Only process if it had a valid node and wasn't flagged to skip AABB updates
        if (base_node_index > 0 && !no_aabb_update) {
          // Remove all instances from the tree
          for (let i = 0; i < instance_count; i++) {
            const entity_instance_offset = slot + i;
            const aabb_instance_index = aabb_node_index_fragment[entity_instance_offset];

            if (aabb_instance_index > 0) {
              const is_free = AABB.is_node_free(aabb_instance_index);
              if (!is_free) {
                this.tree_processor.remove_node_from_tree(aabb_instance_index);
                AABB.free_node(aabb_instance_index);
              }
              // Clear the index in the fragment data *after* processing
              aabb_node_index_fragment[entity_instance_offset] = 0;
            }
          }
        } else {
          // Even if skipped or no node, clear the fragment data for all instances to be safe during deletion
          for (let i = 0; i < instance_count; i++) {
            const entity_instance_offset = slot + i;
            aabb_node_index_fragment[entity_instance_offset] = 0;
          }
        }
      }
      // --- Handle Active, Dirty Entities ---
      else if (is_aabb_dirty && !no_aabb_update) {
        const base_node_index = aabb_node_index_fragment[slot];
        // Check if the base instance has a node index (should generally be true if dirty)
        if (base_node_index > 0) {
          for (let i = 0; i < instance_count; i++) {
            const entity_instance_offset = slot + i;
            const aabb_instance_index = aabb_node_index_fragment[entity_instance_offset];
            // Only mark dirty if the instance has a valid node index
            if (aabb_instance_index > 0) {
              this.tree_processor.mark_node_dirty(aabb_instance_index);
              // Clear the dirty flag for the specific instance
              EntityManager.set_entity_flags(entity, entity_flags & ~EntityFlags.AABB_DIRTY);
            }
          }
        } else {
          // Clear the dirty flag even if the node index is missing/zero to prevent re-processing
          for (let i = 0; i < instance_count; i++) {
            const entity_instance_offset = slot + i;
            EntityManager.set_entity_flags(entity, entity_flags & ~EntityFlags.AABB_DIRTY);
          }
          warn(
            `AABBEntityAdapter: Entity in slot ${slot} marked AABB_DIRTY but base_node_index is ${base_node_index}.`
          );
        }
      }
    });
  }

  _on_post_render(graph, frame_data, encoder) {
    const buffered_frame = Renderer.get().get_buffered_frame_number();
    if (AABB.node_bounds_cpu_buffer[buffered_frame]?.buffer.mapState === unmapped_state) {
      AABB.node_bounds_buffer.copy_buffer(encoder, 0, AABB.node_bounds_cpu_buffer[buffered_frame]);
    }
  }
}
