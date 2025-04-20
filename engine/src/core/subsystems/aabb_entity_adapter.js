import { Renderer } from "../../renderer/renderer.js";
import { ComputeTaskQueue } from "../../renderer/compute_task_queue.js";
import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager, EntityID } from "../ecs/entity.js";
import { EntityMasks } from "../ecs/query.js";
import { EntityTransformFlags } from "../minimal.js";
import { TransformFragment } from "../ecs/fragments/transform_fragment.js";
import { AABB, AABB_NODE_FLAGS } from "../../acceleration/aabb.js";
import { AABBTreeProcessor } from "../../acceleration/aabb_tree_processor.js";
import { profile_scope } from "../../utility/performance.js";

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
    this.entity_query = EntityManager.create_query({
      fragment_requirements: [TransformFragment],
    });
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
    const transform_fragment = EntityManager.get_fragment(entity, TransformFragment);
    if (!transform_fragment) return;

    const entity_offset = EntityID.get_absolute_index(entity);
    const entity_instances = EntityID.get_instance_count(entity);
    for (let i = 0; i < entity_instances; i++) {
      this.tree_processor.mark_node_dirty(transform_fragment.aabb_node_index[entity_offset + i]);
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
    const transforms = EntityManager.get_fragment_array(TransformFragment);
    if (!transforms) return;

    this.bounds_update_input_list[0] = transforms.transforms_buffer;
    this.bounds_update_input_list[1] = transforms.flags_buffer;
    this.bounds_update_input_list[2] = transforms.dirty_buffer;
    this.bounds_update_input_list[3] = AABB.data.node_data_buffer;
    this.bounds_update_input_list[4] = AABB.data.node_bounds_buffer;
    this.bounds_update_input_list[5] = transforms.aabb_node_index_buffer;

    this.bounds_update_output_list[0] = transforms.flags_buffer;
    this.bounds_update_output_list[1] = AABB.data.node_bounds_buffer;

    ComputeTaskQueue.get().new_task(
      entity_bounds_update_task_name,
      entity_bounds_update_wgsl_path,
      this.bounds_update_input_list,
      this.bounds_update_output_list,
      Math.max(1, Math.floor((TransformFragment.size + 255) / 256))
    );

    Renderer.get().enqueue_post_commands(
      copy_aabb_data_to_buffer_name,
      this.on_post_render_callback
    );
  }

  /**
   * Process changes to entities
   */
  _process_entity_changes() {
    const transforms = EntityManager.get_fragment_array(TransformFragment);
    if (!transforms) return;

    // Get all entities with transforms
    const entities = this.entity_query.matching_entities.get_data();
    const entity_offsets = this.entity_query.matching_entity_ids.get_data();
    const entity_states = this.entity_query.entity_states.get_data();
    const entity_instance_counts = this.entity_query.matching_entity_instance_counts.get_data();

    // Process entities that need updating
    for (let i = 0; i < this.entity_query.matching_entities.length; i++) {
      const entity = entities[i];
      const entity_state = entity_states[i];
      const entity_offset = entity_offsets[i];

      const node_index = transforms.aabb_node_index[entity_offset];
      const no_aabb_update = (transforms.flags[entity_offset] & EntityTransformFlags.NO_AABB_UPDATE) !== 0;

      // Handle removed entities
      if (entity_state === EntityMasks.Removed && node_index) {
        // Make sure to remove all instances of the entity from the tree
        const entity_count = entity_instance_counts[i];
        for (let i = 0; i < entity_count; i++) {
          const entity_instance_offset = entity_offset + i;
          const aabb_instance_index = transforms.aabb_node_index[entity_instance_offset];
          const is_free = AABB.is_node_free(aabb_instance_index);

          if (!no_aabb_update || is_free) {
            this.tree_processor.remove_node_from_tree(aabb_instance_index);
          }

          if (!is_free) {
            AABB.free_node(aabb_instance_index);
          }

          transforms.aabb_node_index[entity_instance_offset] = 0;
        }
      }
      // Handle new or updated entities
      else if (node_index > 0 && (transforms.flags[entity_offset] & EntityTransformFlags.AABB_DIRTY) !== 0 && !no_aabb_update) {
        const entity_instances = EntityID.get_instance_count(entity);
        for (let i = 0; i < entity_instances; i++) {
          const entity_instance_offset = entity_offset + i;
          const aabb_instance_index = transforms.aabb_node_index[entity_instance_offset];
          this.tree_processor.mark_node_dirty(aabb_instance_index);
          transforms.flags[entity_instance_offset] &= ~EntityTransformFlags.AABB_DIRTY;
        }
      }
    }
  }

  _on_post_render(graph, frame_data, encoder) {
    const buffered_frame = Renderer.get().get_buffered_frame_number();
    if (AABB.data.node_bounds_cpu_buffer[buffered_frame]?.buffer.mapState === unmapped_state) {
      AABB.data.node_bounds_buffer.copy_buffer(encoder, 0, AABB.data.node_bounds_cpu_buffer[buffered_frame]);
    }
  }
}
