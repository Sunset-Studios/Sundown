import { Sector } from "./solar/sector.js";
import { FragmentGpuBuffer } from "./solar/memory.js";
import { SceneGraph } from "../scene_graph.js";
import { TypedQueue } from "../../memory/container.js";
import { EntityFlags } from "../minimal.js";
import { warn, error } from "../../utility/logging.js";

const entity_image_buffer_name = "entity_image_buffer";

/**
 * Manages entities and their associated fragments.
 * @class EntityManager
 */
export class EntityManager {
  static sector = new Sector();
  static entity_fragments = new Map();
  static fragment_types = new Set();
  static pending_entity_deletes = new TypedQueue(256, Uint32Array);

  /**
   * Registers fragments with the sector.
   * @param {typeof import('./fragment.js').Fragment[]} fragment_classes - The fragment classes to register.
   */
  static register_fragments(fragment_classes) {
    FragmentGpuBuffer.init_gpu_buffers(fragment_classes);
    for (let i = 0; i < fragment_classes.length; i++) {
      fragment_classes[i].total_subscribed_instances = 0;
    }
  }

  /**
   * Creates a new entity with the specified fragments and instance count.
   * @param {typeof import('./fragment.js').Fragment[]} fragments - The fragments to add to the entity.
   * @param {number} [instance_count=1] - The number of instances to create for the entity.
   * @returns {number} The newly created entity ID.
   */
  static create_entity(fragments, instance_count = 1) {
    const entity = this.sector.create_entity(fragments, instance_count);
    this.entity_fragments.set(entity, new Set(fragments));
    this.set_entity_flags(entity, EntityFlags.ALIVE);

    for (let i = 0; i < fragments.length; i++) {
      fragments[i].total_subscribed_instances += instance_count;
    }

    return entity;
  }

  /**
   * Deletes an entity.
   * @param {number} entity - The entity to delete.
   */
  static delete_entity(entity) {
    if (!this.entity_fragments.has(entity)) {
      return;
    }

    this.pending_entity_deletes.push(entity);

    const flags = this.get_entity_flags(entity);
    this.set_entity_flags(entity, flags | EntityFlags.PENDING_DELETE);
  }

  /**
   * Checks if an entity exists.
   * @param {number} entity - The entity to check.
   * @returns {boolean} True if the entity exists, false otherwise.
   */
  static entity_exists(entity) {
    return this.entity_fragments.has(entity);
  }

  /**
   * Retrieves the entity ID for a given chunk and slot.
   * @param {Chunk} chunk - The chunk containing the entity.
   * @param {number} slot - The slot index in the chunk.
   * @returns {number} The entity ID.
   */
  static get_entity_for(chunk, slot) {
    return this.sector.get_entity_for(chunk, slot);
  }

  /**
   * Adds a fragment to an entity.
   * @param {number} entity - The entity to add the fragment to.
   * @param {typeof import('./fragment.js').Fragment} FragmentType - The fragment class to add.
   * @returns {import('./solar/view.js').SolarFragmentView | null} The fragment view, or null if not found.
   */
  static add_fragment(entity, FragmentType) {
    if (!this.fragment_types.has(FragmentType)) {
      this.fragment_types.add(FragmentType);
    }
    const already_has_fragment = this.entity_fragments.get(entity).has(FragmentType);
    const fragment_view = this.sector.add_fragment(entity, FragmentType);

    if (!already_has_fragment) {
      this.entity_fragments.get(entity).add(FragmentType);
      const instances = this.get_entity_instance_count(entity);
      FragmentType.total_subscribed_instances += instances;
    }

    return fragment_view;
  }

  /**
   * Removes a fragment from an entity.
   * @param {number} entity - The entity to remove the fragment from.
   * @param {typeof import('./fragment.js').Fragment} FragmentType - The fragment class to remove.
   */
  static remove_fragment(entity, FragmentType) {
    if (
      !this.entity_fragments.has(entity) ||
      !this.entity_fragments.get(entity).has(FragmentType)
    ) {
      return;
    }

    const instances = this.get_entity_instance_count(entity);
    FragmentType.total_subscribed_instances -= instances;

    this.sector.remove_fragment(entity, FragmentType);
    this.entity_fragments.get(entity).delete(FragmentType);
  }

  /**
   * Adds a tag to an entity.
   * @param {number} entity - The entity to add the tag to.
   * @param {typeof import('./fragment.js').Fragment} Tag - The tag class to add.
   */
  static add_tag(entity, Tag) {
    if (!this.fragment_types.has(Tag)) {
      this.fragment_types.add(Tag);
    }
    this.entity_fragments.get(entity).add(Tag);
  }

  /**
   * Removes a tag from an entity.
   * @param {number} entity - The entity to remove the tag from.
   * @param {typeof import('./fragment.js').Fragment} Tag - The tag class to remove.
   */
  static remove_tag(entity, Tag) {
    if (!this.entity_fragments.has(entity) || !this.entity_fragments.get(entity).has(Tag)) {
      return;
    }
    this.entity_fragments.get(entity).delete(Tag);
  }

  /**
   * Retrieves a fragment for an entity.
   * @param {number} entity - The entity to get the fragment for.
   * @param {typeof import('./fragment.js').Fragment} FragmentType - The fragment class to get.
   * @param {number} instance - The instance index of the fragment to get.
   * @returns {import('./solar/view.js').SolarFragmentView | null} The fragment view, or null if not found.
   */
  static get_fragment(entity, FragmentType, instance = 0) {
    if (
      !this.entity_fragments.has(entity) ||
      !this.entity_fragments.get(entity).has(FragmentType)
    ) {
      return null;
    }
    return this.sector.get_fragment(entity, FragmentType, instance);
  }

  /**
   * Checks if an entity has a specific fragment.
   * @param {number} entity - The entity to check.
   * @param {typeof import('./fragment.js').Fragment} FragmentType - The fragment class to check for.
   * @returns {boolean} True if the entity has the fragment, false otherwise.
   */
  static has_fragment(entity, FragmentType) {
    return this.entity_fragments.has(entity) && this.entity_fragments.get(entity).has(FragmentType);
  }

  /**
   * Checks if an entity has a specific tag.
   * @param {number} entity - The entity to check.
   * @param {typeof import('./fragment.js').Fragment} Tag - The tag class to check for.
   * @returns {boolean} True if the entity has the tag, false otherwise.
   */
  static has_tag(entity, Tag) {
    return this.entity_fragments.has(entity) && this.entity_fragments.get(entity).has(Tag);
  }

  /**
   * Retrieves the total number of entities managed by the EntityManager.
   * @returns {number} The total number of entities.
   */
  static get_entity_count() {
    return this.entity_fragments.size;
  }

  /**
   * Retrieves the total instance count for an entity.
   * @param {number} entity - The entity handle to get the instance count for.
   * @returns {number}
   */
  static get_entity_instance_count(entity) {
    return this.sector.get_total_instance_count(entity);
  }

  /**
   * Sets the instance count for an entity.
   * @param {number} entity - The entity to set the instance count for.
   * @param {number} instance_count - The new instance count for the entity.
   * @returns {number} The entity ID.
   */
  static set_entity_instance_count(entity, instance_count) {
    const old_count = this.get_entity_instance_count(entity);
    const handle = this.sector.update_instance_count(entity, instance_count);
    const delta = instance_count - old_count;

    if (delta !== 0) {
      for (const fragment_class of this.entity_fragments.get(entity)) {
        fragment_class.total_subscribed_instances += delta;
      }
    }

    return handle;
  }

  /**
   * Retrieves the flags for an entity.
   * @param {number} entity - The ID of the entity to get the flags for.
   * @returns {number} The flags for the entity.
   */
  static get_entity_flags(entity) {
    const handle = this.sector.entity_map.get(entity.id);
    if (!handle) {
      warn(`Entity ${entity.id} not found in sector map.`);
      return 0;
    }
    const { segments, slot } = handle;
    return segments[0].chunk.flags_meta[slot];
  }

  /**
   * Sets the flags for an entity.
   * @param {number} entity - The ID of the entity to set the flags for.
   * @param {number} flags - The flags to set for the entity.
   */
  static set_entity_flags(entity, flags) {
    const handle = this.sector.entity_map.get(entity.id);
    if (!handle) {
      warn(`Entity ${entity.id} not found in sector map.`);
      return;
    }
    const { segments, slot } = handle;

    for (let i = 0; i < segments.length; i++) {
      const segment = segments[i];
      segment.chunk.flags_meta[segment.slot] = flags;
      segment.chunk.mark_dirty();
    }
  }

  /**
   * Set the dirty flag for an entity.
   * @param {EntityHandle} entity - The entity to set the dirty flag for.
   */
  static set_entity_dirty(entity) {
    const handle = this.sector.entity_map.get(entity.id);
    if (!handle) {
      warn(`Entity ${entity.id} not found in sector map.`);
      return;
    }
    const { segments, slot } = handle;

    for (let i = 0; i < segments.length; i++) {
      const segment = segments[i];
      segment.chunk.flags_meta[segment.slot] |= EntityFlags.DIRTY;
      segment.chunk.mark_dirty();
    }
  }

  /**
   * Get the dirty flag for an entity.
   * @param {EntityHandle} entity - The entity to get the dirty flag for.
   * @returns {number} The dirty flag value.
   */
  static get_entity_dirty(entity) {
    const handle = this.sector.entity_map.get(entity.id);
    if (!handle) {
      warn(`Entity ${entity.id} not found in sector map.`);
      return 0;
    }
    const { segments, slot } = handle;
    return segments[0].chunk.flags_meta[slot];
  }

  /**
   * Processes pending entity deletes.
   */
  static process_pending_deletes() {
    while (this.pending_entity_deletes.length > 0) {
      const entity = this.pending_entity_deletes.pop();
      const entity_flags = this.get_entity_flags(entity);
      if ((entity_flags & EntityFlags.ALIVE) !== 0) {
        this.pending_entity_deletes.push(entity);
        this.set_entity_flags(entity, entity_flags & ~EntityFlags.ALIVE);
        continue;
      }

      // subtract this entity's instances from each fragment's counter
      const instances = this.get_entity_instance_count(entity);
      for (const FragmentType of this.entity_fragments.get(entity)) {
        FragmentType.total_subscribed_instances -= instances;
        FragmentType.remove_entity?.(entity);
      }

      this.sector.destroy_entity(entity);
      SceneGraph.remove(entity);
      this.entity_fragments.delete(entity);
    }
  }

  /**
   * Creates a query for entities matching the given fragment requirements.
   * @param {Object} fragment_requirements - The fragment requirements for the query.
   * @returns {Query} The created query.
   */
  static create_query(fragment_requirements = []) {
    return this.sector.query(fragment_requirements);
  }

  /**
   * Retrieves the GPU buffer associated with a specific fragment field or custom buffer definition.
   * This provides convenient access to the buffers managed by the Sector.
   * Uses precomputed keys for better performance.
   *
   * @param {typeof import('./fragment.js').Fragment} FragmentClass - The fragment class (e.g., TransformFragment).
   * @param {string} field_or_buffer_key - The name of the field (e.g., 'position') or the key
   *                                       of a custom buffer defined in `FragmentClass.gpu_buffers`.
   * @returns {import('./solar/memory.js').FragmentGpuBuffer | null} The GPU buffer instance, or null if not found or not configured for GPU buffering.
   */
  static get_fragment_gpu_buffer(FragmentClass, field_or_buffer_key) {
    if (!FragmentClass?.id) {
      error("EntityManager.get_fragment_gpu_buffer: Invalid FragmentClass provided.");
      return null;
    }
    return FragmentGpuBuffer.get_buffer(FragmentClass, field_or_buffer_key);
  }

  /**
   * Flushes the GPU buffers.
   */
  static flush_gpu_buffers() {
    SceneGraph.flush_gpu_buffers();
    FragmentGpuBuffer.flush_gpu_buffers(this.sector.alloc);
  }

  /**
   * Syncs all GPU buffers down to the CPU.
   */
  static sync_all_buffers() {
    FragmentGpuBuffer.sync_all_buffers();
  }

  /**
   * Copies the GPU buffers to the CPU.
   * @param {GPUCommandEncoder} encoder - The command encoder.
   */
  static copy_gpu_to_cpu_buffers(encoder) {
    FragmentGpuBuffer.copy_to_cpu_buffers(encoder);
  }

  /**
   * Enable or disable entity compaction.
   * @param {boolean} flag - True to enable, false to disable.
   */
  static set_entity_compaction_enabled(flag) {
    FragmentGpuBuffer.set_entity_compaction_enabled(flag);
  }

  /**
   * Check if entity compaction is enabled.
   * @returns {boolean} True if enabled, false otherwise.
   */
  static is_entity_compaction_enabled() {
    return FragmentGpuBuffer.is_entity_compaction_enabled();
  }

  /**
   * Get the entity index map buffer.
   * @returns {GPUBuffer | null} The entity index map buffer, or null if not enabled.
   */
  static get_entity_index_map_buffer() {
    return FragmentGpuBuffer.get_entity_index_map_buffer();
  }

  /**
   * Creates an image buffer for storing entity images.
   * @returns {import('./buffer.js').Buffer} The created image buffer.
   */
  static get_entity_image_buffer() {
    return Buffer.create({
      name: entity_image_buffer_name,
      raw_data: this.get_entity_count() * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  /**
   * Reparent an entity in the global scene-graph.
   */
  static set_entity_parent(entity, parent) {
    SceneGraph.set_parent(entity, parent);
  }

  /**
   * Get an entity's parent (or null).
   */
  static get_entity_parent(entity) {
    return SceneGraph.get_parent(entity);
  }

  /**
   * Get an entity's children array.
   */
  static get_entity_children(entity) {
    return SceneGraph.get_children(entity);
  }

  /** Read back the cached total for a fragment class */
  static get_total_subscribed(fragment_class) {
    return fragment_class.total_subscribed_instances || 0;
  }
}
