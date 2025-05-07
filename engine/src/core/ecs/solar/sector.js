import { Archetype } from "./archetype.js";
import { EntityAllocator } from "./memory.js";
import { Query } from "./query.js";
import { Chunk } from "./chunk.js";
import { EntityHandle, LOCAL_SLOT_BITS } from "./types.js";
import { warn, error } from "../../../utility/logging.js";
import { Name } from "../../../utility/names.js";

const DEFAULT_CHUNK_CAPACITY = 1 << LOCAL_SLOT_BITS; // = 1024

/**
 * Sector is the primary container for ECS data management.
 *
 * It manages entity creation, destruction, and component storage using a chunked
 * archetype-based approach. Each entity belongs to an archetype based on its
 * component composition, and data is stored in type-homogeneous chunks for
 * cache-friendly access patterns.
 *
 * Key responsibilities:
 * - Entity lifecycle management (creation/destruction)
 * - Fragment data storage and access
 * - Archetype management
 * - GPU buffer synchronization for rendering (per-field, or per custom buffer definition)
 * - Query execution against entity data
 *
 * The Sector uses a flat memory layout with SharedArrayBuffers for efficient
 * data access and supports instance counts for entity instancing.
 *
 * Entity IDs are stable and remain consistent even when fragments are
 * added/removed or instance counts change.
 */
export class Sector {
  // EntityAllocator is a shared resource, so we don't need to create a new one for each sector
  alloc = new EntityAllocator();
  // Entity ID → { archetype: Archetype, segments: { chunk: Chunk, slot: number, count: number }[] }
  entity_map = new Map();

  /**
   * Create a new entity with the given fragments and instance count.
   *
   * @param {Fragments[]} fragments - Array of fragments
   * @param {number} [instance_count=1] - Physical rows (≤ 2^LOCAL_SLOT_BITS - 1)
   * @return {number} The stable entity_id for the created entity.
   */
  create_entity(fragments, instance_count = 1) {
    if (instance_count <= 0) {
      throw new Error("Instance count must be positive.");
    }

    const archetype = Archetype.create(fragments);

    // break big instance counts into N <= 1K-sized chunks
    const segments = archetype.claim_segments(instance_count);

    // create 1 stable ID from the first segment
    const first = segments[0];
    const entity_id = this.alloc.create(first.chunk, first.slot, first.count);

    // register the rest of the segments (allocator side-effect mapping)
    for (let i = 1; i < segments.length; i++) {
      const seg = segments[i];
      this.alloc.map_entity_chunk(entity_id, seg.chunk, seg.slot, seg.count);
    }

    // Store archetype and chunk array for the entity
    this.entity_map.set(entity_id, {
      archetype,
      segments,
      total_instance_count: instance_count,
    });

    // Mark all new chunks as dirty
    for (let i = 0; i < segments.length; i++) {
      segments[i].chunk.mark_dirty();
    }

    return EntityHandle.create(entity_id);
  }

  /**
   * Destroy an entity and release its resources.
   *
   * @param {EntityHandle} handle - The handle of the entity to destroy
   */
  destroy_entity(handle) {
    const entity_id = handle.id;

    const location = this.entity_map.get(entity_id);
    if (!location) {
      warn(`Entity ${entity_id} not found, cannot destroy.`);
      return;
    }

    const { segments } = location;

    // Destroy tells the allocator to release the ID and associated resources (like the slot).
    // Allocator's destroy method handles freeing the row in the chunk.
    if (segments.length > 0) {
      // Free all additional segments first
      for (let i = 1; i < segments.length; i++) {
        const seg = segments[i];
        this.alloc.free_segment(seg.chunk, seg.slot, seg.count);
      }
      // Finally destroy the entity using the first segment (also frees the first segment under the hood)
      this.alloc.destroy(entity_id, segments[0].chunk, segments[0].slot);
    }

    // Remove from our map
    this.entity_map.delete(entity_id);

    handle.id = 0;

    EntityHandle.destroy(handle);
  }

  /**
   * Adds a single fragment type to an existing entity.
   * If the entity already has the fragment, it returns a view to the existing fragment data.
   * Otherwise, it migrates the entity to a new archetype containing the added fragment
   * and returns a view to the newly added fragment data.
   * The entity ID remains stable.
   *
   * @template {import('../fragment.js').Fragment} T
   * @param {EntityHandle} handle - The handle of the entity to add the fragment to.
   * @param {T} fragment_class - The class constructor of the Fragment to add (e.g., TransformFragment).
   * @param {number} [instance_index=0] - The instance index to return the view for (if the fragment already exists or after adding).
   * @returns {import('./view.js').SolarFragmentView | null} A configured SolarFragmentView for the added/existing fragment,
   *                                                       or null if the entity doesn't exist or the fragment class is invalid.
   */
  add_fragment(handle, fragment_class, instance_index = 0) {
    const entity_id = handle.id;

    // 1. Validate Fragment Class
    if (!fragment_class.is_valid()) {
      error(
        `Sector.add_fragment: Invalid fragment_class provided. Missing static 'id', 'fields', or 'view_allocator'.`,
        fragment_class
      );
      return null;
    }

    const fragment_id_to_add = fragment_class.id;

    // 2. Get Entity Location
    const location = this.entity_map.get(entity_id);
    if (!location) {
      warn(`Sector.add_fragment: Entity ${entity_id} not found.`);
      return null; // Entity doesn't exist
    }

    const { archetype: old_archetype } = location;

    // 3. Check if Fragment Already Exists
    const already_has_fragment = old_archetype.fragments.some((f) => f.id === fragment_id_to_add);

    if (already_has_fragment) {
      // Fragment exists, just return a view to it
      return this.get_fragment(handle, fragment_class, instance_index);
    } else {
      // Fragment needs to be added, requires migration
      const final_fragments = [...old_archetype.fragments, fragment_class];

      // Migrate will update the entity in place. We ignore the returned ID (it's stable).
      this.migrate(handle, final_fragments);

      // After migration, the entity *definitely* has the fragment. Get a view.
      return this.get_fragment(handle, fragment_class, instance_index);
    }
  }

  /**
   * Removes a single fragment type from an existing entity.
   * The entity ID remains stable.
   *
   * @param {EntityHandle} handle - The handle of the entity to remove the fragment from.
   * @param {typeof import('../fragment.js').Fragment} fragment_class - The class constructor of the Fragment to remove (e.g., TransformFragment).
   * @returns {void}
   */
  remove_fragment(handle, fragment_class) {
    const entity_id = handle.id;

    const location = this.entity_map.get(entity_id);
    if (!location) {
      warn(`Sector.remove_fragment: Entity ${entity_id} not found.`);
      return; // Entity doesn't exist
    }

    // Check if the entity actually has the fragment
    const current_fragments = location.archetype.fragments;
    const fragment_index = current_fragments.findIndex((f) => f.id === fragment_class.id);

    if (fragment_index === -1) {
      return; // Fragment not present, no-op
    }

    // Create the list of remaining fragments
    const final_fragments = current_fragments.filter(
      (fragment) => fragment.id !== fragment_class.id
    );

    // Check if removal would result in an empty archetype
    if (final_fragments.length === 0) {
      warn(
        `Sector.remove_fragment: Entity ${entity_id} would have no fragments after removal of ${fragment_class.id}.`
      );
      return; // Entity doesn't exist
    }

    // Migrate will update the entity in place and return the same ID (which we ignore)
    this.migrate(handle, final_fragments);
  }

  /**
   * Retrieves a reusable view object pointing to the data of a specific fragment
   * for a given entity instance.
   *
   * @template {import('../fragment.js').Fragment} T
   * @param {EntityHandle} handle - The handle of the entity.
   * @param {T} fragment_class - The class constructor of the Fragment (e.g., TransformFragment).
   * @param {number} [instance_index=0] - The index of the instance to view (for multi-instance entities).
   * @returns {import('./view.js').SolarFragmentView | null} A configured SolarFragmentView instance from the fragment's
   *                                                      allocator, or null if the entity doesn't exist,
   *                                                      doesn't have the fragment, the instance index is invalid,
   *                                                      or the fragment class is improperly configured.
   */
  get_fragment(handle, fragment_class, instance_index = 0) {
    const entity_id = handle.id;

    // 1. Validate Fragment Class Configuration
    if (!fragment_class.is_valid()) {
      error(
        `Sector.get_fragment: Invalid fragment_class provided. Missing static 'id' or 'view_allocator'.`,
        fragment_class
      );
      return null;
    }

    const fragment_id = fragment_class.id;

    // 2. Look up Entity Location
    const location = this.entity_map.get(entity_id);
    if (!location) {
      return null;
    }

    const { archetype, segments, total_instance_count } = location;

    // 3. Check if Archetype contains the Fragment
    if (!archetype.fragments.some((f) => f.id === fragment_id)) {
      return null;
    }

    // 4. Validate Instance Index
    if (instance_index < 0 || instance_index >= total_instance_count) {
      error(
        `Sector.get_fragment: Invalid instance_index ${instance_index} for entity ${entity_id}. Count is ${total_instance_count}.`
      );
      return null;
    }

    // 5. Allocate and Configure View
    try {
      const segment_index = Math.floor(instance_index / DEFAULT_CHUNK_CAPACITY);
      const instance = instance_index % DEFAULT_CHUNK_CAPACITY;
      const { chunk, slot } = segments[segment_index];
      return fragment_class.view_allocator.allocate().view(handle, chunk, slot, instance);
    } catch (err) {
      error(
        `Sector.get_fragment: Error allocating or configuring view for fragment '${Name.string(fragment_id)}':`,
        err
      );
      return null;
    }
  }

  /**
   * Retrieves the entity ID for a given chunk and slot.
   *
   * @param {Chunk} chunk - The chunk containing the entity
   * @param {number} slot - The slot index in the chunk
   * @returns {number} The entity ID
   */
  get_entity_for(chunk, slot) {
    return EntityHandle.create(this.alloc.get_entity_id_for(chunk, slot));
  }

  /**
   * Update the instance count of an existing entity. The entity ID remains stable.
   *
   * @param {EntityHandle} handle - The handle of the entity to update
   * @param {number} new_instance_count - The new instance count (must be > 0)
   * @return {EntityHandle} The original handle.
   */
  update_instance_count(handle, new_instance_count) {
    const entity_id = handle.id;

    if (new_instance_count <= 0) {
      throw new Error("New instance count must be positive.");
    }

    const location = this.entity_map.get(entity_id);
    if (!location) throw new Error(`Entity ${entity_id} not found`);

    const { archetype, segments, total_instance_count } = location;

    // no-op if unchanged
    if (new_instance_count === total_instance_count) {
      return handle;
    }

    // break the big count into N <= 1K-sized chunks
    let new_segments = archetype.claim_segments(new_instance_count);

    // Copy data from old segments to new segments
    for (let i = 0; i < archetype.fragments.length; i++) {
      const fragment = archetype.fragments[i];
      const fragment_id = fragment.id;
      const fragment_field_entries = Object.entries(fragment.fields);

      for (let j = 0; j < segments.length; j++) {
        const old_seg = segments[j];
        const new_seg = new_segments[j];
        const source_views = old_seg.chunk.fragment_views[fragment_id];
        const dest_views = new_seg.chunk.fragment_views[fragment_id];

        for (let k = 0; k < fragment_field_entries.length; k++) {
          const [field_name, field_spec] = fragment_field_entries[k];
          // skip variable-sized/container fields here
          if (field_spec.is_container) continue;

          const element_count = field_spec.elements;
          const source_offset = old_seg.slot * element_count;
          const copy_elements = old_seg.count * element_count;
          const dest_offset = new_seg.slot * element_count;

          dest_views[field_name].set(
            source_views[field_name].subarray(source_offset, source_offset + copy_elements),
            dest_offset
          );
        }
      }
    }

    // Inform allocator to release old segments and update tracking for the entity ID
    for (let i = 1; i < segments.length; i++) {
      const seg = segments[i];
      this.alloc.free_segment(seg.chunk, seg.slot, seg.count);
    }

    // Update allocation tracking for the entity ID, to point to the new first chunk
    const chunk_instance_count = Math.min(new_instance_count, DEFAULT_CHUNK_CAPACITY);
    const new_entity_id = this.alloc.update_allocation(
      entity_id,
      segments[0].chunk,
      segments[0].slot,
      new_segments[0].chunk,
      new_segments[0].slot,
      chunk_instance_count
    );

    // Update our map entry for the *same* entity ID
    this.entity_map.set(new_entity_id, {
      archetype,
      segments: new_segments,
      total_instance_count: new_instance_count,
    });

    // Mark the new chunks/slots as dirty for all relevant fragments (if tracked)
    for (let i = 0; i < new_segments.length; i++) {
      const seg = new_segments[i];
      seg.chunk.mark_dirty();
    }

    handle.id = new_entity_id;

    return handle; // Return the original, stable ID
  }

  /**
   * Create a new query object to iterate over entities with matching fragments.
   * @return {Query}
   */
  query(fragment_requirements) {
    // Pass map of archetypes to the query system
    return Query.create(fragment_requirements);
  }

  /**
   * Migrate an entity to a new archetype with the given fragments.
   * The entity ID remains stable. Assumes new_fragments is sorted by ID.
   *
   * @private
   * @param {EntityHandle} handle - The handle of the entity to migrate
   * @param {Fragments[]} new_fragments - Array of fragments for the new archetype (sorted by ID)
   * @return {EntityHandle} The original handle.
   */
  migrate(handle, new_fragments) {
    const entity_id = handle.id;

    const location = this.entity_map.get(entity_id);
    // Should always exist if called from public methods, but double-check
    if (!location) throw new Error(`Entity ${entity_id} disappeared during migration`);

    const { archetype: old_archetype, segments, total_instance_count } = location;

    const new_archetype_key = Archetype.get_id(new_fragments);

    // Avoid migration if archetype doesn't actually change
    if (new_archetype_key === old_archetype.id) {
      return handle;
    }

    // Create the new archetype
    let new_archetype = Archetype.create(new_fragments);
    // Claim space in the new archetype
    let new_segments = new_archetype.claim_segments(total_instance_count);

    // Copy data from old segments to new segments
    for (let i = 0; i < old_archetype.fragments.length; i++) {
      const fragment = old_archetype.fragments[i];
      const fragment_id = fragment.id;
      const fragment_field_entries = Object.entries(fragment.fields);

      for (let j = 0; j < segments.length; j++) {
        const old_seg = segments[j];
        const new_seg = new_segments[j];
        const source_views = old_seg.chunk.fragment_views[fragment_id];
        const dest_views = new_seg.chunk.fragment_views[fragment_id];

        for (let k = 0; k < fragment_field_entries.length; k++) {
          const [field_name, field_spec] = fragment_field_entries[k];
          // skip variable-sized/container fields here
          if (field_spec.is_container) continue;

          const element_count = field_spec.elements;
          const source_offset = old_seg.slot * element_count;
          const copy_elements = old_seg.count * element_count;
          const dest_offset = new_seg.slot * element_count;

          dest_views[field_name].set(
            source_views[field_name].subarray(source_offset, source_offset + copy_elements),
            dest_offset
          );
        }
      }
    }

    // Update allocation: Release old segments, update tracking to new location
    for (let i = 1; i < segments.length; i++) {
      const seg = segments[i];
      this.alloc.free_segment(seg.chunk, seg.slot, seg.count);
    }

    // Update allocation tracking for the entity ID, to point to the new first chunk
    const chunk_instance_count = Math.min(total_instance_count, DEFAULT_CHUNK_CAPACITY);
    const new_entity_id = this.alloc.update_allocation(
      entity_id,
      segments[0].chunk,
      segments[0].slot,
      new_segments[0].chunk,
      new_segments[0].slot,
      chunk_instance_count
    );

    // Update the entity map entry for the *same* entity ID
    this.entity_map.set(new_entity_id, {
      archetype: new_archetype,
      segments: new_segments,
      total_instance_count: total_instance_count,
    });

    // Mark the new chunks/slots as dirty
    for (const { chunk } of new_segments) {
      chunk.mark_dirty();
    }

    // Check if old_segments became empty and trigger archetype cleanup/compaction?
    // if (old_segments.length === 0) { old_archetype.release_segments(segments); }

    handle.id = new_entity_id;

    return handle; // Return the original, stable ID
  }

  /**
   * Fetch the total number of instances for an entity.
   * @param {EntityHandle} handle
   * @returns {number}
   */
  get_total_instance_count(handle) {
    const entity_id = handle.id;
    const loc = this.entity_map.get(entity_id);
    return loc ? loc.total_instance_count : 0;
  }

  /**
   * Get the layout of an entity.
   * @param {EntityHandle} handle
   * @returns {Object}
   */
  get_entity_layout(handle) {
    const entity_id = handle.id;
    return this.entity_map.get(entity_id);
  }
}
