import { SharedEntityMetadataBuffer } from "../shared_data.js";
import { EntityQuery } from "./query.js";
import { TypedVector, TypedQueue } from "../../memory/container.js";
import { clamp } from "../../utility/math.js";
import { EntityFlags } from "../minimal.js";

const entity_image_buffer_name = "entity_image_buffer";
const object_name = "object";

export class EntityID {
  static get_absolute_index(entity) {
    return SharedEntityMetadataBuffer.get_entity_offset(entity);
  }

  static get_instance_count(entity) {
    return SharedEntityMetadataBuffer.get_entity_count(entity);
  }

  static set_instance_count(entity, instance_count) {
    SharedEntityMetadataBuffer.set_entity_instance_count(entity, instance_count);
  }

  static get_entity_flags(entity) {
    return SharedEntityMetadataBuffer.get_entity_flags(entity);
  }

  static set_entity_flags(entity, flags) {
    SharedEntityMetadataBuffer.set_entity_flags(entity, flags);
  }
}

export class EntityManager {
  static next_entity_id = 0;
  static highest_entity = 0;
  static entity_fragments = new Map();
  static fragment_types = new Set();
  static entities = new TypedVector(256, 0, Float64Array);
  static deleted_entities = new Set();
  static pending_instance_count_changes = new Map();
  static pending_entity_deletes = new TypedQueue(256, Uint32Array);
  static needs_entity_refresh = false;

  static preinit_fragments(...fragment_types) {
    for (const fragment_type of fragment_types) {
      if (!this.fragment_types.has(fragment_type)) {
        fragment_type.initialize();
        this.fragment_types.add(fragment_type);
      }
    }
  }

  static reserve_entities(size, rebuild_buffers = true) {
    SharedEntityMetadataBuffer.resize(size);

    for (const fragment_type of this.fragment_types) {
      fragment_type.resize?.(size);
    }

    if (rebuild_buffers) {
      this.rebuild_buffers();
    }
  }

  static create_entity() {
    let entity;

    if (this.deleted_entities.size > 0) {
      entity = this.deleted_entities.values().next().value;
      this.deleted_entities.delete(entity);
    } else {
      entity = this.next_entity_id++;
    }

    SharedEntityMetadataBuffer.add_entity(entity, this.pending_instance_count_changes);

    // Resize all fragment data arrays to fit the new entity
    for (const fragment_type of this.fragment_types) {
      fragment_type.resize?.(entity);
    }

    this.entities.push(entity);
    this.entity_fragments.set(entity, new Set());
    this.needs_entity_refresh = true;

    EntityID.set_entity_flags(entity, 0);

    return entity;
  }

  static delete_entity(entity) {
    if (!this.entity_fragments.has(entity)) {
      return;
    }

    this.pending_entity_deletes.push(entity);

    const flags = EntityID.get_entity_flags(entity);
    EntityID.set_entity_flags(entity, flags | EntityFlags.PENDING_DELETE);

    this.needs_entity_refresh = true;
  }

  static duplicate_entity(entity, instance = 0) {
    const new_entity = this.create_entity();

    for (const FragmentType of this.entity_fragments.get(entity)) {
      if (FragmentType.data) {
        const data = FragmentType.duplicate_entity_data?.(entity, instance);
        const new_frag_view = this.add_fragment(new_entity, FragmentType);
        for (const [key, value] of Object.entries(data)) {
          if (value !== null) {
            if (Array.isArray(value)) {
              new_frag_view[key] = [...value];
            } else if (typeof value === object_name) {
              for (const [sub_key, sub_value] of Object.entries(value)) {
                new_frag_view[key][sub_key] = sub_value;
              }
            } else {
              new_frag_view[key] = value;
            }
          } else {
            new_frag_view[key] = value;
          }
        }
      } else {
        this.add_tag(new_entity, FragmentType);
      }
    }

    return new_entity;
  }

  static entity_exists(entity) {
    return SharedEntityMetadataBuffer.entity_exists(entity);
  }

  static add_fragment(entity, FragmentType) {
    if (!this.fragment_types.has(FragmentType)) {
      FragmentType.initialize();
      this.fragment_types.add(FragmentType);
    }
    const fragment_view = FragmentType.add_entity(entity);
    this.entity_fragments.get(entity).add(FragmentType);
    this.needs_entity_refresh = true;
    return fragment_view;
  }

  static remove_fragment(entity, FragmentType) {
    if (
      !this.entity_fragments.has(entity) ||
      !this.entity_fragments.get(entity).has(FragmentType)
    ) {
      return;
    }
    FragmentType.remove_entity(entity);
    this.entity_fragments.get(entity).delete(FragmentType);
    this.needs_entity_refresh = true;
  }

  static add_tag(entity, Tag) {
    if (!this.fragment_types.has(Tag)) {
      this.fragment_types.add(Tag);
    }
    this.entity_fragments.get(entity).add(Tag);
    this.needs_entity_refresh = true;
  }

  static remove_tag(entity, Tag) {
    if (!this.entity_fragments.has(entity) || !this.entity_fragments.get(entity).has(Tag)) {
      return;
    }
    this.entity_fragments.get(entity).delete(Tag);
    this.needs_entity_refresh = true;
  }

  static get_fragment(entity, FragmentType, instance = 0) {
    if (
      !this.entity_fragments.has(entity) ||
      !this.entity_fragments.get(entity).has(FragmentType)
    ) {
      return null;
    }
    return FragmentType.get_entity_data(entity, instance);
  }

  static has_fragment(entity, FragmentType) {
    return this.entity_fragments.has(entity) && this.entity_fragments.get(entity).has(FragmentType);
  }

  static has_tag(entity, Tag) {
    return this.entity_fragments.has(entity) && this.entity_fragments.get(entity).has(Tag);
  }

  static get_fragment_array(FragmentType) {
    return FragmentType.data;
  }

  static get_entity_count() {
    return this.entities.length;
  }

  static get_entity_instance_count(entity) {
    return EntityID.get_instance_count(entity);
  }

  static set_entity_instance_count(entity, instance_count) {
    const last_count = EntityID.get_instance_count(entity);
    if (instance_count === last_count) return;
    EntityID.set_instance_count(entity, instance_count);
    this.pending_instance_count_changes.set(entity, [last_count, instance_count]);
    this.needs_entity_refresh = true;
  }

  static mark_needs_entity_refresh() {
    this.needs_entity_refresh = true;
  }

  static process_pending_deletes() {
    while (this.pending_entity_deletes.length > 0) {
      const entity = this.pending_entity_deletes.pop();

      for (const FragmentType of this.entity_fragments.get(entity)) {
        FragmentType.remove_entity?.(entity);
      }
  
      SharedEntityMetadataBuffer.remove_entity(entity);
      
      this.entity_fragments.delete(entity);
      this.entities.remove(this.entities.index_of(entity));
      this.deleted_entities.add(entity);
      this.needs_entity_refresh = true;

      EntityID.set_entity_flags(entity, 0);
    }
  }

  static flush_instance_count_changes() {
    // 1. If no pending changes, nothing to do
    if (this.pending_instance_count_changes.size === 0) return;

    // 1.5. Update offsets
    SharedEntityMetadataBuffer.update_offsets();

    // 2. Sort changes by ascending entity index
    const sorted_changes = Array.from(this.pending_instance_count_changes.entries()).sort(
      ([entityA], [entityB]) =>
        EntityID.get_absolute_index(entityA) - EntityID.get_absolute_index(entityB)
    );

    // 3. Retrieve the highest entity and cache it in the entity manager
    const highest_entity = sorted_changes[sorted_changes.length - 1][0];
    const highest_entity_new_count = sorted_changes[sorted_changes.length - 1][1][1];
    this.highest_entity = EntityID.get_absolute_index(highest_entity) + highest_entity_new_count;

    // 4. Compute total net change, and also build a difference array
    //    that marks how many slots get added/removed starting at entity_index+old_count
    let total_size_change = 0;
    for (const [entity, [last_count, new_count]] of sorted_changes) {
      total_size_change += new_count - last_count;
    }

    // 5. Resize the fragment arrays to the new total size
    for (const frag of this.fragment_types) {
      if (!frag.size) continue;
      if (this.highest_entity > frag.size) {
        frag.resize(this.highest_entity);
      }
    }

    // 6. Build the “difference array” + prefix sum to figure out the shift for each slot.
    //    Let’s define `max_size` as the final capacity in the largest fragment.
    const max_size = Math.max(...Array.from(this.fragment_types).map((ft) => ft.size ? ft.size : 0));
    const shifts = new Int32Array(max_size + 1);

    // Build difference array (use absolute index + last_count, not raw entity ID)
    for (const [entity, [last_count, new_count]] of sorted_changes) {
      // compute where this entity’s old block ends
      const entity_offset = EntityID.get_absolute_index(entity);
      const shift_start = entity_offset + last_count;
      if (shift_start <= max_size) {
        shifts[shift_start] += (new_count - last_count);
      }
    }

    // Turn it into an actual offset array by prefix sum
    for (let i = 1; i < shifts.length; i++) {
      shifts[i] += shifts[i - 1];
    }
    // Now `shifts[i]` = how far index `i` in the old layout should move (positive => right)

    // 7A. Expansions pass (not needed here, as it's the last pass as well?)
    //     We do a single loop from (max_size - 1) down to 0.
    //     If shifts[i] > 0, we move from i -> i+shifts[i].
    //     Then we mark that we used up that old data.
    for (let i = max_size - 1; i >= 0; i--) {
      const shift = shifts[i];
      if (shift > 0) {
        for (const frag of this.fragment_types) {
          if (!frag.size) continue;
          const to = clamp(i + shift, 0, frag.size - 1);
          const from = clamp(i, 0, frag.size - 1);
          frag.copy_entity_instance?.(to, from);
        }
      }
    }

    // 7B. Contractions pass
    //     We do a single loop from 0 up to (max_size - 1).
    //     If shifts[i] < 0, we move from i -> i+shifts[i].
    //     Then we mark that we used up that old data.
    for (let i = 0; i < max_size; i++) {
      const shift = shifts[i];
      if (shift < 0) {
        for (const frag of this.fragment_types) {
          if (!frag.size) continue;
          const to = clamp(i + shift, 0, frag.size - 1);
          const from = clamp(i, 0, frag.size - 1);
          frag.copy_entity_instance?.(to, from);
        }
      }
    }

    // 9. Now we handle truly “new” slots that didn’t exist before.
    //    For example, if an entity had old_count=1, new_count=3 => we shifted the *existing* slot,
    //    but we haven’t filled the brand‐new 2 slots with data yet.
    //
    //    To do that, we’ll iterate over each entity in ascending order.
    //    For the newly added slots, we replicate from the "original entity" index
    //    (which might be the first slot of that entity, or some known reference index).
    //    We find these new slots by scanning the range [entity_index + old_count, entity_index + new_count).
    //    The shift array told us where to shift old data, but new data is “uninitialized.”

    for (const [entity, [last_count, new_count]] of sorted_changes) {
      const added = new_count - last_count;
      if (added <= 0) continue;
      
      const entity_index = EntityID.get_absolute_index(entity);
      for (let j = last_count; j < new_count; j++) {
        const new_absolute_index = entity_index + j;
        for (const frag of this.fragment_types) {
          frag.copy_entity_instance?.(new_absolute_index, entity_index);
        }
      }
    }

    // 10. Clear pending changes
    this.pending_instance_count_changes.clear();
  }

  static get_entities() {
    return this.entities;
  }

  static create_query({ fragment_requirements }) {
    return EntityQuery.create(fragment_requirements);
  }

  static rebuild_buffers() {
    SharedEntityMetadataBuffer.rebuild();
    for (const fragment_type of this.fragment_types) {
      fragment_type.rebuild_buffers?.();
    }
  }

  static refresh_entities() {
    if (this.needs_entity_refresh) {
      SharedEntityMetadataBuffer.write();
      for (let i = 0; i < EntityQuery.query_cache.length; i++) {
        EntityQuery.query_cache[i].update_matching_entities();
      }
      this.needs_entity_refresh = false;
    }
  }

  static process_query_changes() {
    for (let i = 0; i < EntityQuery.query_cache.length; i++) {
      EntityQuery.query_cache[i].process_entity_changes();
    }
  }

  static get_entity_image_buffer() {
    return Buffer.create({
      name: entity_image_buffer_name,
      raw_data: this.get_entity_count() * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }
}

// Example usage
// const entity_manager = new EntityManager();

// const entity1 = EntityManager.create_entity();
// EntityManager.add_fragment(entity1, PositionFragment);
// EntityManager.add_fragment(entity1, VelocityFragment);

// const entity2 = EntityManager.create_entity();
// EntityManager.add_fragment(entity2, PositionFragment);

// const query = EntityManager.create_query({ fragment_requirements: [PositionFragment, VelocityFragment] });

// // System example: update positions
// function update_positions(delta_time) {
//     const positions = EntityManager.get_fragment_array(PositionFragment);
//     const velocities = EntityManager.get_fragment_array(VelocityFragment);

//     for (const entity of query) {
//         positions.x[entity] += velocities.vx[entity] * delta_time;
//         positions.y[entity] += velocities.vy[entity] * delta_time;
//     }
// }

// // Usage
// update_positions(0.16);
