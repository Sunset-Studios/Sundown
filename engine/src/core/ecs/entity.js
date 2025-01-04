import { SharedEntityMetadataBuffer } from "../shared_data.js";
import { EntityQuery } from "./query.js";
import { Vector } from "../../memory/container.js";

// The semantics of the entity id are as follows:
//
// entity_index: 32 bits
// instance_count: 32 bits
//
// The entity index is the index of the entity in the entity manager.
// The instance count is the number of instances of that entity, where an entity is like a more granular archetype.
// The entity index is used to identify the entity, and the instance count is used internally by fragments to manage buffer data.
// This usually means that entity instances run contiguously after the entity index to optimize for cache performance.
// Deleting an entity should wholesale free all of its instances as well.
// You can duplicate or get data for a specific instance of an entity.
// All other operations are done across all instances of an entity.
// It is the responsibility of the caller to manage the returned entity id and to manage the instance count via calls to change_entity_instance_count.
//
// TODO: Maybe find a better way to internally manage instance counts for entities? Divergences can occur if the application stores multiple copies
// of the entity id with differing instance counts, which can lead to undefined behavior.

const entity_image_buffer_name = "entity_image_buffer";
const object_name = "object";

export class EntityID {
  static get_absolute_index(entity) {
    return SharedEntityMetadataBuffer.get_entity_offset(entity);
  }

  static get_instance_count(entity) {
    return SharedEntityMetadataBuffer.get_entity_count(entity);
  }
}

export class EntityManager {
  static next_entity_id = 0;
  static entity_fragments = new Map();
  static fragment_types = new Set();
  static entities = new Vector(256, Float64Array);
  static deleted_entities = new Set();
  static queries = [];

  static reserve_entities(size) {
    SharedEntityMetadataBuffer.reserve(size);

    for (const fragment_type of this.fragment_types) {
      fragment_type.resize?.(size);
    }
  }

  static create_entity(refresh_entity_data = true) {
    let entity;

    if (this.deleted_entities.size > 0) {
      entity = this.deleted_entities.values().next().value;
      this.deleted_entities.delete(entity);
    } else {
      entity = this.next_entity_id++;
    }

    SharedEntityMetadataBuffer.add_entity(entity);

    // Resize all fragment data arrays to fit the new entity
    if (refresh_entity_data) {
      for (const fragment_type of this.fragment_types) {
        fragment_type.resize?.(entity);
      }
    }

    this.entities.push(entity);
    this.entity_fragments.set(entity, new Set());
    if (refresh_entity_data) {
      this.update_queries();
    }

    return entity;
  }

  static delete_entity(entity, refresh_entity_data = true) {
    if (!this.entity_fragments.has(entity)) {
      return;
    }
    for (const FragmentType of this.entity_fragments.get(entity)) {
      FragmentType.remove_entity?.(entity);
    }
    this.entity_fragments.delete(entity);
    this.entities.remove(this.entities.index_of(entity));
    this.deleted_entities.add(entity);
    if (refresh_entity_data) {
      this.update_queries();
    }
  }

  static duplicate_entity(entity, refresh_entity_data = true, instance = 0) {
    const new_entity = this.create_entity(refresh_entity_data);

    for (const FragmentType of this.entity_fragments.get(entity)) {
      if (FragmentType.data) {
        const data = FragmentType.duplicate_entity_data?.(entity, instance);
        const new_frag_view = this.add_fragment(new_entity, FragmentType, refresh_entity_data);
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
        this.add_tag(new_entity, FragmentType, refresh_entity_data);
      }
    }

    return new_entity;
  }

  static add_fragment(entity, FragmentType, refresh_entity_data = true) {
    if (!this.fragment_types.has(FragmentType)) {
      FragmentType.initialize();
      this.fragment_types.add(FragmentType);
    }
    const fragment_view = FragmentType.add_entity(entity);
    this.entity_fragments.get(entity).add(FragmentType);
    if (refresh_entity_data) {
      this.update_queries();
    }
    return fragment_view;
  }

  static remove_fragment(entity, FragmentType, refresh_entity_data = true) {
    if (
      !this.entity_fragments.has(entity) ||
      !this.entity_fragments.get(entity).has(FragmentType)
    ) {
      return;
    }
    FragmentType.remove_entity(entity);
    this.entity_fragments.get(entity).delete(FragmentType);
    if (refresh_entity_data) {
      this.update_queries();
    }
  }

  static add_tag(entity, Tag, refresh_entity_data = true) {
    if (!this.fragment_types.has(Tag)) {
      this.fragment_types.add(Tag);
    }
    this.entity_fragments.get(entity).add(Tag);
    if (refresh_entity_data) {
      this.update_queries();
    }
  }

  static remove_tag(entity, Tag, refresh_entity_data = true) {
    if (!this.entity_fragments.has(entity) || !this.entity_fragments.get(entity).has(Tag)) {
      return;
    }
    this.entity_fragments.get(entity).delete(Tag);
    if (refresh_entity_data) {
      this.update_queries();
    }
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

  static get_fragment_array(FragmentType) {
    return FragmentType.data;
  }

  static get_entity_count() {
    return this.entities.length;
  }

  static get_entity_instance_count(entity) {
    return EntityID.get_instance_count(entity);
  }

  static change_entity_instance_count(entity, instance_count) {
    if (!this.entity_fragments.has(entity)) {
      return null;
    }

    SharedEntityMetadataBuffer.set_entity_count(entity, instance_count);

    for (const fragment_type of this.fragment_types) {
      fragment_type.resize?.(this.next_entity_id - 1);
      fragment_type.entity_instance_count_changed?.(entity);
    }

    return entity;
  }

  static get_entities() {
    return this.entities;
  }

  static create_query({ fragment_requirements }) {
    const query = new EntityQuery(this, fragment_requirements);
    this.queries.push(query);
    return query;
  }

  static update_queries() {
    for (let i = 0; i < this.queries.length; i++) {
      this.queries[i].update_matching_entities();
    }
  }

  static process_query_changes() {
    for (let i = 0; i < this.queries.length; i++) {
      this.queries[i].process_entity_changes();
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
// EntityManager.add_fragment(entity1, PositionFragment, false /* refresh_entity_data */);
// EntityManager.add_fragment(entity1, VelocityFragment, false /* refresh_entity_data */);

// const entity2 = EntityManager.create_entity();
// EntityManager.add_fragment(entity2, PositionFragment, false /* refresh_entity_data */);

// const query = EntityManager.create_query({ fragment_requirements: [PositionFragment, VelocityFragment] });

// EntityManager.update_queries();

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
