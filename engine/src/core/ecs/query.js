import { EntityID } from "./entity.js";
import { TypedVector } from "../../memory/container.js";
import { profile_scope } from "../../utility/performance.js";

export const EntityMasks = {
  Added: 0x1, // Flag for newly added entities
  Removed: 0x2, // Flag for newly removed entities
};

export class EntityQuery {
  static query_cache = [];

  #seen_entities = new Set();
  #matching_count = 0;
  previous_entity_count = 0;

  constructor(entity_manager, fragment_requirements) {
    this.entity_manager = entity_manager;
    this.fragment_requirements = fragment_requirements;
    this.matching_entities = new TypedVector(256, -1, Float64Array);
    this.matching_entity_ids = new TypedVector(256, -1, Uint32Array);
    this.matching_entity_instance_counts = new TypedVector(256, -1, Uint32Array);
    this.entity_states = new TypedVector(256, 0, Uint32Array);
    this.entities_to_filter = new TypedVector(256, -1, Float64Array);
    this.update_matching_entities();
  }

  _check_entity_fragment_requirements(entity) {
    return this.fragment_requirements.every((fragment_type) =>
      this.entity_manager.entity_fragments.get(entity)?.has(fragment_type)
    );
  }

  update_matching_entities() {
    profile_scope("EntityQuery.update_matching_entities", () => {
      this.matching_entities.union(this.entity_manager.get_entities());

      const new_matching_entities = new Float64Array(this.matching_entities.length);
      const new_matching_entity_ids = new Uint32Array(this.matching_entities.length);
      const new_matching_entity_instance_counts = new Uint32Array(this.matching_entities.length);
      const new_entity_states = new Uint32Array(this.matching_entities.length);

      this.#matching_count = 0;
      for (let i = 0; i < this.matching_entities.length; i++) {
        const entity = this.matching_entities.get(i);

        const passes_requirements = this._check_entity_fragment_requirements(entity);
        const in_seen_entities = this.#seen_entities.has(entity);

        if (passes_requirements && !in_seen_entities) {
          this.#seen_entities.add(entity);
          new_entity_states[this.#matching_count] = EntityMasks.Added;
          new_matching_entities[this.#matching_count] = entity;
          new_matching_entity_ids[this.#matching_count] = EntityID.get_absolute_index(entity);
          new_matching_entity_instance_counts[this.#matching_count] =
            EntityID.get_instance_count(entity);
          this.#matching_count++;
        } else if (!passes_requirements && in_seen_entities) {
          this.#seen_entities.delete(entity);
          new_entity_states[this.#matching_count] = EntityMasks.Removed;
          new_matching_entities[this.#matching_count] = entity;
          new_matching_entity_ids[this.#matching_count] = EntityID.get_absolute_index(entity);
          new_matching_entity_instance_counts[this.#matching_count] =
            EntityID.get_instance_count(entity);
          this.entities_to_filter.push(entity);
          this.#matching_count++;
        } else if (passes_requirements) {
          new_matching_entities[this.#matching_count] = entity;
          new_matching_entity_ids[this.#matching_count] = EntityID.get_absolute_index(entity);
          new_matching_entity_instance_counts[this.#matching_count] =
            EntityID.get_instance_count(entity);
          this.#matching_count++;
        }
      }

      this.entity_states.set_data(new_entity_states.slice(0, this.#matching_count));
      this.matching_entities.set_data(new_matching_entities.slice(0, this.#matching_count));
      this.matching_entity_ids.set_data(new_matching_entity_ids.slice(0, this.#matching_count));
      this.matching_entity_instance_counts.set_data(
        new_matching_entity_instance_counts.slice(0, this.#matching_count)
      );
    });
  }

  process_entity_changes() {
    profile_scope("EntityQuery.process_entity_changes", () => {
      if (this.entities_to_filter.length === 0) return;

      this.#matching_count -= this.entities_to_filter.length;

      const entities_to_filter_data = this.entities_to_filter.get_data();
      for (let i = 0; i < this.entities_to_filter.length; i++) {
        const entity = entities_to_filter_data[i];
        const index = this.matching_entities.index_of(entity);
        if (index !== -1) {
          this.matching_entities.remove(index);
          this.entity_states.remove(index);
        }
      }

      this.entities_to_filter.clear();
    });
  }

  static create(entity_manager, fragment_requirements) {
    let query = this.query_cache.find(query => {
      if (query.fragment_requirements.length !== fragment_requirements.length) {
        return false;
      }
      return query.fragment_requirements.every(req => 
        fragment_requirements.includes(req)
      );
    });

    if (query) {
      return query;
    }

    query = new EntityQuery(entity_manager, fragment_requirements);
    this.query_cache.push(query);
    return query;
  }
}
