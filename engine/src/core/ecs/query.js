import { profile_scope } from "../../utility/performance.js";
import _ from "lodash";

export const EntityMasks = {
  Added: 0x1, // Flag for newly added entities
  Removed: 0x2, // Flag for newly removed entities
};

export class EntityQuery {
  #seen_entities = new Set();
  #matching_count = 0;
  previous_entity_count = 0;

  constructor(entity_manager, fragment_requirements) {
    this.entity_manager = entity_manager;
    this.fragment_requirements = fragment_requirements;
    this.matching_entities = new Uint32Array();
    this.entity_states = new Uint32Array();
    this.entities_to_filter = [];
    this.update_matching_entities();
  }

  _check_entity_fragment_requirements(entity) {
    return this.fragment_requirements.every((fragment_type) =>
      this.entity_manager.entity_fragments.get(entity)?.has(fragment_type)
    );
  }

  update_matching_entities() {
    profile_scope("EntityQuery.update_matching_entities", () => {
      const entities = _.union(this.entity_manager.get_entities(), this.matching_entities);

      const new_matching_entities = new Uint32Array(entities.length);
      const new_entity_states = new Uint32Array(entities.length);

      this.#matching_count = 0;
      for (let i = 0; i < entities.length; i++) {
        const entity = entities[i];

        new_matching_entities[this.#matching_count] = entity;

        const passes_requirements =
          this._check_entity_fragment_requirements(entity);
        const in_seen_entities = this.#seen_entities.has(entity);

        if (passes_requirements && !in_seen_entities) {
          this.#seen_entities.add(entity);
          new_entity_states[this.#matching_count] = EntityMasks.Added;
        } else if (!passes_requirements && in_seen_entities) {
          this.#seen_entities.delete(entity);
          new_entity_states[this.#matching_count] = EntityMasks.Removed;
          this.entities_to_filter.push(entity);
        }

        this.#matching_count++;
      }

      this.matching_entities = new_matching_entities.slice(
        0,
        this.#matching_count
      );
      this.entity_states = new_entity_states.slice(0, this.#matching_count);
    });
  }

  process_entity_changes() {
    profile_scope("EntityQuery.process_entity_changes", () => {
      if (this.entities_to_filter.length === 0) return;

      this.#matching_count -= this.entities_to_filter.length;

      for (let i = 0; i < this.entities_to_filter.length; i++) {
        const entity = this.entities_to_filter[i];
        const index = this.matching_entities.indexOf(entity);
        if (index !== -1) {
          this.matching_entities = this.matching_entities.filter(
            (_, idx) => idx !== index
          );
          this.entity_states = this.entity_states.filter(
            (_, idx) => idx !== index
          );
        }
      }

      this.entities_to_filter.length = 0;
    });
  }
}
