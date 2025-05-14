import { Archetype } from "./archetype";

/**
 * @typedef {Object} Query
 * @property {string[]} fragment_requirements - The fragment requirements for the query.
 * @property {Archetype[]} archetypes - The archetypes that match the query.
 */
export class Query {
  static query_cache = new Map();

  /**
   * @param {Archetype[]} archetypes - The archetypes that match the query.
   * @param {string[]} fragment_requirements - The fragment requirements for the query.
   */
  constructor(archetypes, fragment_requirements) {
    this.archetypes = archetypes;
    this.fragment_requirements = fragment_requirements;
  }

  /**
   * Hot iterator: no generators, no allocations
   * @param {function} callback - The callback function to execute for each archetype.
   */
  for_each(callback) {
    for (let i = 0; i < this.archetypes.length; i++) {
      const archetype = this.archetypes[i];

      for (let k = 0; k < archetype.chunks.length; k++) {
        const target_chunk = archetype.chunks[k];
        const instance_counts = target_chunk.icnt_meta;

        for (let slot_index = 0; slot_index < target_chunk.capacity; ) {
          const instance_count = instance_counts[slot_index];
          if (instance_count) {
            callback(target_chunk, slot_index, instance_count, archetype);
          }
          slot_index += instance_count > 0 ? instance_count : 1;
        }
      }
    }
  }
  
  /**
   * Updates the archetypes for the query.
   * @param {Archetype} new_archetype - The new archetype to add to the query.
   */
  static update_archetypes(new_archetype) {
    const query_cache_entries = this.query_cache.entries();
    for (const [query_id, query] of query_cache_entries) {
      if (new_archetype.fullfills_fragment_requirements(query.fragment_requirements)) {
        query.archetypes.push(new_archetype);
      }
    }
  }

  /**
   * Creates a new query with the given fragment requirements.
   * @param {string[]} fragment_requirements - The fragment requirements to use for the query.
   * @returns {Query} The new query.
   */
  static create(fragment_requirements) {
    const query_id = Archetype.get_id(fragment_requirements);

    let query = this.query_cache.get(query_id);
    if (query) {
      return query;
    }

    const archetypes = Archetype.with(fragment_requirements);
    query = new Query(archetypes, fragment_requirements);
    this.query_cache.set(query_id, query);
    return query;
  }
}
