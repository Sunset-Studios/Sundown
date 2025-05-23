import { DEFAULT_CHUNK_CAPACITY } from "./types.js";
import { Chunk } from "./chunk.js";
import { Query } from "./query.js";
import { Name } from "../../../utility/names.js";

/**
 * Represents an archetype in the ECS system.
 *
 * An archetype is a collection of fragments that define the fragment
 * types of an entity. It is used to manage the storage and retrieval of
 * fragment data for a specific set of fragment types, to ensure uniform
 * memory layout for all entities with the same fragment types and
 * efficient GPU buffer allocation.
 *
 * Key responsibilities:
 * - Storage of fragment data in chunks
 * - Management of chunk allocation and management
 *
 */
export class Archetype {
  static archetype_cache = new Map();

  constructor(fragments) {
    this.fragments = [...fragments];
    this.id = Archetype.get_id(fragments);
    this.chunks = [new Chunk(this.fragments)];
  }

  /**
   * Find space or create new chunks
   * @param {number} [instance_count=1] - Number of instances to allocate
   * @returns {Object} - Object containing chunk and slot information
   */
  claim(instance_count = 1) {
    let claimed = null;

    // 1. find existing chunks if possible
    for (let i = 0; i < this.chunks.length; i++) {
      const existing_chunk = this.chunks[i];
      const allocation_slot = existing_chunk.claim_rows(instance_count);
      if (allocation_slot !== null) {
        claimed = { chunk: existing_chunk, slot: allocation_slot };
        break;
      }
    }

    // 2. allocate new chunk
    if (!claimed) {
      const chunk = new Chunk(this.fragments);
      const slot = chunk.claim_rows(instance_count);
      this.chunks.push(chunk);
      claimed = { chunk, slot };
    }

    if (claimed) {
      // mark the chunk as dirty so that the GPU buffers are updated
      claimed.chunk.mark_dirty();
    }

    return claimed;
  }

  /**
   * Claim segments of the archetype.
   * @param {number} [instance_count=1] - Number of instances to allocate
   * @returns {Object[]} - Array of objects containing chunk and slot information
   */
  claim_segments(instance_count = 1) {
    const segments = [];
    let remaining = instance_count;
    while (remaining > 0) {
      const max_chunk_size = Math.min(remaining, DEFAULT_CHUNK_CAPACITY);
      const { chunk, slot } = this.claim(max_chunk_size);
      segments.push({ chunk, slot, count: max_chunk_size });
      remaining -= max_chunk_size;
    }
    return segments;
  }

  /**
   * Release a chunk from the archetype.
   * @param {Chunk} chunk - The chunk to release.
   */
  release_chunk(chunk) {
    const idx = this.chunks.indexOf(chunk);
    if (idx !== -1) {
      this.chunks.splice(idx, 1);
    }
    chunk.defragment();
  }

  /**
   * Check if the archetype fullfills the fragment requirements.
   * @param {Fragment[]} fragment_requirements - The fragment requirements to check.
   * @returns {boolean} True if the archetype fullfills the fragment requirements, false otherwise.
   */
  fullfills_fragment_requirements(fragment_requirements) {
    return fragment_requirements.every((fragment) => this.fragments.includes(fragment));
  }

  /**
   * Create a new archetype with the given fragments.
   * @param {Fragment[]} fragments - The fragments to include in the archetype.
   * @returns {Archetype} The new archetype.
   */
  static create(fragments) {
    const id = this.get_id(fragments);
    if (this.archetype_cache.has(id)) {
      return this.archetype_cache.get(id);
    }

    const archetype = new Archetype(fragments);
    this.archetype_cache.set(id, archetype);

    Query.update_archetypes(archetype);

    return archetype;
  }

  /**
   * Get an archetype by its fragments.
   * @param {Fragment[]} fragments - The fragments to include in the archetype.
   * @returns {Archetype} The archetype.
   */
  static get(fragments) {
    const id = this.get_id(fragments);
    if (this.archetype_cache.has(id)) {
      return this.archetype_cache.get(id);
    }
    return null;
  }

  /**
   * Get all archetypes that contain the specified fragments.
   * @param {Fragment[]} fragments - The fragments to include in the archetype.
   * @returns {Archetype[]} The archetypes with the specified fragments.
   */
  static with(fragments) {
    if (fragments.length === 0) {
      return Array.from(this.archetype_cache.values());
    }

    // Get all archetypes that contain the specified fragments
    const fragment_ids = fragments.map((fragment) => fragment.id);
    // Filter the cache to find archetypes containing all the requested fragments
    const matching_archetypes = [];
    // Iterate over all archetypes in the cache
    const cache_entries = this.archetype_cache.values();
    for (let i = 0; i < cache_entries.length; i++) {
      const archetype = cache_entries[i];
      const archetype_fragment_ids = archetype.fragments.map((fragment) => fragment.id);
      // Check if all requested fragments are present in this archetype
      const has_all_fragments = fragment_ids.every((id) => archetype_fragment_ids.includes(id));
      if (has_all_fragments) {
        matching_archetypes.push(archetype);
      }
    }

    return matching_archetypes;
  }

  /**
   * Get the ID of an archetype by its fragments.
   * @param {Fragment[]} fragments - The fragments to include in the archetype.
   * @returns {string} The ID.
   */
  static get_id(fragments) {
    return Name.from(
      fragments
        .sort((a, b) => a.id - b.id)
        .map((fragment) => fragment.id)
        .join("|")
    );
  }
}
