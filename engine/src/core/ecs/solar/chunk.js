import { EntityLinearDataContainer } from "./memory.js";
import { Name } from "../../../utility/names.js";

/**
 * Represents a memory chunk in the solar ECS system that stores entity fragment data.
 *
 *            ┌────── Archetype Chunk (fixed N, e.g. 256 logical entities) ──────┐
 * SharedBuffer ──► [Position.x N] [Position.y N] … [Velocity.y N] … [Meta N]
 *                                                                  ▲
 * Component views (Float32Array, Int32Array…) slice this one SAB ──┘
 *
 * Chunks are the fundamental storage unit in solar ECS. Each chunk:
 * - Stores data for entities with identical fragment composition (archetype)
 * - Manages memory in contiguous ranges for cache-friendly access
 * - Provides efficient allocation and deallocation of entity slots
 * - Contains typed array views for fast access to component data
 * - Supports metadata for entity generation and instance counts
 *
 * Key responsibilities:
 * - Efficient memory management with free-list allocation
 * - Storage of fragment data in type-homogeneous arrays
 * - Tracking of entity metadata (generation, instance counts)
 * - Defragmentation to optimize memory layout
 *
 * The chunk system enables cache-friendly iteration over entities with
 * identical fragment layouts, which is a core performance optimization
 * in the solar ECS architecture.
 */
export class Chunk {
  /**
   * Set of dirty chunks.
   * @type {Set<Chunk>}
   */
  static dirty = new Set();
  /**
   * Free chunk indices.
   * @type {Uint32Array}
   */
  static free_chunk_indices = [];
  /**
   * Next chunk index.
   * @type {number}
   */
  static next_chunk_index = 0;
  /**
   * All chunks.
   * @type {Chunk[]}
   */
  static all_chunks = [];

  /**
   * Creates a new chunk with the specified fragments and capacity.
   *
   * @param {Fragments[]} fragments - The fragments to include in the chunk
   * @param {number} [capacity=256] - The maximum number of rows the chunk can hold
   */
  constructor(fragments, capacity = 256) {
    this.fragments = fragments;
    this.capacity = capacity;
    this.free_ranges = [0, capacity]; 
    this.fragment_views = Object.create(null);
    this.variable_stores = new Map();

    this.chunk_index = Chunk.free_chunk_indices.length
      ? Chunk.free_chunk_indices.pop()
      : Chunk.next_chunk_index++;

    Chunk.all_chunks.length = Math.max(Chunk.all_chunks.length, this.chunk_index + 1);
    Chunk.all_chunks[this.chunk_index] = this;

    this._build_SAB(fragments, capacity);
    this._initialize_variable_stores(fragments);
  }

  destroy() {
    Chunk.all_chunks[this.chunk_index] = null;
    Chunk.free_chunk_indices.push(this.chunk_index);
    this.buffer = null;
    this.fragments = null;
    this.free_ranges = null;
    this.fragment_views = null;
    this.variable_stores = null;
    this.chunk_index = null;
  }

  /**
   * Allocate contiguous rows in the chunk (O(1))
   * @param {number} [requested_row_count=1] - Number of rows to allocate
   * @returns {number} - Index of the first allocated row
   */
  claim_rows(requested_row_count = 1) {
    for (let range_index = this.free_ranges.length - 2; range_index >= 0; range_index -= 2) {
      const range_start = this.free_ranges[range_index];
      const range_size = this.free_ranges[range_index + 1];
      if (range_size >= requested_row_count) {
        this.free_ranges[range_index + 1] -= requested_row_count;
        const allocated_row_index = range_start + range_size - requested_row_count;
        if (this.free_ranges[range_index + 1] === 0) this.free_ranges.splice(range_index, 2);
        return allocated_row_index;
      }
    }
    return null;
  }

  /**
   * Free contiguous rows in the chunk (O(1))
   * @param {number} released_row_index - Index of the first released row
   * @param {number} released_row_count - Number of rows to release
   */
  free_rows(released_row_index, released_row_count) {
    this.free_ranges.push(released_row_index, released_row_count);
  }

  /**
   * Optional compaction when holes accumulate
   */
  defragment() {
    if (this.free_ranges.length <= 2) return; // already dense

    // --- snapshot old state ---
    const old_fragment_views = this.fragment_views;
    const old_icnt_meta = this.icnt_meta.slice();
    const old_gen_meta = this.gen_meta.slice();
    const old_flags_meta = this.flags_meta.slice();
    const old_capacity = this.capacity;

    // precompute non-container fields for each fragment
    const fragment_field_entries = this.fragments.map(fragment => {
      const entries = Object.entries(fragment.fields)
        .filter(([, spec]) => !spec.is_container);
      return { id: fragment.id, entries };
    });

    // rebuild the SharedArrayBuffer and views in-place
    this._build_SAB(this.fragments, old_capacity);

    // reset free_ranges; we'll re-add the trailing hole below
    this.free_ranges = [];

    // grab new views/metadata handles
    const new_fragment_views = this.fragment_views;
    const new_icnt_meta = this.icnt_meta;
    const new_gen_meta = this.gen_meta;
    const new_flags_meta = this.flags_meta;

    let dest_row_index = 0;
    // walk old rows, copy any used blocks
    for (let src_row_index = 0; src_row_index < old_capacity; ) {
      const block_count = old_icnt_meta[src_row_index];
      if (!block_count) {
        src_row_index++;
        continue;
      }

      // copy each fragment's fixed fields for this block
      for (let i = 0; i < fragment_field_entries.length; i++) {
        const { id: frag_id, entries } = fragment_field_entries[i];

        const src_views = old_fragment_views[frag_id];
        const dst_views = new_fragment_views[frag_id];

        for (let j = 0; j < entries.length; j++) {
          const [field_name, spec] = entries[j];

          const el_count = spec.elements;
          dst_views[field_name].set(
            src_views[field_name].subarray(
              src_row_index * el_count,
              (src_row_index + block_count) * el_count
            ),
            dest_row_index * el_count
          );
        }
      }

      // copy the metadata (instance count, generation, flags) at block start
      new_icnt_meta[dest_row_index]  = block_count;
      new_gen_meta[dest_row_index]   = old_gen_meta[src_row_index];
      new_flags_meta[dest_row_index] = old_flags_meta[src_row_index];

      // advance both pointers by the block size
      src_row_index  += block_count;
      dest_row_index += block_count;
    }

    // anything left at the end is a single free range
    if (dest_row_index < old_capacity) {
      this.free_ranges.push(dest_row_index, old_capacity - dest_row_index);
    }
  }

  /**
   * Marks this chunk as dirty.
   */
  mark_dirty() {
    Chunk.dirty.add(this);
    this.dirty = true;
  }

  /**
   * Clears the dirty flag for this chunk. Called after flushing.
   */
  clear_dirty() {
    Chunk.dirty.delete(this);
    this.dirty = false;
  }

  /**
   * Get the fragment view for the given fragment ID.
   * @param {typeof import('../fragment.js').Fragment} fragment_class - The fragment class to get the view for.
   * @returns {Object} - The fragment view.
   */
  get_fragment_view(fragment_class) {
    return this.fragment_views[fragment_class.id];
  }

  /**
   * Initializes the variable stores based on fragment definitions.
   * @param {Fragments[]} fragments
   */
  _initialize_variable_stores(fragments) {
    this.variable_stores.clear();

    for (let i = 0; i < fragments.length; i++) {
      const fragment = fragments[i];

      const fields = Object.entries(fragment.fields);
      for (let j = 0; j < fields.length; j++) {
        const [field_name, field_spec] = fields[j];

        if (field_spec.is_container) {
          this.variable_stores.set(
            Name.from(`${fragment.id}.${field_name}`),
            new EntityLinearDataContainer(field_spec.type.array)
          );
        }
      }
    }
  }

  /**
   * Build SAB + TypedArray views for fixed-size data.
   * @param {Fragments[]} fragments
   * @param {number} buffer_capacity
   */
  _build_SAB(fragments, buffer_capacity) {
    // 1. calculate total byte count for fixed-size data ONLY
    let total_byte_count = 0;
    for (let i = 0; i < fragments.length; i++) {
      const fragment = fragments[i];
      const fields = Object.values(fragment.fields);
      for (let j = 0; j < fields.length; j++) {
        const field_spec = fields[j];
        // Skip fields that use a separate variable store
        if (field_spec.is_container) {
          continue;
        }
        total_byte_count +=
          field_spec.elements * field_spec.ctor.BYTES_PER_ELEMENT * buffer_capacity;
      }
    }
    total_byte_count += buffer_capacity * 1; // generation byte per row
    total_byte_count += buffer_capacity * 4; // instance_count (Uint32) per row
    total_byte_count += buffer_capacity * 2; // flags (Uint16) per row

    // 2. allocate a shared buffer for this chunk
    this.buffer = typeof SharedArrayBuffer === "function" ? new SharedArrayBuffer(total_byte_count) : new ArrayBuffer(total_byte_count);

    // 3. build TypedArray views for fixed-size data ONLY
    let byte_offset = 0;
    for (let i = 0; i < fragments.length; i++) {
      const fragment = fragments[i];
      const fields = Object.entries(fragment.fields);
      const field_views_map = {};

      for (let j = 0; j < fields.length; j++) {
        const [field_name, field_spec] = fields[j];

        // Skip fields that use a separate variable store
        if (field_spec.is_container) {
          continue;
        }

        const field_view = new field_spec.ctor(
          this.buffer,
          byte_offset,
          field_spec.elements * buffer_capacity
        );

        const default_fill_value = Array.isArray(field_spec.default)
          ? field_spec.default[0]
          : (field_spec.default ?? 0);

        field_view.fill(default_fill_value);
        field_views_map[field_name] = field_view;

        byte_offset += field_view.byteLength;
      }

      this.fragment_views[fragment.id] = field_views_map;
    }

    // Metadata remains the same
    this.icnt_meta = new Uint32Array(this.buffer, byte_offset, buffer_capacity);
    byte_offset += buffer_capacity * 4;
    this.gen_meta = new Uint8Array(this.buffer, byte_offset, buffer_capacity);
    byte_offset += buffer_capacity;
    this.flags_meta = new Uint16Array(this.buffer, byte_offset, buffer_capacity);
  }
}
