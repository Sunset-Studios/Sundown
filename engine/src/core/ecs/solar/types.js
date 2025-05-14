import { FreeListAllocator } from "../../../memory/allocator.js";

// 32-bit entity ID ─ 10 bits local slot | 18 bits chunk index | 4 bits generation
// Maximum logical entity count = 2^ROW_BITS = 268,435,456
export const ROW_BITS = 28;
export const LOCAL_SLOT_BITS = 7;
export const CHUNK_INDEX_BITS = ROW_BITS - LOCAL_SLOT_BITS; // = 21
export const GEN_BITS = 4;
export const ROW_MASK = (1 << ROW_BITS) - 1;
export const LOCAL_SLOT_MASK = (1 << LOCAL_SLOT_BITS) - 1;
export const CHUNK_INDEX_MASK = ((1 << CHUNK_INDEX_BITS) - 1) << LOCAL_SLOT_BITS;
export const GEN_MASK = (1 << GEN_BITS) - 1;

/**
 * Manages entity IDs.
 * @class EntityID
 */
export class EntityID {
  static get_row_index(entity) {
    return this.unpack(entity).row_index;
  }

  static get(chunk, slot) {
    const generation = chunk.gen_meta[slot];
    const local_slot = slot;
    const chunk_index = chunk.chunk_index;
    return this.make(local_slot, chunk_index, generation);
  }

  /**
   * Compose the 28-bit row field from chunk_index and local_slot.
   * Low LOCAL_SLOT_BITS bits = local_slot; next CHUNK_INDEX_BITS bits = chunk_index.
   */
  static make_row_field(local_slot, chunk_index) {
    return ((chunk_index << LOCAL_SLOT_BITS) | (local_slot & LOCAL_SLOT_MASK)) & ROW_MASK;
  }

  /**
   * Extract the 28-bit row field (chunk_index+local_slot) from a full entity ID.
   */
  static get_row_field(eid) {
    return eid & ROW_MASK;
  }

  /**
   * Build an ID combining:
   *   - LOCAL_SLOT_BITS bits of local_slot,
   *   - CHUNK_INDEX_BITS bits of chunk_index,
   *   - GEN_BITS bits of generation.
   */
  static make(local_slot, chunk_index, generation) {
    const row_field  = EntityID.make_row_field(local_slot, chunk_index);
    const gen_part   = (generation & GEN_MASK) << ROW_BITS;
    return row_field | gen_part;
  }

  /**
   * Break an ID into its components.
   */
  static unpack(eid) {
    const row_field  = eid & ROW_MASK;
    return {
      row_index:   row_field,
      local_index: row_field & LOCAL_SLOT_MASK,
      chunk_index: (row_field & CHUNK_INDEX_MASK) >> LOCAL_SLOT_BITS,
      generation: (eid >> ROW_BITS) & GEN_MASK,
    };
  }
}

/**
 * A thin wrapper around your 32‐bit u32 ID.
 * Callers hold onto this object; we can swap out its `.id` behind the scenes.
 */
export class EntityHandle {
  static handle_allocator = new FreeListAllocator(1024, new EntityHandle());
  static id_to_handle_map = new Map();

  constructor(initial_id) {
    this._entity_id = initial_id;
    this._archetype = null;
    this._segments = [];
    this._instance_count = 0;
  }

  get id() {
    return this._entity_id;
  }

  get archetype() {
    return this._archetype;
  }

  get segments() {
    return this._segments;
  }

  get instance_count() {
    return this._instance_count;
  }

  set id(new_id) {
    EntityHandle.id_to_handle_map.set(this._entity_id, null);
    this._entity_id = new_id;
    EntityHandle.id_to_handle_map.set(this._entity_id, this);
  }

  set archetype(new_archetype) {
    this._archetype = new_archetype;
  }

  set segments(new_segments) {
    this._segments = new_segments;
  }

  set instance_count(new_instance_count) {
    this._instance_count = new_instance_count;
  }

  valueOf() {
    return this._entity_id;
  }

  toString() {
    return `${this._entity_id}`;
  }

  static create(id = 0, archetype = null, segments = [], instance_count = 0) {
    const handle = this.handle_allocator.allocate();
    handle.id = id;
    handle.archetype = archetype;
    handle.segments = segments;
    handle.instance_count = instance_count;
    return handle;
  }

  static destroy(handle) {
    handle.id = 0;
    handle.archetype = null;
    handle.segments = [];
    handle.instance_count = 0;
    this.handle_allocator.deallocate(handle);
  }

  static get(id) {
    return EntityHandle.id_to_handle_map.get(id);
  }

  static get_or_create(id) {
    const handle = EntityHandle.get(id);
    if (!handle) {
      return EntityHandle.create(id);
    }
    return handle;
  }
}
