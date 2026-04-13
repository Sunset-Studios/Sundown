import { EntityID, DEFAULT_CHUNK_CAPACITY, ROW_MASK } from "./types.js";
import { Chunk } from "./chunk.js";
import { Renderer } from "../../../renderer/renderer.js";
import { Buffer, BufferSync } from "../../../renderer/buffer.js";
import { npot } from "../../../utility/math.js";
import { Name } from "../../../utility/names.js";
import { log, warn, error } from "../../../utility/logging.js";

const unmapped_state = "unmapped";
const index_map_buffer_name = "entity_index_map";
const entity_flags_buffer_name = "entity_flags";

export class EntityAllocator {
  constructor() {
    this.row_to_entity_id = new Map(); // Map row_index back to the full entity ID (including generation)
    this.row_generation = new Map(); // Tracks current generation for each row_index
  }

  /**
   * Creates a new entity ID and maps it to the target chunk.
   * @param {Chunk} target_chunk - The chunk to allocate the entity in.
   * @param {number} allocation_slot - The slot index in the target chunk.
   * @param {number} instance_count - The number of instances to allocate.
   * @returns {number} The new entity ID.
   */
  create(target_chunk, allocation_slot, instance_count) {
    // Make row index from allocation slot and chunk index.
    const row_index = EntityID.make_row_field(allocation_slot, target_chunk.chunk_index);

    // Reuse the current generation for this row index.
    const generation = this.row_generation.get(row_index);

    // Assemble the entity ID by masking together the allocation slot, chunk index, generation and 1-bit flag.
    const entity_id = EntityID.make(allocation_slot, target_chunk.chunk_index, generation);

    // Map the entity ID to the row index of every instance in the chunk.
    for (let i = 0; i < instance_count; i++) {
      const row = EntityID.make_row_field(allocation_slot + i, target_chunk.chunk_index);
      this.row_to_entity_id.set(row, entity_id);
    }

    // Store metadata in the target chunk.
    target_chunk.gen_meta[allocation_slot] = generation;
    target_chunk.icnt_meta[allocation_slot] = instance_count;

    // Mark the chunk as dirty
    target_chunk.mark_dirty();

    return entity_id;
  }

  /**
   * Updates the allocation record for a stable entity ID when its location changes.
   * Releases the old slot but keeps the entity ID alive.
   *
   * @param {number} entity_id - The stable entity ID being updated.
   * @param {Chunk} old_chunk - The chunk the entity is moving from.
   * @param {number} old_slot - The slot index in the old chunk.
   * @param {Chunk} new_chunk - The chunk the entity is moving to.
   * @param {number} new_slot - The slot index in the new chunk.
   * @param {number} instance_count - The number of instances (remains the same during migration, might change during instance update).
   */
  update_allocation(entity_id, old_chunk, old_slot, new_chunk, new_slot, instance_count) {
    const { row_index: old_row } = EntityID.unpack(entity_id);

    // 1. Release the old slot in the old chunk.
    // We need the instance count associated *with the old slot*.
    // Assuming the caller already copied the data and we just need to free space.
    // The count might have changed if this is called from update_instance_count,
    // but the caller passes the *new* instance_count. The chunk needs the *old* count
    // to free correctly. Let's retrieve it directly.
    const old_instance_count = old_chunk.icnt_meta[old_slot];
    this.free_segment(old_chunk, old_slot, old_instance_count);

    // Bump generation for the row_index. Max 4 bits (16 generations) per row.
    // The & 0x0f ensures it wraps around from 15 back to 0.
    // Consider implications if an entity ID lives longer than 16 destroy/create cycles at the same row_index.
    this.row_generation.set(old_row, (this.row_generation.get(old_row) + 1) & 0x0f);

    // Remove the old row from the row_to_entity_id map.
    // this.row_to_entity_id.delete(old_row);

    // Get the new generation for the new row index.
    const new_row = EntityID.make_row_field(new_slot, new_chunk.chunk_index);
    const new_generation = this.row_generation.get(new_row);

    // Assemble the new entity ID by masking together the allocation slot, chunk index, generation and 1-bit flag.
    const new_entity_id = EntityID.make(new_slot, new_chunk.chunk_index, new_generation);

    // Map the entity ID to the row index of every instance in the chunk.
    for (let i = 0; i < instance_count; i++) {
      const row = EntityID.make_row_field(new_slot + i, new_chunk.chunk_index);
      this.row_to_entity_id.set(row, new_entity_id);
    }

    // 2. Update metadata in the *new* chunk's slot.
    // The generation comes from the stable entity ID.
    new_chunk.gen_meta[new_slot] = new_generation;
    new_chunk.icnt_meta[new_slot] = instance_count;

    // Update the mapping for the new row index.
    this.row_to_entity_id.set(new_row, new_entity_id);

    // Mark the chunks as dirty
    old_chunk.mark_dirty();
    new_chunk.mark_dirty();

    return new_entity_id;
  }

  /**
   * Destroys an entity and releases its resources.
   *
   * @param {number} entity_id - The ID of the entity to destroy
   * @param {Chunk} target_chunk - The chunk containing the entity
   * @param {number} allocation_slot - The slot index in the chunk
   */
  destroy(entity_id, target_chunk, allocation_slot) {
    const { row_index } = EntityID.unpack(entity_id);

    // Check if the ID in the map actually matches before destroying.
    // This prevents issues if destroy is called multiple times or with stale IDs.
    if (this.row_to_entity_id.get(row_index) !== entity_id) {
      warn(
        `Attempted to destroy entity ${entity_id} (row ${row_index}), but it seems already destroyed or row reused.`
      );
      return;
    }

    // Get instance count *before* freeing the row
    const instance_count = target_chunk.icnt_meta[allocation_slot];
    // Free the rows in the chunk
    this.free_segment(target_chunk, allocation_slot, instance_count);

    // Bump generation for the row_index. Max 4 bits (16 generations) per row.
    // The & 0x0f ensures it wraps around from 15 back to 0.
    // Consider implications if an entity lives longer than 16 destroy/create cycles at the same row_index.
    this.row_generation.set(row_index, (this.row_generation.get(row_index) + 1) & 0x0f);

    // Remove the old row from the row_to_entity_id map.
    // this.row_to_entity_id.delete(row_index);

    // Mark the chunk as dirty
    target_chunk.mark_dirty();
  }

  /**
   * Maps an entity ID to a slot in a chunk.
   *
   * @param {number} entity_id - The ID of the entity
   * @param {Chunk} chunk - The chunk containing the entity
   * @param {number} slot - The slot index in the chunk
   * @param {number} count - The number of rows to map
   */
  map_entity_chunk(entity_id, chunk, slot, count) {
    // like create(), but without making a new ID
    const { generation } = EntityID.unpack(entity_id);
    // mark each row index in row_to_entity_id if you need reverse lookup
    for (let i = 0; i < count; i++) {
      const row = EntityID.make_row_field(slot + i, chunk.chunk_index);
      this.row_to_entity_id.set(row, entity_id);
    }
    // set per-slot metadata
    chunk.gen_meta[slot] = generation;
    chunk.icnt_meta[slot] = count;

    chunk.mark_dirty();

  }

  /**
   * Frees a segment of rows in a chunk.
   *
   * @param {Chunk} chunk - The chunk containing the entity
   * @param {number} slot - The slot index in the chunk
   * @param {number} count - The number of rows to free
   */
  free_segment(chunk, slot, count) {
    if (count > 0) {
      chunk.free_rows(slot, count);
      for (let i = 0; i < count; i++) {
        const row = EntityID.make_row_field(slot + i, chunk.chunk_index);
        this.row_to_entity_id.delete(row);
        chunk.flags_meta[slot + i] = 0;
        chunk.icnt_meta[slot + i] = 0;
        chunk.gen_meta[slot + i] = 0;
      }
      chunk.mark_dirty();
    }
  }

  /**
   * Retrieves the entity ID for a given chunk and slot.
   *
   * @param {Chunk} chunk - The chunk containing the entity
   * @param {number} slot - The slot index in the chunk
   * @returns {number} The entity ID
   */
  get_entity_id_for(chunk, slot) {
    const row_field = EntityID.make_row_field(slot, chunk.chunk_index);
    return this.row_to_entity_id.get(row_field);
  }

  /**
   * Retrieves the entity ID for a given row index.
   * @param {number} row - The row index
   * @returns {number} The entity ID
   */
  get_base_entity_id(id) {
    return this.row_to_entity_id.get(id & ROW_MASK);
  }
}

/**
 * FragmentGpuBuffer manages GPU memory for ECS fragment data.
 *
 * This class handles the allocation, synchronization, and management of GPU buffers
 * that store ECS fragment data. It provides mechanisms for:
 *
 * - Creating and managing GPU buffers for fragment fields
 * - Tracking chunk-to-buffer mappings for efficient updates
 * - Synchronizing data between CPU and GPU memory
 * - Managing buffer resizing when entity counts grow
 * - Supporting both combined and individual field buffer layouts
 *
 * Each FragmentGpuBuffer instance represents a single buffer on the GPU that
 * may contain data for multiple chunks. The system tracks base row offsets
 * for each chunk to enable efficient updates when entity data changes.
 *
 * The class supports optional CPU readback buffers for cases where GPU data
 * needs to be read back to the CPU for processing.
 */

export class FragmentGpuBuffer {
  static all_buffers = []; // FragmentGpuBuffer[]
  static entity_index_map_buffer = null;
  static entity_flags_buffer = null;
  static initial_max_rows = 1024;
  static need_full_flush = true;
  static invalid_row = 0xffffffff;

  /**
   * @param {string} name - base name for GPU and CPU buffers
   * @param {number} max_rows - initial number of rows
   * @param {number} byte_stride - bytes per row
   * @param {boolean} [cpu_readback=false] - whether to create CPU readback buffers
   * @param {boolean} [dispatch=true] - whether to dispatch events when buffer changes
   * @param {number} [usage=GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST] - GPU buffer usage flags
   * @param {typeof import('../fragment.js').Fragment | null} [fragment_class_ref=null] - Reference to the fragment class
   * @param {string | null} [config_key_within_fragment=null] - Key for buffer/field config within the fragment class
   * @param {function | null} [sync_target_accessor=null] - Function to get target view for sync operation
   * @param {boolean} [global_binding=false] - Whether this buffer is bound globally
   * @param {number} [capacity_multiplier=1] - Multiplier applied to capacity on resize
   */
  constructor(
    name,
    max_rows,
    byte_stride,
    cpu_readback = false,
    dispatch = true,
    usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    fragment_class_ref = null,
    config_key_within_fragment = null,
    sync_target_accessor = null,
    global_binding = false,
    capacity_multiplier = 1
  ) {
    this.name = name;
    this.max_rows = max_rows;
    this.byte_stride = byte_stride;
    this.usage = usage;
    this.dispatch = dispatch;
    this.cpu_readback = cpu_readback;
    this.fragment_class_ref = fragment_class_ref;
    this.config_key_within_fragment = config_key_within_fragment;
    this.sync_target_accessor = sync_target_accessor;
    this.global_binding = global_binding;
    this.capacity_multiplier = Math.max(1, Number(capacity_multiplier) || 1);

    // initial GPU buffer
    this.buffer = Buffer.create({
      name: this.name,
      size: this.max_rows * (this.byte_stride / 4) * this.capacity_multiplier,
      usage: this.usage,
      force: true,
      dispatch: this.dispatch,
      cpu_readback: this.cpu_readback,
    });

    // track exact buffer segments (chunk + row range) needing CPU sync
    this.pending_sync_segments = [];

    FragmentGpuBuffer.all_buffers.push(this);
  }

  /**
   * Update a region of the GPU buffer corresponding to a chunk segment.
   * @param {number} base_row - The starting row index in the GPU buffer for this chunk.
   * @param {ArrayBufferView} packed_chunk_data - A TypedArray (e.g., Uint8Array) containing the packed data for all rows of this fragment type within the chunk.
   * @param {number} row_count - The number of rows included in packed_chunk_data.
   */
  update_chunk(base_row, packed_chunk_data, row_count, chunk_for_sync = null) {
    if (!packed_chunk_data) return;

    const byte_offset = base_row * this.byte_stride;
    const write_bytes = row_count * this.byte_stride; // Calculate bytes to write based on rows

    // Ensure packed data size matches expected size
    if (packed_chunk_data.byteLength < write_bytes) {
      error(
        `GPU buffer ${this.name} update failed: Provided data size (${packed_chunk_data.byteLength} bytes) is less than expected size (${write_bytes} bytes) for ${row_count} rows.`
      );
      // Optionally slice or handle error
      // write_bytes = packed_chunk_data.byteLength; // Write only what's provided? Risky.
      return; // Prevent partial/incorrect write
    }

    const required_logical_rows = Math.ceil((byte_offset + write_bytes) / this.byte_stride);
    const required_end_byte = required_logical_rows * this.byte_stride * this.capacity_multiplier;
    const write_elements = Math.ceil(row_count * (this.byte_stride / packed_chunk_data.BYTES_PER_ELEMENT));

    // Grow buffer if needed
    if (required_end_byte > this.buffer.config.size) {
      this._resize_buffer(npot(required_logical_rows)); // Resize based on logical rows; multiplier is applied in _resize_buffer
    }

    // Write the packed data
    this.buffer.write_raw(
      packed_chunk_data, // Source ArrayBufferView
      byte_offset, // Destination offset in GPU buffer (bytes)
      write_elements
    );

    // record exactly which rows need CPU readback
    if (this.cpu_readback && chunk_for_sync && row_count > 0) {
      this.pending_sync_segments.push({ chunk: chunk_for_sync, base_row, row_count });
    }
  }

  /**
   * Read back the CPU buffer for the given frame into 'array'.
   */
  async readback_buffers() {
    // Return early if no CPU readbacks or no pending segments
    if (!this.cpu_readback) return;

    // Map the entire CPU buffer once for reading
    const buffered_frame = Renderer.get().get_buffered_frame_number();
    const cpu_buf = this.buffer.cpu_buffers[buffered_frame];
    if (cpu_buf.mapState !== unmapped_state) return;

    await cpu_buf.mapAsync(GPUMapMode.READ);
    const mapped_range = cpu_buf.getMappedRange();

    if (this.pending_sync_segments.length === 0) return;

    if (this.sync_target_accessor) {
      for (const { chunk, base_row, row_count } of this.pending_sync_segments) {
        const target_view = this.sync_target_accessor(chunk);
        if (!target_view) continue;

        const offset_bytes = base_row * this.byte_stride;
        const slice = new Uint32Array(mapped_range, offset_bytes, row_count);
        target_view.set(slice.subarray(0, Math.min(row_count, target_view.length)), 0);
      }
    } else {
      const frag_id = this.fragment_class_ref.id;
      const field_key = this.config_key_within_fragment;
      const spec = this.fragment_class_ref.fields[field_key];

      for (const { chunk, base_row, row_count } of this.pending_sync_segments) {
        const frag_views = chunk.fragment_views[frag_id];
        if (!frag_views) continue;
        const target_view = frag_views[field_key];
        if (!target_view) continue;

        const offset_bytes = base_row * this.byte_stride;
        const element_count = (row_count * this.byte_stride) / spec.ctor.BYTES_PER_ELEMENT;
        const slice = new spec.ctor(mapped_range, offset_bytes, element_count);
        target_view.set(slice.subarray(0, Math.min(element_count, target_view.length)), 0);
      }
    }

    // Unmap and clear segments
    cpu_buf.unmap();

    this.pending_sync_segments.length = 0;
  }

  _resize_buffer(new_max_rows) {
    if (new_max_rows == this.max_rows) return;

    this.max_rows = new_max_rows;
    const new_size =
      this.max_rows * (this.byte_stride / 4) * this.capacity_multiplier;

    // resize GPU buffer
    this.buffer = Buffer.create({
      name: this.name,
      size: new_size,
      usage: this.usage,
      force: true,
      dispatch: this.dispatch,
      cpu_readback: this.cpu_readback,
    });
    // NOTE: Data from the old buffer is NOT automatically copied here.
    // If resizing needs to preserve existing data, a GPU copy operation
    // from old_buffer to the new this.buffer would be needed before
    // releasing the old_buffer. For simplicity, assuming buffers are
    // fully updated after resize or that data preservation isn't needed here.

    if (this.global_binding) {
      Renderer.get().refresh_global_shader_bindings();
    }

    // schedule this buffer for a full-chunk update on next flush
    FragmentGpuBuffer.need_full_flush = true;
  }

  /**
   * Initialize the entity index map buffer.
   */
  static init_entity_compaction_index_map() {
    // Create as a FragmentGpuBuffer so we get .cpu_buffers for readback
    this.entity_index_map_buffer = new FragmentGpuBuffer(
      index_map_buffer_name, // name
      this.initial_max_rows, // max_rows
      4, // 4 bytes per row (Uint32)
      false, // cpu_readback
      true, // dispatch
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      null, // fragment_class_ref
      null, // config_key_within_fragment
      null, // sync_target_accessor
      true // global_binding
    );
  }

  /**
   * Initialize GPU buffers based on Fragment class definitions.
   * Creates a buffer per field marked with `gpu_buffer: true` or for custom
   * combined buffer entries under the `gpu_buffers` property.
   *
   * @param {Array<typeof import('../fragment.js').Fragment>} fragment_classes - Array of fragment classes.
   * @param {object} [options={}] - Options for buffer creation.
   * @param {boolean} [options.dispatch=true] - Whether to dispatch global events.
   */
  static #processed_fields_init = new Set();
  static init_gpu_buffers(fragment_classes, options = {}) {
    const { dispatch = true } = options;

    for (let i = 0; i < fragment_classes.length; i++) {
      const fragment_class = fragment_classes[i];
      // Basic validation
      if (!fragment_class.is_valid()) {
        error(`Sector.init_gpu_buffers: Invalid fragment_class provided.`, fragment_class);
        continue;
      }

      fragment_class.field_key_map.clear();
      this.#processed_fields_init.clear();

      const frag_id = fragment_class.id;

      // 1. Process Custom Combined Buffers (if defined)
      if (fragment_class.gpu_buffers) {
        const gpu_buffer_entries = Object.entries(fragment_class.gpu_buffers);

        for (let i = 0; i < gpu_buffer_entries.length; i++) {
          const [buffer_key, buffer_config] = gpu_buffer_entries[i];

          const has_valid_fields =
            Array.isArray(buffer_config.fields) && buffer_config.fields.length > 0;
          const has_valid_gpu_data =
            typeof buffer_config.gpu_data === "function" ||
            typeof buffer_config.gpu_data === "string";
          const has_valid_global_data = typeof buffer_config.global_data === "function";
          if (!buffer_config || (!has_valid_fields && !has_valid_gpu_data && !has_valid_global_data)) {
            error(
              `Invalid config for custom GPU buffer '${buffer_key}' in fragment '${Name.string(frag_id)}'`
            );
            continue;
          }

          // gather valid fields and mark them processed
          const fields_in_buffer = [];
          if (has_valid_fields) {
            for (let j = 0; j < buffer_config.fields.length; j++) {
              const field_name = buffer_config.fields[j];
              const field_spec = fragment_class.fields[field_name];
              if (!field_spec) {
                error(
                  `Invalid field spec for '${field_name}' in custom buffer '${buffer_key}' of fragment '${Name.string(frag_id)}'`
                );
                continue;
              }
              fields_in_buffer.push(field_name);
              this.#processed_fields_init.add(field_name);
            }
          }

          const valid_buffer =
            buffer_config.stride > 0 &&
            (
              (fields_in_buffer.length > 0 && !has_valid_gpu_data && !has_valid_global_data) ||
              has_valid_gpu_data ||
              has_valid_global_data
            );

          // use offline-computed stride
          if (valid_buffer) {
            const actual_gpu_buffer_key = buffer_config.buffer_name;
            fragment_class.field_key_map.set(buffer_key, actual_gpu_buffer_key);

            // Determine initial row capacity using max buffer_multiplier across included fields
            let effective_multiplier = 1;
            for (let j = 0; j < fields_in_buffer.length; j++) {
              const fname = fields_in_buffer[j];
              const fspec = fragment_class.fields[fname];
              const m = typeof fspec?.buffer_multiplier === "number" ? fspec.buffer_multiplier : 1;
              if (m > effective_multiplier) effective_multiplier = m;
            }

            const flat_buf = new FragmentGpuBuffer(
              actual_gpu_buffer_key,
              FragmentGpuBuffer.initial_max_rows,
              buffer_config.stride,
              buffer_config.cpu_readback,
              dispatch,
              buffer_config.usage,
              fragment_class,
              buffer_key,
              null,
              false,
              effective_multiplier
            );

            fragment_class.buffer_data.set(buffer_config.buffer_name, {
              buffer: flat_buf,
              stride: buffer_config.stride,
            });
          } else {
            warn(
              `Combined buffer '${buffer_key}' for fragment '${Name.string(frag_id)}' resulted in zero stride or no valid fields. Skipping.`
            );
          }
        }
      }

      // 2. Process Remaining Individual Fields
      const field_entries = Object.entries(fragment_class.fields);
      for (let i = 0; i < field_entries.length; i++) {
        const [field_name, field_spec] = field_entries[i];

        // Skip if already in a combined buffer OR not marked for GPU buffer
        if (this.#processed_fields_init.has(field_name) || !field_spec.gpu_buffer) {
          continue;
        }

        const ctor = field_spec.ctor;
        const elements = field_spec.elements || 0;
        if (!ctor || elements <= 0 || !ctor.BYTES_PER_ELEMENT) {
          error(
            `Invalid field spec for individual GPU buffer: ${Name.string(frag_id)}.${field_name}.`
          );
          continue;
        }

        const byte_stride = elements * ctor.BYTES_PER_ELEMENT;
        if (byte_stride <= 0) {
          error(
            `Invalid byte_stride (${byte_stride}) for individual field ${Name.string(frag_id)}.${field_name}. Skipping.`
          );
          continue;
        }

        const actual_gpu_buffer_key = field_spec.buffer_name;
        fragment_class.field_key_map.set(field_name, actual_gpu_buffer_key);

        const multiplier = typeof field_spec.buffer_multiplier === "number" ? field_spec.buffer_multiplier : 1;

        const flat_buf = new FragmentGpuBuffer(
          actual_gpu_buffer_key,
          FragmentGpuBuffer.initial_max_rows,
          byte_stride,
          field_spec.cpu_readback,
          dispatch,
          field_spec.usage,
          fragment_class,
          field_name,
          null,
          false,
          multiplier
        );

        fragment_class.buffer_data.set(field_spec.buffer_name, {
          buffer: flat_buf,
          stride: byte_stride,
        });
      }
    }

    if (this.entity_flags_buffer === null) {
      // Initialize global GPU buffer for entity flags
      this.entity_flags_buffer = new FragmentGpuBuffer(
        entity_flags_buffer_name, // name
        FragmentGpuBuffer.initial_max_rows, // max rows
        4, // 4 bytes per row (Uint32 flags)
        true, // CPU-readback
        dispatch, // dispatch events?
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        null, // fragment_class_ref
        null, // config_key_within_fragment
        (chunk) => chunk.flags_meta // sync_target_accessor for entity_flags
      );
    }
  }

  /**
   * Flush all fragment SSBOs.
   * - Upload dirty chunks into stable chunk-sized regions keyed by row field.
   * - Only fall back to a full flush when buffers are recreated/resized.
   */
  static flush_gpu_buffers(allocator) {
    // if nothing dirty and no buffer requested a full-chunk repack, bail
    if (Chunk.dirty.size === 0 && !FragmentGpuBuffer.need_full_flush) return;

    const total_rows = Chunk.max_allocated_row();
    const resized = this._ensure_all_buffer_capacity(total_rows);
    const force_full_flush = FragmentGpuBuffer.need_full_flush || resized;

    const chunks_to_upload = force_full_flush
      ? Chunk.all_chunks.filter(Boolean)
      : Array.from(Chunk.dirty);

    this._flush_global_gpu_buffers(force_full_flush);

    for (let i = 0; i < chunks_to_upload.length; i++) {
      const chunk = chunks_to_upload[i];
      this._upload_chunk_index_map(allocator, chunk);
      this._upload_chunk_buffers(chunk, force_full_flush);
    }

    FragmentGpuBuffer.need_full_flush = false;
  }

  static _flush_global_gpu_buffers(force = false) {
    const dirty_global_buffers = new Set();

    if (!force) {
      for (const chunk of Chunk.dirty) {
        for (let i = 0; i < chunk.fragments.length; i++) {
          const fragment = chunk.fragments[i];
          if (!fragment?.gpu_buffers) {
            continue;
          }

          for (const [buffer_key, cfg] of Object.entries(fragment.gpu_buffers)) {
            if (typeof cfg.global_data !== "function") {
              continue;
            }

            if (chunk.dirty_buffers.has(buffer_key)) {
              dirty_global_buffers.add(`${fragment.id}:${buffer_key}`);
            }
          }
        }
      }
    }

    const processed = new Set();

    for (let i = 0; i < FragmentGpuBuffer.all_buffers.length; i++) {
      const gpu_buffer = FragmentGpuBuffer.all_buffers[i];
      const fragment = gpu_buffer.fragment_class_ref;
      const buffer_key = gpu_buffer.config_key_within_fragment;
      const cfg = fragment?.gpu_buffers?.[buffer_key];

      if (!cfg || typeof cfg.global_data !== "function") {
        continue;
      }

      const global_buffer_id = `${fragment.id}:${buffer_key}`;
      if (processed.has(global_buffer_id)) {
        continue;
      }

      if (!force && !dirty_global_buffers.has(global_buffer_id)) {
        continue;
      }

      const buf_data = fragment.buffer_data.get(cfg.buffer_name);
      if (!buf_data) {
        continue;
      }

      const packed = cfg.global_data.call(this, fragment);
      if (packed?.packed_data && packed.row_count > 0) {
        buf_data.buffer.update_chunk(0, packed.packed_data, packed.row_count);
      }

      processed.add(global_buffer_id);
    }
  }

  /**
   * Retrieves a GPU buffer for a given fragment class and field or buffer key.
   * @param {typeof import('../fragment.js').Fragment} FragmentClass - The fragment class.
   * @param {string} field_or_buffer_key - The field or buffer key.
   * @returns {FragmentGpuBuffer | null} The GPU buffer, or null if not found.
   */
  static get_buffer(FragmentClass, field_or_buffer_key) {
    const buffer_data = FragmentClass.buffer_data.get(field_or_buffer_key);
    if (!buffer_data) {
      error(
        `Sector.get_buffer: No GPU buffer key mappings found for ${FragmentClass.name}. Was init_gpu_buffers called?`
      );
      return null;
    }
    return buffer_data.buffer;
  }

  /**
   * Retrieves the buffer name for a given fragment class and field name.
   * @param {typeof import('../fragment.js').Fragment} FragmentClass - The fragment class.
   * @param {string} field_name - The name of the field.
   * @returns {string | null} The buffer name, or null if not found.
   */
  static get_buffer_name(FragmentClass, field_name) {
    return FragmentClass.field_key_map.get(field_name);
  }

  /**
   * Sync all GPU buffers down to the CPU.
   */
  static async sync_all_buffers() {
    for (let i = 0; i < this.all_buffers.length; i++) {
      const buffer = this.all_buffers[i];
      BufferSync.request_readback(buffer);
    }
  }

  /**
   * Prepares the ArrayBufferView for a specific field within a chunk.
   * @private
   * @param {Chunk} chunk - The source chunk.
   * @param {string} frag_id - The fragment ID (e.g., 'transform').
   * @param {string} field_name - The name of the field (e.g., 'position').
   * @returns {{ packed_data: ArrayBufferView, row_count: number } | null} - The field data and row count, or null if not found.
   */
  static _pack_chunk_field_data(chunk, frag_id, field_name) {
    const fragment_views = chunk.fragment_views[frag_id];
    if (!fragment_views) {
      return null;
    }
    const field_view = fragment_views[field_name];
    if (!field_view) {
      return null;
    }
    return { packed_data: field_view, row_count: DEFAULT_CHUNK_CAPACITY };
  }


  /**
   * Packs data for a specific combined buffer definition of a fragment type.
   * @private
   * @param {Chunk} chunk - The source chunk.
   * @param {Fragment} fragment - The fragment to pack.
   * @param {string} buffer_key - The key from gpu_buffers (e.g., 'lighting_params').
   * @returns {{ packed_data: ArrayBufferView, row_count: number } | null}
   */
  static _pack_combined_chunk_data(chunk, fragment, buffer_key) {
    const buffer_config = fragment.gpu_buffers?.[buffer_key];
    const source_fragment_views = chunk.fragment_views[fragment.id];

    if (!fragment || !buffer_config || !source_fragment_views) {
      error(
        `_pack_combined_chunk_data: Missing info for fragment '${fragment.id}', buffer_key '${buffer_key}'`
      );
      return null;
    }

    const field_names = buffer_config.fields;
    const row_count = DEFAULT_CHUNK_CAPACITY;
    const field_details = []; // Store { view, byte_size, offset_in_stride, ctor }

    // Pre-calculate field strides and gather other field details
    let current_field_offset_in_stride = 0;
    for (let i = 0; i < field_names.length; i++) {
      const field_name = field_names[i];
      const field_spec = fragment.fields[field_name];
      const source_field_view = source_fragment_views[field_name];
      if (
        !field_spec ||
        !source_field_view ||
        !field_spec.ctor ||
        !field_spec.ctor.BYTES_PER_ELEMENT ||
        !field_spec.elements
      ) {
        error(
          `_pack_combined_chunk_data: Invalid/missing spec or view for field '${Name.string(frag_id)}.${field_name}' in buffer '${buffer_key}'`
        );
        continue;
      }
      field_details.push({
        view: source_field_view,
        element_size: field_spec.elements,
        offset_in_stride: current_field_offset_in_stride,
        ctor: field_spec.ctor,
      });
      current_field_offset_in_stride += field_spec.elements;
    }

    if (buffer_config.stride === 0 || field_details.length === 0) {
      return { packed_data: new Uint8Array(0), row_count: row_count };
    }

    // Create a buffer of the appropriate type based on the first field's type
    // This assumes all fields in a combined buffer are of the same type
    const first_field = field_details[0];
    const bytes_per_element = first_field.ctor.BYTES_PER_ELEMENT;
    const num_buffer_elements = buffer_config.stride / bytes_per_element;
    const total_elements = row_count * num_buffer_elements;
    const packed_buffer = new first_field.ctor(total_elements);

    for (let local_index = 0; local_index < row_count; local_index++) {
      const dest_row_base_offset = local_index * num_buffer_elements;
      for (let j = 0; j < field_details.length; j++) {
        const field_detail = field_details[j];
        const src_start = local_index * field_detail.element_size;
        const src_end = src_start + field_detail.element_size;
        const dest_offset = dest_row_base_offset + field_detail.offset_in_stride;
        packed_buffer.set(field_detail.view.subarray(src_start, src_end), dest_offset);
      }
    }

    return { packed_data: packed_buffer, row_count: row_count };
  }

  static _ensure_all_buffer_capacity(total_rows) {
    let resized = false;
    for (let i = 0; i < FragmentGpuBuffer.all_buffers.length; i++) {
      const gpu_buffer = FragmentGpuBuffer.all_buffers[i];
      const cfg =
        gpu_buffer.fragment_class_ref?.gpu_buffers?.[gpu_buffer.config_key_within_fragment] ?? null;
      if (cfg && typeof cfg.global_data === "function") {
        continue;
      }
      if (gpu_buffer.max_rows < total_rows) {
        gpu_buffer._resize_buffer(npot(total_rows));
        resized = true;
      }
    }
    return resized;
  }

  static _get_chunk_base(chunk) {
    return chunk.chunk_index * DEFAULT_CHUNK_CAPACITY;
  }

  static _build_chunk_index_map_data(allocator, chunk) {
    const row_map = allocator.row_to_entity_id;
    const chunk_base = this._get_chunk_base(chunk);
    const index_map = new Uint32Array(DEFAULT_CHUNK_CAPACITY);
    index_map.fill(FragmentGpuBuffer.invalid_row);

    for (let local_index = 0; local_index < DEFAULT_CHUNK_CAPACITY; local_index++) {
      const row = chunk_base + local_index;
      if (row_map.get(row) !== undefined && row_map.get(row) !== null) {
        index_map[local_index] = row;
      }
    }

    return index_map;
  }

  static _upload_chunk_index_map(allocator, chunk) {
    const chunk_base = this._get_chunk_base(chunk);
    const index_map = this._build_chunk_index_map_data(allocator, chunk);
    this.entity_index_map_buffer.update_chunk(
      chunk_base,
      index_map,
      DEFAULT_CHUNK_CAPACITY,
      chunk
    );
  }

  static _upload_chunk_buffers(chunk, force_full_flush = false) {
    const chunk_base = this._get_chunk_base(chunk);

    this.entity_flags_buffer.update_chunk(
      chunk_base,
      chunk.flags_meta,
      DEFAULT_CHUNK_CAPACITY,
      chunk
    );

    for (let i = 0; i < chunk.fragments.length; i++) {
      const fragment = chunk.fragments[i];
      if (!fragment) continue;

      if (fragment.gpu_buffers) {
        for (const [buffer_key, cfg] of Object.entries(fragment.gpu_buffers)) {
          if (typeof cfg.global_data === "function") {
            continue;
          }

          if (!force_full_flush && !chunk.dirty_buffers.has(buffer_key)) {
            continue;
          }

          const buf_data = fragment.buffer_data.get(cfg.buffer_name);
          const packed =
            typeof cfg.gpu_data === "function"
              ? cfg.gpu_data.call(this, chunk, fragment)
              : this._pack_combined_chunk_data(chunk, fragment, buffer_key);
          if (packed?.packed_data.byteLength) {
            buf_data.buffer.update_chunk(chunk_base, packed.packed_data, packed.row_count, chunk);
          }
        }
      }

      for (const [field_name, spec] of Object.entries(fragment.fields)) {
        if (!spec.gpu_buffer) continue;
        if (!force_full_flush && !chunk.dirty_buffers.has(field_name)) {
          continue;
        }

        const buf_data = fragment.buffer_data.get(spec.buffer_name);
        const packed = this._pack_chunk_field_data(chunk, fragment.id, field_name);
        if (packed?.packed_data.byteLength) {
          buf_data.buffer.update_chunk(chunk_base, packed.packed_data, packed.row_count, chunk);
        }
      }
    }
  }
}
