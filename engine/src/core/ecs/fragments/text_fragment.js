import { Fragment } from "../fragment.js";
import { SolarFragmentView } from "../solar/view.js";
import { RingBufferAllocator } from "../../../memory/allocator.js";
import { Name } from "../../../utility/names.js";
import { FontCache } from "../../../ui/text/font_cache.js";
import { EntityManager } from "../entity.js";
import { EntityFlags } from "../../minimal.js";
import { DEFAULT_CHUNK_CAPACITY } from "../solar/types.js";

/**
 * The Text fragment class.
 * Use `EntityManager.get_fragment(entity, Text)` to get a fragment instance for an entity.
 */
export class TextFragment extends Fragment {
  static id = Name.from("text");
  static field_key_map = new Map();
  static fields = {
    text: {
      ctor: Uint32Array,
      elements: 1,
      default: 0,
      gpu_buffer: true,
      buffer_name: "text",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      setter(value, typed_array, element_offset) {
        if (value === null || value === undefined) {
          value = ""; // Treat null/undefined as a request to clear, equivalent to empty string.
        }

        const text_value_as_string = String(value);
        const target_instance_count = text_value_as_string.length;

        if (
          !this.entity ||
          !this.entity.segments ||
          this.entity.segments.length === 0
        ) {
          if (this.entity) {
            // Still try to set instance count if handle exists
            EntityManager.set_entity_instance_count(
              this.entity,
              target_instance_count,
            );
          }
          return;
        }

        // Get font ID from the entity's *current* primary segment's chunk and slot
        const current_primary_segment = this.entity.segments[0];
        const chunk_for_font_lookup = current_primary_segment.chunk;
        const slot_for_font_lookup = current_primary_segment.slot; // This is the base slot for the entity in this chunk

        const font_data_array =
          chunk_for_font_lookup.fragment_views[this.fragment_id]?.font;

        if (!font_data_array) {
          EntityManager.set_entity_instance_count(
            this.entity,
            target_instance_count,
          );
          return;
        }

        // The 'font' field is a single Int32, so use the entity's base slot in its current primary chunk
        const font_id = font_data_array[slot_for_font_lookup];
        const font = FontCache.get_font_object(font_id);

        if (!font && target_instance_count > 0) {
          EntityManager.set_entity_instance_count(
            this.entity,
            target_instance_count,
          );
          return;
        }

        let code_point_indexes = [];
        if (target_instance_count > 0 && font) {
          code_point_indexes = Array.from(text_value_as_string).map((char) => {
            const code_point = char.codePointAt(0);
            const index = font.code_point_index_map.get(code_point);
            if (index === undefined) {
              const fallback_index = font.code_point_index_map.get(
                " ".codePointAt(0),
              );
              return fallback_index !== undefined ? fallback_index : 0;
            }
            return index;
          });
        }

        // Determine how many instances we had before and how many we want now:
        EntityManager.set_entity_instance_count(
          this.entity,
          target_instance_count,
        );

        // Write data to the (now potentially new) segments stored in entity_handle.segments.
        let write_offset = 0;
        const segments_to_write = this.entity.segments || [];
        for (let i = 0; i < segments_to_write.length; i++) {
          const { chunk, slot, count } = segments_to_write[i];

          const frag_views_in_current_chunk =
            chunk.fragment_views[this.fragment_id];
          const text_array_in_current_chunk = frag_views_in_current_chunk?.text;

          const slice_to_write = code_point_indexes.slice(
            write_offset,
            write_offset + count,
          );

          text_array_in_current_chunk.set(slice_to_write, slot);

          for (let j = 0; j < count; j++) {
            // Ensure we don't write past flags_meta if count is unexpectedly large
            chunk.flags_meta[slot + j] |= EntityFlags.DIRTY;
          }
          chunk.mark_dirty();

          write_offset += count;
        }
      },
      cpu_readback: false,
    },
    font: {
      ctor: Int32Array,
      elements: 1,
      default: 0,
      gpu_buffer: false,
      buffer_name: "font",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
    },
    font_size: {
      ctor: Uint32Array,
      elements: 1,
      default: 0,
      gpu_buffer: false,
      buffer_name: "font_size",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
    },
    text_color: {
      ctor: Float32Array,
      elements: 4,
      default: 0,
      gpu_buffer: true,
      buffer_name: "text_color",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
    },
    text_emissive: {
      ctor: Float32Array,
      elements: 1,
      default: 0,
      gpu_buffer: true,
      buffer_name: "text_emissive",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
    },
  };
  static buffer_data = new Map(); // key → { buffer: FragmentGpuBuffer, stride: number }

  static gpu_buffers = {
    string_data: {
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      stride: 32,
      buffer_name: "string_data",
      cpu_readback: false,

      gpu_data(chunk, fragment) {
        const row_count = DEFAULT_CHUNK_CAPACITY;
        const fragment_views = chunk.fragment_views[fragment.id];
        const packed_data = new Float32Array(Math.max(row_count * 8, 8));
        for (let row = 0; row < row_count; row++) {
          const gpu_data_offset = row * 8;
          const font = FontCache.get_font_object(fragment_views.font[row]);
          packed_data[gpu_data_offset + 0] = fragment_views.text_color[row * 4];
          packed_data[gpu_data_offset + 1] =
            fragment_views.text_color[row * 4 + 1];
          packed_data[gpu_data_offset + 2] =
            fragment_views.text_color[row * 4 + 2];
          packed_data[gpu_data_offset + 3] =
            fragment_views.text_color[row * 4 + 3];
          packed_data[gpu_data_offset + 4] = font?.texture_width ?? 0;
          packed_data[gpu_data_offset + 5] = font?.texture_height ?? 0;
          packed_data[gpu_data_offset + 6] = fragment_views.text_emissive[row];
          packed_data[gpu_data_offset + 7] = 0; // padding
        }
        return { packed_data, row_count };
      },
    },
  };

  static get view_allocator() {
    if (!this._view_allocator) {
      this._view_allocator = new RingBufferAllocator(
        256,
        new SolarFragmentView(this),
      );
    }
    return this._view_allocator;
  }

  static is_valid() {
    return this.id && this.fields && this.view_allocator;
  }

  static get_buffer_name(field_name) {
    return this.field_key_map.get(field_name);
  }
}
