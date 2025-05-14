import { Fragment } from "../fragment.js";
import { SolarFragmentView } from "../solar/view.js";
import { RingBufferAllocator } from "../../../memory/allocator.js";
import { Name } from "../../../utility/names.js";
import { FontCache } from "../../../ui/text/font_cache.js";
import { EntityManager } from "../entity.js";
import { EntityFlags } from "../../minimal.js";

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
        if (!value) return;

        const font_typed_array =
          this.chunk.fragment_views[this.fragment_id]?.font;
        const font = FontCache.get_font_object(font_typed_array[this.slot]);

        // 1) turn string → array of code-point indices
        const code_point_indexes = Array.from(value).map((char) => {
          return font.code_point_index_map.get(char.codePointAt(0));
        });

        // 2) re-allocate the entity to match the new length
        EntityManager.set_entity_instance_count(
          this.entity,
          code_point_indexes.length,
        );

        // 3) pull back the brand-new layout (one or more segments)
        let write_offset = 0;
        for (let i = 0; i < this.entity.segments.length; i++) {
          const { chunk, slot, count } = this.entity.segments[i];

          // grab the *fresh* views for this segment
          const frag_views = chunk.fragment_views[this.fragment_id];
          const text_array = frag_views.text; // Uint32Array view for .text
          const slice = code_point_indexes.slice(
            write_offset,
            write_offset + count,
          );

          // write into the correct slot
          text_array.set(slice, slot);

          for (let j = 0; j < count; j++) {
            chunk.flags_meta[slot + j] |= EntityFlags.DIRTY;
          }

          // mark it so it ends up in the next GPU flush
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
        const row_count = chunk.capacity;
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
