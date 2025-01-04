import { EntityLinearDataContainer } from "../entity_utils.js";
import { Fragment } from "../fragment.js";
import { Renderer } from "../../../renderer/renderer.js";
import { Buffer } from "../../../renderer/buffer.js";
import { global_dispatcher } from "../../../core/dispatcher.js";
import { RingBufferAllocator } from "../../../memory/allocator.js";
import { EntityID } from "../entity.js";
import { FontCache } from "../../../ui/text/font_cache.js";

const text_buffer_name = "text_buffer";
const text_cpu_buffer_name = "text_cpu_buffer";
const text_event = "text";
const text_update_event = "text_update";

const offsets_buffer_name = "offsets_buffer";
const offsets_cpu_buffer_name = "offsets_cpu_buffer";
const offsets_event = "offsets";
const offsets_update_event = "offsets_update";

const string_data_buffer_name = "string_data_buffer";
const string_data_cpu_buffer_name = "string_data_cpu_buffer";
const string_data_event = "string_data";
const string_data_update_event = "string_data_update";

class TextDataView {
  current_entity = -1n;

  constructor() {}

  get text() {
    const font = FontCache.get_font_object(
      TextFragment.data.font[this.current_entity],
    );
    const code_point_indexes = TextFragment.data.text.get_data_for_entity(
      this.current_entity,
    );
    return code_point_indexes
      .map((code_point_index) =>
        String.fromCodePoint(font.code_point[code_point_index]),
      )
      .join("");
  }

  set text(value) {
    if (value) {
      const font = FontCache.get_font_object(
        TextFragment.data.font[this.current_entity],
      );
      const code_point_indexes = Array.from(value).map((char) =>
        font.code_point_index_map.get(char.codePointAt(0)),
      );
      TextFragment.data.text.update(this.current_entity, code_point_indexes);
    }
    if (TextFragment.data.dirty) {
      TextFragment.data.dirty[this.current_entity] = 1;
    }
    TextFragment.data.gpu_data_dirty = true;
  }

  get offsets() {
    return TextFragment.data.offsets.get_data_for_entity(this.current_entity);
  }

  set offsets(value) {
    TextFragment.data.offsets.update(this.current_entity, value ?? []);
    if (TextFragment.data.dirty) {
      TextFragment.data.dirty[this.current_entity] = 1;
    }
    TextFragment.data.gpu_data_dirty = true;
  }

  get font() {
    return TextFragment.data.font[this.current_entity];
  }

  set font(value) {
    TextFragment.data.font[this.current_entity] =
      TextFragment.data.font instanceof BigInt64Array ? BigInt(value) : value;
    if (TextFragment.data.dirty) {
      TextFragment.data.dirty[this.current_entity] = 1;
    }
    TextFragment.data.gpu_data_dirty = true;
  }

  get font_size() {
    return TextFragment.data.font_size[this.current_entity];
  }

  set font_size(value) {
    TextFragment.data.font_size[this.current_entity] =
      TextFragment.data.font_size instanceof BigInt64Array
        ? BigInt(value)
        : value;
    if (TextFragment.data.dirty) {
      TextFragment.data.dirty[this.current_entity] = 1;
    }
    TextFragment.data.gpu_data_dirty = true;
  }

  get dirty() {
    return TextFragment.data.dirty[this.current_entity];
  }

  set dirty(value) {
    TextFragment.data.dirty[this.current_entity] =
      TextFragment.data.dirty instanceof BigInt64Array ? BigInt(value) : value;
    if (TextFragment.data.dirty) {
      TextFragment.data.dirty[this.current_entity] = 1;
    }
    TextFragment.data.gpu_data_dirty = true;
  }

  view_entity(entity) {
    this.current_entity = entity;

    return this;
  }
}

export class TextFragment extends Fragment {
  static data_view_allocator = new RingBufferAllocator(256, TextDataView);

  static initialize() {
    this.data = {
      text: new EntityLinearDataContainer(Uint32Array),
      offsets: new EntityLinearDataContainer(Float32Array),
      font: new Int32Array(1),
      font_size: new Uint32Array(1),
      dirty: new Uint8Array(1),
      text_buffer: null,
      offsets_buffer: null,
      string_data_buffer: null,
      gpu_data_dirty: true,
    };

    this.rebuild_buffers();
  }

  static resize(new_size) {
    if (!this.data) this.initialize();
    super.resize(new_size);

    Fragment.resize_array(this.data, "font", new_size, Int32Array, 1);
    Fragment.resize_array(this.data, "font_size", new_size, Uint32Array, 1);
    Fragment.resize_array(this.data, "dirty", new_size, Uint8Array, 1);

    this.rebuild_buffers();
  }

  static add_entity(entity) {
    super.add_entity(entity);
    return this.get_entity_data(entity);
  }

  static remove_entity(entity) {
    super.remove_entity(entity);

    const instance_count = EntityID.get_instance_count(entity);
    const entity_offset = EntityID.get_absolute_index(entity);

    for (let i = 0; i < instance_count; ++i) {
      const entity_index = entity_offset + i;
      this.data.font[entity_index] = 0;
      this.data.font_size[entity_index] = 0;

      this.data.text.remove(entity_index);
      this.data.offsets.remove(entity_index);
    }
  }

  static get_entity_data(entity, instance = 0) {
    const entity_index = EntityID.get_absolute_index(entity) + instance;
    const data_view = this.data_view_allocator.allocate();
    data_view.fragment = this;
    data_view.view_entity(entity_index);
    return data_view;
  }

  static duplicate_entity_data(entity, instance = 0) {
    const entity_offset = EntityID.get_absolute_index(entity);
    const data = {};
    data.text = String.fromCodePoint(
      ...this.data.text.get_data_for_entity(entity_offset),
    );
    data.font = this.data.font[entity_offset];
    return data;
  }

  static to_gpu_data() {
    if (!this.data) this.initialize();

    if (!this.data.gpu_data_dirty) {
      return {
        text_buffer: this.data.text_buffer,
        offsets_buffer: this.data.offsets_buffer,
        string_data_buffer: this.data.string_data_buffer,
      };
    }

    this.rebuild_buffers();

    return {
      text_buffer: this.data.text_buffer,
      offsets_buffer: this.data.offsets_buffer,
      string_data_buffer: this.data.string_data_buffer,
    };
  }

  static rebuild_buffers() {
    {
      const gpu_data = this.data.text.get_data();

      if (
        !this.data.text_buffer ||
        this.data.text_buffer.config.size < gpu_data.byteLength
      ) {
        this.data.text_buffer = Buffer.create({
          name: text_buffer_name,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          raw_data: gpu_data,
          force: true,
        });

        Renderer.get().mark_bind_groups_dirty(true);
        global_dispatcher.dispatch(text_event, this.data.text_buffer);
      } else {
        this.data.text_buffer.write(gpu_data);
      }

      global_dispatcher.dispatch(text_update_event);
    }

    {
      const gpu_data = this.data.offsets.get_data();

      if (
        !this.data.offsets_buffer ||
        this.data.offsets_buffer.config.size < gpu_data.byteLength
      ) {
        this.data.offsets_buffer = Buffer.create({
          name: offsets_buffer_name,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          raw_data: gpu_data,
          force: true,
        });

        Renderer.get().mark_bind_groups_dirty(true);
        global_dispatcher.dispatch(offsets_event, this.data.offsets_buffer);
      } else {
        this.data.offsets_buffer.write(gpu_data);
      }

      global_dispatcher.dispatch(offsets_update_event);
    }

    {
      const gpu_data = new Uint32Array(Math.max(this.size * 6, 6));
      for (let i = 0; i < this.size; i++) {
        const metadata = this.data.text.get_metadata(i);
        const font = FontCache.get_font_object(this.data.font[i]);
        const gpu_data_offset = i * 6;
        gpu_data[gpu_data_offset + 0] = metadata?.start ?? 0;
        gpu_data[gpu_data_offset + 1] = metadata?.count ?? 0;
        gpu_data[gpu_data_offset + 2] = font?.texture_width ?? 0;
        gpu_data[gpu_data_offset + 3] = font?.texture_height ?? 0;
        gpu_data[gpu_data_offset + 4] = this.data.font_size[i];
        gpu_data[gpu_data_offset + 5] = 0; // padding
      }

      if (
        !this.data.string_data_buffer ||
        this.data.string_data_buffer.config.size < gpu_data.byteLength
      ) {
        this.data.string_data_buffer = Buffer.create({
          name: string_data_buffer_name,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          raw_data: gpu_data,
          force: true,
        });

        Renderer.get().mark_bind_groups_dirty(true);
        global_dispatcher.dispatch(
          string_data_event,
          this.data.string_data_buffer,
        );
      } else {
        this.data.string_data_buffer.write(gpu_data);
      }

      global_dispatcher.dispatch(string_data_update_event);
    }

    this.data.gpu_data_dirty = false;
  }

  static async sync_buffers() {}
}
