import { EntityLinearDataContainer } from "../entity_utils.js";
import { Fragment } from "../fragment.js";
import { Renderer } from "../../../renderer/renderer.js";
import { Buffer } from "../../../renderer/buffer.js";
import { global_dispatcher } from "../../../core/dispatcher.js";
import { RingBufferAllocator } from "../../../memory/allocator.js";
import { EntityID } from "../entity.js";
import { EntityManager } from "../entity.js";
import { MAX_BUFFERED_FRAMES } from "../../../core/minimal.js";
import { FontCache } from "../../../ui/text/font_cache.js";

const text_buffer_name = "text_buffer";
const text_cpu_buffer_name = "text_cpu_buffer";
const text_event = "text";
const text_update_event = "text_update";

const string_data_buffer_name = "string_data_buffer";
const string_data_cpu_buffer_name = "string_data_cpu_buffer";
const string_data_event = "string_data";
const string_data_update_event = "string_data_update";

class ColorDataView {
  constructor() {
    this.current_entity = -1n;
    this.absolute_entity = -1n;
  }

  get r() {
    return TextFragment.data.color.r[this.current_entity];
  }

  set r(value) {
    TextFragment.data.color.r[this.current_entity] = value;
    if (TextFragment.data.dirty) {
      TextFragment.data.dirty[this.current_entity] = 1;
    }
    TextFragment.data.gpu_data_dirty = true;
  }

  get g() {
    return TextFragment.data.color.g[this.current_entity];
  }

  set g(value) {
    TextFragment.data.color.g[this.current_entity] = value;
    if (TextFragment.data.dirty) {
      TextFragment.data.dirty[this.current_entity] = 1;
    }
    TextFragment.data.gpu_data_dirty = true;
  }

  get b() {
    return TextFragment.data.color.b[this.current_entity];
  }

  set b(value) {
    TextFragment.data.color.b[this.current_entity] = value;
    if (TextFragment.data.dirty) {
      TextFragment.data.dirty[this.current_entity] = 1;
    }
    TextFragment.data.gpu_data_dirty = true;
  }

  get a() {
    return TextFragment.data.color.a[this.current_entity];
  }

  set a(value) {
    TextFragment.data.color.a[this.current_entity] = value;
    if (TextFragment.data.dirty) {
      TextFragment.data.dirty[this.current_entity] = 1;
    }
    TextFragment.data.gpu_data_dirty = true;
  }

  view_entity(entity, instance = 0) {
    this.current_entity = entity;
    this.absolute_entity = EntityID.get_absolute_index(entity) + instance;
    return this;
  }
}

class TextDataView {
  current_entity = -1n;
  absolute_entity = -1n;

  constructor() {
    this.color = new ColorDataView(this);
  }

  get text() {
    const font = FontCache.get_font_object(
      TextFragment.data.font[this.current_entity],
    );
    return TextFragment.data.text.get_data_for_entity(this.current_entity);
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
      EntityManager.set_entity_instance_count(
        this.current_entity,
        code_point_indexes.length,
      );
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

  get emissive() {
    return TextFragment.data.emissive[this.current_entity];
  }

  set emissive(value) {
    TextFragment.data.emissive[this.current_entity] =
      TextFragment.data.emissive instanceof BigInt64Array
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

  view_entity(entity, instance = 0) {
    this.current_entity = entity;
    this.absolute_entity = EntityID.get_absolute_index(entity) + instance;

    this.color.view_entity(entity, instance);

    return this;
  }
}

const unmapped_state = "unmapped";

export class TextFragment extends Fragment {
  static data_view_allocator = new RingBufferAllocator(256, TextDataView);
  static size = 0;
  static data = null;

  static initialize() {
    this.data = {
      text: new EntityLinearDataContainer(Uint32Array),
      offsets: new EntityLinearDataContainer(Float32Array),
      font: new Int32Array(1),
      font_size: new Uint32Array(1),
      color: {
        r: new Float32Array(1),
        g: new Float32Array(1),
        b: new Float32Array(1),
        a: new Float32Array(1),
      },
      emissive: new Float32Array(1),
      dirty: new Uint8Array(1),
      text_buffer: null,
      string_data_buffer: null,
      valid_prev: new Int32Array(1),
      valid_next: new Int32Array(1),
      first_valid_index: -1,
      last_valid_index: -1,
      gpu_data_dirty: true,
    };

    this.rebuild_buffers();
  }

  static resize(new_size) {
    if (new_size <= this.size) return;

    new_size *= 2;
    this.size = new_size;

    if (!this.data) this.initialize();

    Fragment.resize_array(this.data, "font", new_size, Int32Array, 1);
    Fragment.resize_array(this.data, "font_size", new_size, Uint32Array, 1);
    Object.keys(this.data.color).forEach((axis) => {
      Fragment.resize_array(this.data.color, axis, new_size, Float32Array);
    });
    Fragment.resize_array(this.data, "emissive", new_size, Float32Array, 1);
    Fragment.resize_array(this.data, "dirty", new_size, Uint8Array, 1);

    Fragment.resize_array(this.data, "valid_prev", new_size, Int32Array, 1);
    Fragment.resize_array(this.data, "valid_next", new_size, Int32Array, 1);

    this.data.gpu_data_dirty = true;
  }

  static add_entity(entity) {
    const absolute_entity = EntityID.get_absolute_index(entity);
    if (absolute_entity >= this.size) {
      this.resize(absolute_entity * 2);
    }

    const idx = Number(absolute_entity);
    const tail = this.data.last_valid_index;
    if (tail >= 0) {
      this.data.valid_next[tail] = idx;
      this.data.valid_prev[idx] = tail;
    } else {
      this.data.first_valid_index = idx;
      this.data.valid_prev[idx] = -1;
    }
    this.data.valid_next[idx] = -1;
    this.data.last_valid_index = idx;

    return this.get_entity_data(entity);
  }

  static remove_entity(entity) {
    const instance_count = EntityID.get_instance_count(entity);
    const entity_offset = EntityID.get_absolute_index(entity);

    this.data.text.remove(entity);
    this.data.offsets.remove(entity);
    this.data.font[entity] = this.data.font instanceof BigInt64Array ? 0n : 0;
    this.data.font_size[entity] =
      this.data.font_size instanceof BigInt64Array ? 0n : 0;
    this.data.color.r[entity] =
      this.data.color instanceof BigInt64Array ? 0n : 0;
    this.data.color.g[entity] =
      this.data.color instanceof BigInt64Array ? 0n : 0;
    this.data.color.b[entity] =
      this.data.color instanceof BigInt64Array ? 0n : 0;
    this.data.color.a[entity] =
      this.data.color instanceof BigInt64Array ? 0n : 0;

    this.data.emissive[entity] =
      this.data.emissive instanceof BigInt64Array ? 0n : 0;

    for (let i = 0; i < instance_count; ++i) {
      const entity_index = entity_offset + i;
    }

    this.data.gpu_data_dirty = true;

    // unlink from the live‐list in O(1)
    const idx = Number(EntityID.get_absolute_index(entity));
    const p = this.data.valid_prev[idx];
    const n = this.data.valid_next[idx];

    if (p >= 0) {
      this.data.valid_next[p] = n;
    } else {
      this.data.first_valid_index = n;
    }

    if (n >= 0) {
      this.data.valid_prev[n] = p;
    } else {
      this.data.last_valid_index = p;
    }

    // clear pointers for safety
    this.data.valid_prev[idx] = -1;
    this.data.valid_next[idx] = -1;
  }

  static get_entity_data(entity, instance = 0) {
    const data_view = this.data_view_allocator.allocate();
    data_view.view_entity(entity, instance);
    return data_view;
  }

  static duplicate_entity_data(entity, instance = 0) {
    const data = {};
    data.text = String.fromCodePoint(
      ...this.data.text.get_data_for_entity(entity),
    );
    data.font = this.data.font[entity];
    data.font_size = this.data.font_size[entity];
    data.emissive = this.data.emissive[entity];
    data.color = {
      r: this.data.color.r[entity],
      g: this.data.color.g[entity],
      b: this.data.color.b[entity],
      a: this.data.color.a[entity],
    };
    return data;
  }

  static to_gpu_data() {
    if (!this.data) this.initialize();

    if (!this.data.gpu_data_dirty) {
      return {
        text_buffer: this.data.text_buffer,
        string_data_buffer: this.data.string_data_buffer,
      };
    }

    this.rebuild_buffers();

    return {
      text_buffer: this.data.text_buffer,
      string_data_buffer: this.data.string_data_buffer,
    };
  }

  static rebuild_buffers() {
    if (!this.data.gpu_data_dirty) return;

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
      const gpu_data = new Float32Array(Math.max(this.size * 12, 12));
      for (let i = 0; i < this.size; i++) {
        const metadata = this.data.text.get_metadata(i);
        const font = FontCache.get_font_object(this.data.font[i]);
        const gpu_data_offset = i * 12;
        gpu_data[gpu_data_offset + 0] = metadata?.start ?? 0;
        gpu_data[gpu_data_offset + 1] = metadata?.count ?? 0;
        gpu_data[gpu_data_offset + 2] = font?.texture_width ?? 0;
        gpu_data[gpu_data_offset + 3] = font?.texture_height ?? 0;
        gpu_data[gpu_data_offset + 4] = this.data.color.r[i];
        gpu_data[gpu_data_offset + 5] = this.data.color.g[i];
        gpu_data[gpu_data_offset + 6] = this.data.color.b[i];
        gpu_data[gpu_data_offset + 7] = this.data.color.a[i];
        gpu_data[gpu_data_offset + 8] = this.data.emissive[i];
        gpu_data[gpu_data_offset + 9] = 0; // padding
        gpu_data[gpu_data_offset + 10] = 0; // padding
        gpu_data[gpu_data_offset + 11] = 0; // padding
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

  static async sync_buffers() {
    const buffered_frame = Renderer.get().get_buffered_frame_number();
  }

  static copy_entity_instance(to_index, from_index) {
    this.data.gpu_data_dirty = true;
  }
}
