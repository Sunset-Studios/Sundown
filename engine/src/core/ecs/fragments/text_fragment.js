import { EntityLinearDataContainer } from "../entity_utils.js";
import { Fragment } from "../fragment.js";
import { Renderer } from "../../../renderer/renderer.js";
import { Buffer } from "../../../renderer/buffer.js";
import { global_dispatcher } from "../../../core/dispatcher.js";
import { RingBufferAllocator } from "../../../memory/allocator.js";

const text_buffer_name = "text_buffer";
const text_cpu_buffer_name = "text_cpu_buffer";
const text_event = "text";
const text_update_event = "text_update";

const dirty_buffer_name = "dirty_buffer";
const dirty_cpu_buffer_name = "dirty_cpu_buffer";
const dirty_event = "dirty";
const dirty_update_event = "dirty_update";

class TextDataView {
  current_entity = -1;

  constructor() {}

  get text() {
    return String.fromCodePoint(
      ...TextFragment.data.text.get_data_for_entity(this.current_entity),
    );
  }

  set text(value) {
    if (value) {
      const code_points = Array.from(value).map((char) => char.codePointAt(0));
      TextFragment.data.text.update(this.current_entity, code_points);
    }
    if (TextFragment.data.dirty) {
      TextFragment.data.dirty[this.current_entity] = 1;
    }
    TextFragment.data.gpu_data_dirty = true;
  }

  get font() {
    return TextFragment.data.font[this.current_entity];
  }

  set font(value) {
    TextFragment.data.font[this.current_entity] = value;
    if (TextFragment.data.dirty) {
      TextFragment.data.dirty[this.current_entity] = 1;
    }
    TextFragment.data.gpu_data_dirty = true;
  }

  get dirty() {
    return TextFragment.data.dirty[this.current_entity];
  }

  set dirty(value) {
    TextFragment.data.dirty[this.current_entity] = value;
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
      font: new Int32Array(1),
      dirty: new Uint8Array(1),
      text_buffer: null,
      dirty_buffer: null,
      gpu_data_dirty: true,
    };

    this.rebuild_buffers(Renderer.get().graphics_context);
  }

  static resize(new_size) {
    if (!this.data) this.initialize();
    super.resize(new_size);

    Fragment.resize_array(this.data, "font", new_size, Int32Array, 1);
    Fragment.resize_array(this.data, "dirty", new_size, Uint8Array, 1);

    this.rebuild_buffers(Renderer.get().graphics_context);
  }

  static add_entity(entity) {
    super.add_entity(entity);
    return this.get_entity_data(entity);
  }

  static remove_entity(entity) {
    super.remove_entity(entity);
    this.data.font[entity] = 0;

    this.data.text.remove(entity);
  }

  static get_entity_data(entity) {
    const data_view = this.data_view_allocator.allocate();
    data_view.fragment = this;
    data_view.view_entity(entity);
    return data_view;
  }

  static duplicate_entity_data(entity) {
    const data = {};
    data.text = String.fromCodePoint(
      ...this.data.text.get_data_for_entity(entity),
    );
    data.font = this.data.font[entity];
    return data;
  }

  static to_gpu_data(context) {
    if (!this.data) this.initialize();

    if (!this.data.gpu_data_dirty) {
      return {
        text_buffer: this.data.text_buffer,
        dirty_buffer: this.data.dirty_buffer,
      };
    }

    this.rebuild_buffers(context);

    return {
      text_buffer: this.data.text_buffer,
      dirty_buffer: this.data.dirty_buffer,
    };
  }

  static rebuild_buffers(context) {
    {
      const gpu_data = this.data.text.get_data();

      if (
        !this.data.text_buffer ||
        this.data.text_buffer.config.size < gpu_data.byteLength
      ) {
        this.data.text_buffer = Buffer.create(context, {
          name: text_buffer_name,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          raw_data: gpu_data,
          force: true,
        });

        Renderer.get().mark_bind_groups_dirty(true);
        global_dispatcher.dispatch(text_event, this.data.text_buffer);
      } else {
        this.data.text_buffer.write(context, gpu_data);
      }

      global_dispatcher.dispatch(text_update_event);
    }

    {
      const gpu_data = this.data.dirty
        ? this.data.dirty
        : new Uint32Array(this.size * 1);
      if (
        !this.data.dirty_buffer ||
        this.data.dirty_buffer.config.size < gpu_data.byteLength
      ) {
        this.data.dirty_buffer = Buffer.create(context, {
          name: dirty_buffer_name,
          usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.COPY_SRC,
          raw_data: gpu_data,
          force: true,
        });

        Renderer.get().mark_bind_groups_dirty(true);
        global_dispatcher.dispatch(dirty_event, this.data.dirty_buffer);
      } else {
        this.data.dirty_buffer.write(context, gpu_data);
      }

      global_dispatcher.dispatch(dirty_update_event);
    }

    this.data.gpu_data_dirty = false;
  }

  static async sync_buffers(context) {}
}
