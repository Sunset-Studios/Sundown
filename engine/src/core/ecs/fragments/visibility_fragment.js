import { EntityLinearDataContainer } from "../entity_utils.js";
import { Fragment } from "../fragment.js";
import { Renderer } from "../../../renderer/renderer.js";
import { Buffer } from "../../../renderer/buffer.js";
import { global_dispatcher } from "../../../core/dispatcher.js";

const visible_buffer_name = "visible_buffer";
const visible_cpu_buffer_name = "visible_cpu_buffer";
const visible_event = "visible";
const visible_update_event = "visible_update";

export class VisibilityFragment extends Fragment {
  static initialize() {
    this.data = {
      visible: new Uint8Array(4),
      visible_buffer: null,
      gpu_data_dirty: true,
    };

    this.rebuild_buffers(Renderer.get().graphics_context);
  }

  static resize(new_size) {
    if (!this.data) this.initialize();
    super.resize(new_size);

    Fragment.resize_array(this.data, "visible", new_size, Uint8Array, 4);

    this.rebuild_buffers(Renderer.get().graphics_context);
  }

  static add_entity(entity, data) {
    super.add_entity(entity, data);
  }

  static remove_entity(entity) {
    super.remove_entity(entity);
    this.update_entity_data(entity, {
      visible: Array(4).fill(0),
    });
  }

  static get_entity_data(entity) {
    return super.get_entity_data(entity);
  }

  static duplicate_entity_data(entity) {
    const data = {};
    data.visible = Array(4).fill(0);
    for (let i = 0; i < 4; i++) {
      data.visible[i] = this.data.visible[entity * 4 + i];
    }
    return data;
  }

  static update_entity_data(entity, data) {
    if (!this.data) {
      this.initialize();
    }

    super.update_entity_data(entity, data);

    if (this.data.dirty) {
      this.data.dirty[entity] = 1;
    }
    this.data.gpu_data_dirty = true;
  }

  static to_gpu_data(context) {
    if (!this.data) this.initialize();

    if (!this.data.gpu_data_dirty) {
      return {
        visible_buffer: this.data.visible_buffer,
      };
    }

    this.rebuild_buffers(context);

    return {
      visible_buffer: this.data.visible_buffer,
    };
  }

  static rebuild_buffers(context) {
    {
      const gpu_data = this.data.visible
        ? this.data.visible
        : new Uint8Array(this.size * 4);
      if (
        !this.data.visible_buffer ||
        this.data.visible_buffer.config.size < gpu_data.byteLength
      ) {
        this.data.visible_buffer = Buffer.create(context, {
          name: visible_buffer_name,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          raw_data: gpu_data,
          force: true,
        });

        Renderer.get().mark_bind_groups_dirty(true);
        global_dispatcher.dispatch(visible_event, this.data.visible_buffer);
      } else {
        this.data.visible_buffer.write(context, gpu_data);
      }

      global_dispatcher.dispatch(visible_update_event);
    }

    this.data.gpu_data_dirty = false;
  }

  static async sync_buffers(context) {}
}
