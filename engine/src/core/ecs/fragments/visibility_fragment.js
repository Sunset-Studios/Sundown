import { EntityLinearDataContainer } from "../entity_utils.js";
import { Fragment } from "../fragment.js";
import { Renderer } from "../../../renderer/renderer.js";
import { Buffer } from "../../../renderer/buffer.js";
import { global_dispatcher } from "../../../core/dispatcher.js";
import { RingBufferAllocator } from "../../../memory/allocator.js";
import { EntityID } from "../entity.js";

const visible_buffer_name = "visible_buffer";
const visible_cpu_buffer_name = "visible_cpu_buffer";
const visible_event = "visible";
const visible_update_event = "visible_update";

class VisibilityDataView {
  current_entity = -1n;

  constructor() {}

  get visible() {
    return VisibilityFragment.data.visible[this.current_entity];
  }

  set visible(value) {
    VisibilityFragment.data.visible[this.current_entity] =
      VisibilityFragment.data.visible instanceof BigInt64Array
        ? BigInt(value)
        : value;
    if (VisibilityFragment.data.dirty) {
      VisibilityFragment.data.dirty[this.current_entity] = 1;
    }
    VisibilityFragment.data.gpu_data_dirty = true;
  }

  get dirty() {
    return VisibilityFragment.data.dirty[this.current_entity];
  }

  set dirty(value) {
    VisibilityFragment.data.dirty[this.current_entity] =
      VisibilityFragment.data.dirty instanceof BigInt64Array
        ? BigInt(value)
        : value;
    if (VisibilityFragment.data.dirty) {
      VisibilityFragment.data.dirty[this.current_entity] = 1;
    }
    VisibilityFragment.data.gpu_data_dirty = true;
  }

  view_entity(entity) {
    this.current_entity = entity;

    return this;
  }
}

export class VisibilityFragment extends Fragment {
  static data_view_allocator = new RingBufferAllocator(256, VisibilityDataView);

  static initialize() {
    this.data = {
      visible: new Uint32Array(1),
      dirty: new Uint8Array(1),
      visible_buffer: null,
      gpu_data_dirty: true,
    };

    this.rebuild_buffers();
  }

  static resize(new_size) {
    if (!this.data) this.initialize();
    super.resize(new_size);

    Fragment.resize_array(this.data, "visible", new_size, Uint32Array, 1);
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
      this.data.visible[entity_index] = 0;
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
    const data = {};
    const entity_index = EntityID.get_absolute_index(entity) + instance;
    data.visible = this.data.visible[entity_index];
    data.dirty = this.data.dirty[entity_index];
    return data;
  }

  static to_gpu_data() {
    if (!this.data) this.initialize();

    if (!this.data.gpu_data_dirty) {
      return {
        visible_buffer: this.data.visible_buffer,
      };
    }

    this.rebuild_buffers();

    return {
      visible_buffer: this.data.visible_buffer,
    };
  }

  static rebuild_buffers() {
    {
      const gpu_data = this.data.visible
        ? this.data.visible
        : new Uint32Array(this.size * 1);
      if (
        !this.data.visible_buffer ||
        this.data.visible_buffer.config.size < gpu_data.byteLength
      ) {
        this.data.visible_buffer = Buffer.create({
          name: visible_buffer_name,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          raw_data: gpu_data,
          force: true,
        });

        Renderer.get().mark_bind_groups_dirty(true);
        global_dispatcher.dispatch(visible_event, this.data.visible_buffer);
      } else {
        this.data.visible_buffer.write(gpu_data);
      }

      global_dispatcher.dispatch(visible_update_event);
    }

    this.data.gpu_data_dirty = false;
  }

  static async sync_buffers() {}
}
