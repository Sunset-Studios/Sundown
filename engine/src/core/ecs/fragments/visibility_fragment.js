import { EntityLinearDataContainer } from "../entity_utils.js";
import { Fragment } from "../fragment.js";
import { Renderer } from "../../../renderer/renderer.js";
import { Buffer } from "../../../renderer/buffer.js";
import { global_dispatcher } from "../../../core/dispatcher.js";
import { RingBufferAllocator } from "../../../memory/allocator.js";
import { EntityID } from "../entity.js";
import { EntityManager } from "../entity.js";
import { MAX_BUFFERED_FRAMES } from "../../../core/minimal.js";

const visible_buffer_name = "visible_buffer";
const visible_cpu_buffer_name = "visible_cpu_buffer";
const visible_event = "visible";
const visible_update_event = "visible_update";

class VisibilityDataView {
  current_entity = -1n;
  absolute_entity = -1n;

  constructor() {}

  get visible() {
    return VisibilityFragment.data.visible[this.absolute_entity];
  }

  set visible(value) {
    VisibilityFragment.data.visible[this.absolute_entity] =
      VisibilityFragment.data.visible instanceof BigInt64Array
        ? BigInt(value)
        : value;
    if (VisibilityFragment.data.dirty) {
      VisibilityFragment.data.dirty[this.absolute_entity] = 1;
    }
    VisibilityFragment.data.gpu_data_dirty = true;
  }

  get dirty() {
    return VisibilityFragment.data.dirty[this.absolute_entity];
  }

  set dirty(value) {
    VisibilityFragment.data.dirty[this.absolute_entity] =
      VisibilityFragment.data.dirty instanceof BigInt64Array
        ? BigInt(value)
        : value;
    if (VisibilityFragment.data.dirty) {
      VisibilityFragment.data.dirty[this.absolute_entity] = 1;
    }
    VisibilityFragment.data.gpu_data_dirty = true;
  }

  view_entity(entity, instance = 0) {
    this.current_entity = entity;
    this.absolute_entity = EntityID.get_absolute_index(entity) + instance;

    return this;
  }
}

const unmapped_state = "unmapped";

export class VisibilityFragment extends Fragment {
  static data_view_allocator = new RingBufferAllocator(256, VisibilityDataView);
  static size = 0;
  static data = null;

  static initialize() {
    this.data = {
      visible: new Uint32Array(1),
      dirty: new Uint8Array(1),
      visible_buffer: null,
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

    Fragment.resize_array(this.data, "visible", new_size, Uint32Array, 1);
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

    for (let i = 0; i < instance_count; ++i) {
      const entity_index = entity_offset + i;
      this.data.visible[entity_index] =
        this.data.visible instanceof BigInt64Array ? 0n : 0;
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
    const entity_index = EntityID.get_absolute_index(entity);
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
    if (!this.data.gpu_data_dirty) return;

    {
      const gpu_data = this.data.visible
        ? this.data.visible
        : new Uint32Array(this.size * 1 + 1);
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

  static async sync_buffers() {
    const buffered_frame = Renderer.get().get_buffered_frame_number();
  }

  static copy_entity_instance(to_index, from_index) {
    this.data.visible[to_index * 1 + 0] = this.data.visible[from_index * 1 + 0];

    this.data.dirty[to_index * 1 + 0] = this.data.dirty[from_index * 1 + 0];

    this.data.gpu_data_dirty = true;
  }

  static get highest_entity() {
    return this.data.last_valid_index;
  }
}
