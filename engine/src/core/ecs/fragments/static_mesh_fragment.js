import { EntityLinearDataContainer } from "../entity_utils.js";
import { Fragment } from "../fragment.js";
import { Renderer } from "../../../renderer/renderer.js";
import { Buffer } from "../../../renderer/buffer.js";
import { global_dispatcher } from "../../../core/dispatcher.js";
import { RingBufferAllocator } from "../../../memory/allocator.js";

class StaticMeshDataView {
  current_entity = -1;

  constructor() {}

  get mesh() {
    return StaticMeshFragment.data.mesh[this.current_entity];
  }

  set mesh(value) {
    StaticMeshFragment.data.mesh[this.current_entity] =
      StaticMeshFragment.data.mesh instanceof BigInt64Array
        ? BigInt(value)
        : value;
    if (StaticMeshFragment.data.dirty) {
      StaticMeshFragment.data.dirty[this.current_entity] = 1;
    }
    StaticMeshFragment.data.gpu_data_dirty = true;
  }

  get material_slots() {
    return StaticMeshFragment.data.material_slots.slice(
      this.current_entity * StaticMeshFragment.material_slot_stride,
      (this.current_entity + 1) * StaticMeshFragment.material_slot_stride,
    );
  }

  set material_slots(value) {
    if (
      Array.isArray(value) &&
      value.length <= StaticMeshFragment.material_slot_stride
    ) {
      for (let i = 0; i < value.length; i++) {
        StaticMeshFragment.data.material_slots[
          this.current_entity * StaticMeshFragment.material_slot_stride + i
        ] = BigInt(value[i]);
      }
    }
    if (StaticMeshFragment.data.dirty) {
      StaticMeshFragment.data.dirty[this.current_entity] = 1;
    }
    StaticMeshFragment.data.gpu_data_dirty = true;
  }

  get instance_count() {
    return StaticMeshFragment.data.instance_count[this.current_entity];
  }

  set instance_count(value) {
    StaticMeshFragment.data.instance_count[this.current_entity] =
      StaticMeshFragment.data.instance_count instanceof BigInt64Array
        ? BigInt(value)
        : value;
    if (StaticMeshFragment.data.dirty) {
      StaticMeshFragment.data.dirty[this.current_entity] = 1;
    }
    StaticMeshFragment.data.gpu_data_dirty = true;
  }

  get dirty() {
    return StaticMeshFragment.data.dirty[this.current_entity];
  }

  set dirty(value) {
    StaticMeshFragment.data.dirty[this.current_entity] =
      StaticMeshFragment.data.dirty instanceof BigInt64Array
        ? BigInt(value)
        : value;
    if (StaticMeshFragment.data.dirty) {
      StaticMeshFragment.data.dirty[this.current_entity] = 1;
    }
    StaticMeshFragment.data.gpu_data_dirty = true;
  }

  view_entity(entity) {
    this.current_entity = entity;

    return this;
  }
}

export class StaticMeshFragment extends Fragment {
  static data_view_allocator = new RingBufferAllocator(256, StaticMeshDataView);

  static material_slot_stride = 64;

  static initialize() {
    this.data = {
      mesh: new BigInt64Array(1),
      material_slots: new BigInt64Array(64),
      instance_count: new BigInt64Array(1),
      dirty: new Uint8Array(1),
      gpu_data_dirty: true,
    };
  }

  static resize(new_size) {
    if (!this.data) this.initialize();
    super.resize(new_size);

    Fragment.resize_array(this.data, "mesh", new_size, BigInt64Array, 1);
    Fragment.resize_array(
      this.data,
      "material_slots",
      new_size,
      BigInt64Array,
      64,
    );
    Fragment.resize_array(
      this.data,
      "instance_count",
      new_size,
      BigInt64Array,
      1,
    );
    Fragment.resize_array(this.data, "dirty", new_size, Uint8Array, 1);
  }

  static add_entity(entity) {
    super.add_entity(entity);
    return this.get_entity_data(entity);
  }

  static remove_entity(entity) {
    super.remove_entity(entity);
    const entity_data = this.get_entity_data(entity);
    entity_data.mesh = 0n;
    entity_data.material_slots = Array(this.material_slot_stride).fill(0);
    entity_data.instance_count = 0n;
  }

  static get_entity_data(entity) {
    const data_view = this.data_view_allocator.allocate();
    data_view.fragment = this;
    data_view.view_entity(entity);
    return data_view;
  }

  static duplicate_entity_data(entity) {
    const data = {};
    data.mesh = this.data.mesh[entity];
    data.material_slots = Array(64).fill(0);
    for (let i = 0; i < 64; i++) {
      data.material_slots[i] = this.data.material_slots[entity * 64 + i];
    }
    data.instance_count = this.data.instance_count[entity];
    data.dirty = this.data.dirty[entity];
    return data;
  }
}
