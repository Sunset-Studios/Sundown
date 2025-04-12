import { EntityLinearDataContainer } from "../entity_utils.js";
import { Fragment } from "../fragment.js";
import { Renderer } from "../../../renderer/renderer.js";
import { Buffer } from "../../../renderer/buffer.js";
import { global_dispatcher } from "../../../core/dispatcher.js";
import { RingBufferAllocator } from "../../../memory/allocator.js";
import { EntityID } from "../entity.js";
import { EntityManager } from "../entity.js";
import { MAX_BUFFERED_FRAMES } from "../../../core/minimal.js";

class StaticMeshDataView {
  current_entity = -1n;
  absolute_entity = -1n;

  constructor() {}

  get mesh() {
    return StaticMeshFragment.data.mesh[this.absolute_entity];
  }

  set mesh(value) {
    StaticMeshFragment.data.mesh[this.absolute_entity] =
      StaticMeshFragment.data.mesh instanceof BigInt64Array
        ? BigInt(value)
        : value;
    if (StaticMeshFragment.data.dirty) {
      StaticMeshFragment.data.dirty[this.absolute_entity] = 1;
    }
    StaticMeshFragment.data.gpu_data_dirty = true;
  }

  get material_slots() {
    return StaticMeshFragment.data.material_slots.slice(
      this.absolute_entity * StaticMeshFragment.material_slot_stride,
      (this.absolute_entity + 1) * StaticMeshFragment.material_slot_stride,
    );
  }

  set material_slots(value) {
    if (
      Array.isArray(value) &&
      value.length <= StaticMeshFragment.material_slot_stride
    ) {
      for (let i = 0; i < value.length; i++) {
        StaticMeshFragment.data.material_slots[
          this.absolute_entity * StaticMeshFragment.material_slot_stride + i
        ] = BigInt(value[i]);
      }
    }
    if (StaticMeshFragment.data.dirty) {
      StaticMeshFragment.data.dirty[this.absolute_entity] = 1;
    }
    StaticMeshFragment.data.gpu_data_dirty = true;
  }

  get dirty() {
    return StaticMeshFragment.data.dirty[this.absolute_entity];
  }

  set dirty(value) {
    StaticMeshFragment.data.dirty[this.absolute_entity] =
      StaticMeshFragment.data.dirty instanceof BigInt64Array
        ? BigInt(value)
        : value;
    if (StaticMeshFragment.data.dirty) {
      StaticMeshFragment.data.dirty[this.absolute_entity] = 1;
    }
    StaticMeshFragment.data.gpu_data_dirty = true;
  }

  view_entity(entity, instance = 0) {
    this.current_entity = entity;
    this.absolute_entity = EntityID.get_absolute_index(entity) + instance;

    return this;
  }
}

const unmapped_state = "unmapped";

export class StaticMeshFragment extends Fragment {
  static data_view_allocator = new RingBufferAllocator(256, StaticMeshDataView);
  static size = 0;
  static data = null;

  static material_slot_stride = 16;

  static initialize() {
    this.data = {
      mesh: new BigInt64Array(1),
      material_slots: new BigInt64Array(16),
      dirty: new Uint8Array(1),
      gpu_data_dirty: true,
    };
  }

  static resize(new_size) {
    if (new_size <= this.size) return;

    new_size *= 2;
    this.size = new_size;

    if (!this.data) this.initialize();

    Fragment.resize_array(this.data, "mesh", new_size, BigInt64Array, 1);
    Fragment.resize_array(
      this.data,
      "material_slots",
      new_size,
      BigInt64Array,
      16,
    );
    Fragment.resize_array(this.data, "dirty", new_size, Uint8Array, 1);
  }

  static add_entity(entity) {
    const absolute_entity = EntityID.get_absolute_index(entity);
    if (absolute_entity >= this.size) {
      this.resize(absolute_entity * 2);
    }

    return this.get_entity_data(entity);
  }

  static remove_entity(entity) {
    const entity_offset = EntityID.get_absolute_index(entity);
    const entity_instances = EntityID.get_instance_count(entity);
    for (let i = 0; i < entity_instances; i++) {
      this.data.mesh[entity_offset + i] = 0n;
      for (let j = 0; j < this.material_slot_stride; j++) {
        this.data.material_slots[
          (entity_offset + i) * this.material_slot_stride + j
        ] = 0n;
      }
      this.data.dirty[entity_offset + i] = 1;
    }
  }

  static get_entity_data(entity, instance = 0) {
    const data_view = this.data_view_allocator.allocate();
    data_view.view_entity(entity, instance);
    return data_view;
  }

  static duplicate_entity_data(entity, instance = 0) {
    const data = {};
    const entity_index = EntityID.get_absolute_index(entity);
    data.mesh = this.data.mesh[entity_index];
    data.material_slots = Array(16).fill(0);
    for (let i = 0; i < 16; i++) {
      data.material_slots[i] = this.data.material_slots[entity_index * 16 + i];
    }
    data.dirty = this.data.dirty[entity_index];
    return data;
  }

  static copy_entity_instance(to_index, from_index) {
    this.data.mesh[to_index * 1 + 0] = this.data.mesh[from_index * 1 + 0];

    this.data.material_slots[to_index * 16 + 0] =
      this.data.material_slots[from_index * 16 + 0];
    this.data.material_slots[to_index * 16 + 1] =
      this.data.material_slots[from_index * 16 + 1];
    this.data.material_slots[to_index * 16 + 2] =
      this.data.material_slots[from_index * 16 + 2];
    this.data.material_slots[to_index * 16 + 3] =
      this.data.material_slots[from_index * 16 + 3];
    this.data.material_slots[to_index * 16 + 4] =
      this.data.material_slots[from_index * 16 + 4];
    this.data.material_slots[to_index * 16 + 5] =
      this.data.material_slots[from_index * 16 + 5];
    this.data.material_slots[to_index * 16 + 6] =
      this.data.material_slots[from_index * 16 + 6];
    this.data.material_slots[to_index * 16 + 7] =
      this.data.material_slots[from_index * 16 + 7];
    this.data.material_slots[to_index * 16 + 8] =
      this.data.material_slots[from_index * 16 + 8];
    this.data.material_slots[to_index * 16 + 9] =
      this.data.material_slots[from_index * 16 + 9];
    this.data.material_slots[to_index * 16 + 10] =
      this.data.material_slots[from_index * 16 + 10];
    this.data.material_slots[to_index * 16 + 11] =
      this.data.material_slots[from_index * 16 + 11];
    this.data.material_slots[to_index * 16 + 12] =
      this.data.material_slots[from_index * 16 + 12];
    this.data.material_slots[to_index * 16 + 13] =
      this.data.material_slots[from_index * 16 + 13];
    this.data.material_slots[to_index * 16 + 14] =
      this.data.material_slots[from_index * 16 + 14];
    this.data.material_slots[to_index * 16 + 15] =
      this.data.material_slots[from_index * 16 + 15];

    this.data.dirty[to_index * 1 + 0] = this.data.dirty[from_index * 1 + 0];

    this.data.gpu_data_dirty = true;
  }
}
