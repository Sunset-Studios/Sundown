import { EntityLinearDataContainer } from "../entity_utils.js";
import { Fragment } from "../fragment.js";
import { Renderer } from "../../../renderer/renderer.js";
import { Buffer } from "../../../renderer/buffer.js";
import { global_dispatcher } from "../../../core/dispatcher.js";
import { RingBufferAllocator } from "../../../memory/allocator.js";
import { EntityID } from "../entity.js";
import { EntityManager } from "../entity.js";

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

export class StaticMeshFragment extends Fragment {
  static data_view_allocator = new RingBufferAllocator(256, StaticMeshDataView);
  static size = 0;
  static data = null;

  static material_slot_stride = 64;

  static initialize() {
    this.data = {
      mesh: new BigInt64Array(1),
      material_slots: new BigInt64Array(64),
      dirty: new Uint8Array(1),
      gpu_data_dirty: true,
    };
  }

  static resize(new_size) {
    if (new_size <= this.size) return;

    this.size = new_size;

    if (!this.data) this.initialize();

    Fragment.resize_array(this.data, "mesh", new_size, BigInt64Array, 1);
    Fragment.resize_array(
      this.data,
      "material_slots",
      new_size,
      BigInt64Array,
      64,
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
    const entity_data = this.get_entity_data(entity);
    entity_data.mesh = 0n;
    entity_data.material_slots = Array(this.material_slot_stride).fill(0);
  }

  static get_entity_data(entity, instance = 0) {
    const data_view = this.data_view_allocator.allocate();
    data_view.fragment = this;
    data_view.view_entity(entity, instance);
    return data_view;
  }

  static duplicate_entity_data(entity, instance = 0) {
    const data = {};
    const entity_index = EntityID.get_absolute_index(entity);
    data.mesh = this.data.mesh[entity_index];
    data.material_slots = Array(64).fill(0);
    for (let i = 0; i < 64; i++) {
      data.material_slots[i] = this.data.material_slots[entity_index * 64 + i];
    }
    data.dirty = this.data.dirty[entity_index];
    return data;
  }

  static batch_entity_instance_count_changed(index, shift) {
    const source_index = Math.min(Math.max(0, index - shift), this.size - 1);

    this.data.mesh[index * 1 + 0] = this.data.mesh[source_index * 1 + 0];

    this.data.material_slots[index * 64 + 0] =
      this.data.material_slots[source_index * 64 + 0];
    this.data.material_slots[index * 64 + 1] =
      this.data.material_slots[source_index * 64 + 1];
    this.data.material_slots[index * 64 + 2] =
      this.data.material_slots[source_index * 64 + 2];
    this.data.material_slots[index * 64 + 3] =
      this.data.material_slots[source_index * 64 + 3];
    this.data.material_slots[index * 64 + 4] =
      this.data.material_slots[source_index * 64 + 4];
    this.data.material_slots[index * 64 + 5] =
      this.data.material_slots[source_index * 64 + 5];
    this.data.material_slots[index * 64 + 6] =
      this.data.material_slots[source_index * 64 + 6];
    this.data.material_slots[index * 64 + 7] =
      this.data.material_slots[source_index * 64 + 7];
    this.data.material_slots[index * 64 + 8] =
      this.data.material_slots[source_index * 64 + 8];
    this.data.material_slots[index * 64 + 9] =
      this.data.material_slots[source_index * 64 + 9];
    this.data.material_slots[index * 64 + 10] =
      this.data.material_slots[source_index * 64 + 10];
    this.data.material_slots[index * 64 + 11] =
      this.data.material_slots[source_index * 64 + 11];
    this.data.material_slots[index * 64 + 12] =
      this.data.material_slots[source_index * 64 + 12];
    this.data.material_slots[index * 64 + 13] =
      this.data.material_slots[source_index * 64 + 13];
    this.data.material_slots[index * 64 + 14] =
      this.data.material_slots[source_index * 64 + 14];
    this.data.material_slots[index * 64 + 15] =
      this.data.material_slots[source_index * 64 + 15];
    this.data.material_slots[index * 64 + 16] =
      this.data.material_slots[source_index * 64 + 16];
    this.data.material_slots[index * 64 + 17] =
      this.data.material_slots[source_index * 64 + 17];
    this.data.material_slots[index * 64 + 18] =
      this.data.material_slots[source_index * 64 + 18];
    this.data.material_slots[index * 64 + 19] =
      this.data.material_slots[source_index * 64 + 19];
    this.data.material_slots[index * 64 + 20] =
      this.data.material_slots[source_index * 64 + 20];
    this.data.material_slots[index * 64 + 21] =
      this.data.material_slots[source_index * 64 + 21];
    this.data.material_slots[index * 64 + 22] =
      this.data.material_slots[source_index * 64 + 22];
    this.data.material_slots[index * 64 + 23] =
      this.data.material_slots[source_index * 64 + 23];
    this.data.material_slots[index * 64 + 24] =
      this.data.material_slots[source_index * 64 + 24];
    this.data.material_slots[index * 64 + 25] =
      this.data.material_slots[source_index * 64 + 25];
    this.data.material_slots[index * 64 + 26] =
      this.data.material_slots[source_index * 64 + 26];
    this.data.material_slots[index * 64 + 27] =
      this.data.material_slots[source_index * 64 + 27];
    this.data.material_slots[index * 64 + 28] =
      this.data.material_slots[source_index * 64 + 28];
    this.data.material_slots[index * 64 + 29] =
      this.data.material_slots[source_index * 64 + 29];
    this.data.material_slots[index * 64 + 30] =
      this.data.material_slots[source_index * 64 + 30];
    this.data.material_slots[index * 64 + 31] =
      this.data.material_slots[source_index * 64 + 31];
    this.data.material_slots[index * 64 + 32] =
      this.data.material_slots[source_index * 64 + 32];
    this.data.material_slots[index * 64 + 33] =
      this.data.material_slots[source_index * 64 + 33];
    this.data.material_slots[index * 64 + 34] =
      this.data.material_slots[source_index * 64 + 34];
    this.data.material_slots[index * 64 + 35] =
      this.data.material_slots[source_index * 64 + 35];
    this.data.material_slots[index * 64 + 36] =
      this.data.material_slots[source_index * 64 + 36];
    this.data.material_slots[index * 64 + 37] =
      this.data.material_slots[source_index * 64 + 37];
    this.data.material_slots[index * 64 + 38] =
      this.data.material_slots[source_index * 64 + 38];
    this.data.material_slots[index * 64 + 39] =
      this.data.material_slots[source_index * 64 + 39];
    this.data.material_slots[index * 64 + 40] =
      this.data.material_slots[source_index * 64 + 40];
    this.data.material_slots[index * 64 + 41] =
      this.data.material_slots[source_index * 64 + 41];
    this.data.material_slots[index * 64 + 42] =
      this.data.material_slots[source_index * 64 + 42];
    this.data.material_slots[index * 64 + 43] =
      this.data.material_slots[source_index * 64 + 43];
    this.data.material_slots[index * 64 + 44] =
      this.data.material_slots[source_index * 64 + 44];
    this.data.material_slots[index * 64 + 45] =
      this.data.material_slots[source_index * 64 + 45];
    this.data.material_slots[index * 64 + 46] =
      this.data.material_slots[source_index * 64 + 46];
    this.data.material_slots[index * 64 + 47] =
      this.data.material_slots[source_index * 64 + 47];
    this.data.material_slots[index * 64 + 48] =
      this.data.material_slots[source_index * 64 + 48];
    this.data.material_slots[index * 64 + 49] =
      this.data.material_slots[source_index * 64 + 49];
    this.data.material_slots[index * 64 + 50] =
      this.data.material_slots[source_index * 64 + 50];
    this.data.material_slots[index * 64 + 51] =
      this.data.material_slots[source_index * 64 + 51];
    this.data.material_slots[index * 64 + 52] =
      this.data.material_slots[source_index * 64 + 52];
    this.data.material_slots[index * 64 + 53] =
      this.data.material_slots[source_index * 64 + 53];
    this.data.material_slots[index * 64 + 54] =
      this.data.material_slots[source_index * 64 + 54];
    this.data.material_slots[index * 64 + 55] =
      this.data.material_slots[source_index * 64 + 55];
    this.data.material_slots[index * 64 + 56] =
      this.data.material_slots[source_index * 64 + 56];
    this.data.material_slots[index * 64 + 57] =
      this.data.material_slots[source_index * 64 + 57];
    this.data.material_slots[index * 64 + 58] =
      this.data.material_slots[source_index * 64 + 58];
    this.data.material_slots[index * 64 + 59] =
      this.data.material_slots[source_index * 64 + 59];
    this.data.material_slots[index * 64 + 60] =
      this.data.material_slots[source_index * 64 + 60];
    this.data.material_slots[index * 64 + 61] =
      this.data.material_slots[source_index * 64 + 61];
    this.data.material_slots[index * 64 + 62] =
      this.data.material_slots[source_index * 64 + 62];
    this.data.material_slots[index * 64 + 63] =
      this.data.material_slots[source_index * 64 + 63];

    this.data.dirty[index * 1 + 0] = this.data.dirty[source_index * 1 + 0];

    this.data.gpu_data_dirty = true;
  }
}
