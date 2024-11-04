import { EntityLinearDataContainer } from "../entity_utils.js";
import { Fragment } from "../fragment.js";
import { Renderer } from "../../../renderer/renderer.js";
import { Buffer } from "../../../renderer/buffer.js";
import { global_dispatcher } from "../../../core/dispatcher.js";

export class StaticMeshFragment extends Fragment {
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

  static add_entity(entity, data) {
    super.add_entity(entity, data);
  }

  static remove_entity(entity) {
    super.remove_entity(entity);
    this.update_entity_data(entity, {
      mesh: 0n,
      material_slots: Array(this.material_slot_stride).fill(0),
      instance_count: 0n,
    });
  }

  static get_entity_data(entity) {
    return super.get_entity_data(entity);
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

  static update_entity_data(entity, data) {
    if (!this.data) {
      this.initialize();
    }

    this.data.mesh[entity] = BigInt(data.mesh) ?? 0n;
    this.data.instance_count[entity] = BigInt(data.instance_count) ?? 0n;

    if (
      Array.isArray(data.material_slots) &&
      data.material_slots.length <= this.material_slot_stride
    ) {
      for (let i = 0; i < data.material_slots.length; i++) {
        this.data.material_slots[entity * this.material_slot_stride + i] =
          BigInt(data.material_slots[i]);
      }
    }

    this.data.dirty[entity] = 1;
  }
}
