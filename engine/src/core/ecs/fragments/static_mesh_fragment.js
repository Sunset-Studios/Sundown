import { Fragment } from "../fragment.js";

export class StaticMeshFragment extends Fragment {
  static material_slot_stride = 64;

  static initialize() {
    this.data = {
      mesh: new BigInt64Array(1),
      material_slots: new BigInt64Array(this.material_slot_stride),
      instance_count: new BigInt64Array(1),
      dirty: new Uint8Array(1),
    };
  }

  static remove_entity(entity) {
    super.remove_entity(entity);
    this.update_entity_data(entity, {
      mesh: 0n,
      material_slots: Array(this.material_slot_stride).fill(0),
      instance_count: 0n,
    });
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

  static resize(new_size) {
    super.resize(new_size);

    Fragment.resize_array(this.data, "mesh", new_size, BigInt64Array);
    Fragment.resize_array(this.data, "instance_count", new_size, BigInt64Array);
    Fragment.resize_array(
      this.data,
      "material_slots",
      new_size,
      BigInt64Array,
      this.material_slot_stride
    );
    Fragment.resize_array(this.data, "dirty", new_size, Uint8Array);
  }
}
