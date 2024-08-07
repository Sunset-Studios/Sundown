import { Fragment } from '../fragment.js';

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

    static update_entity_data(entity, data) {
        if (!this.data) {
            this.initialize();
        }

        super.update_entity_data(entity, data);

        if (!Array.isArray(data.material_slots)) {
            throw new Error(`Material slots must be an array for entity ${entity} in fragment ${this.constructor.name}`);
        }

        if (data.material_slots.length > this.material_slot_stride) {
            throw new Error(`Material slots must be less than ${this.material_slot_stride} for entity ${entity} in fragment ${this.constructor.name}`);
        }

        for (let i = 0; i < data.material_slots.length; i++) {
            this.data.material_slots[entity * this.material_slot_stride + i] = BigInt(data.material_slots[i]);
        }

        this.data.dirty[entity] = 1;
    }

    static resize(new_size) {
        super.resize(new_size);

        Fragment.resize_array(this.data, 'mesh', new_size, BigInt64Array);
        Fragment.resize_array(this.data, 'instance_count', new_size, BigInt64Array);
        Fragment.resize_array(this.data, 'material_slots', new_size, BigInt64Array, this.material_slot_stride);
        Fragment.resize_array(this.data, 'dirty', new_size, Uint8Array, 1, true);
    }
}