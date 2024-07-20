import { Fragment } from '../fragment.js';

export class StaticMeshFragment extends Fragment {
    static material_slot_stride = 64;

    static initialize() {
        this.data = {
            mesh: new BigInt64Array(1),
            material_slots: new BigInt64Array(this.material_slot_stride)
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
    }

    static resize(new_size) {
        super.resize(new_size);

        const resize_array = (obj, key, stride, type) => {
            if (obj[key].length < this.size * stride) {
                const prev = obj[key];
                obj[key] = new type(this.size * stride);
                obj[key].set(prev);
            }
        };

        ['mesh'].forEach(prop => {
            resize_array(this.data, prop, 1, BigInt64Array);
        });

        ['material_slots'].forEach(prop => {
            resize_array(this.data, prop, this.material_slot_stride, BigInt64Array);
        });
    }
}