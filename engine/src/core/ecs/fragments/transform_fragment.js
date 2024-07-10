import { Buffer } from '../../../renderer/buffer.js';
import { Fragment } from '../fragment.js';

export class TransformFragment extends Fragment {
    static initialize() {
        this.data = {
            position: {
                x: new Float32Array(1),
                y: new Float32Array(1),
                z: new Float32Array(1)
            },
            rotation: {
                x: new Float32Array(1),
                y: new Float32Array(1),
                z: new Float32Array(1)
            },
            scale: {
                x: new Float32Array(1),
                y: new Float32Array(1),
                z: new Float32Array(1)
            },
            prev_world_transform: new Float32Array(16),
            world_transform: new Float32Array(16),
            inverse_world_transform: new Float32Array(16),
            dirty: new Uint8Array(1),
            gpu_buffer: null
        };
    }

    static resize() {
        if (!this.data) {
            this.initialize();
        }

        const resize_array = (obj, key, ArrayType = Float32Array, stride = 1) => {
            if (obj[key].length < this.size * stride) {
                const prev = obj[key];
                obj[key] = new ArrayType(this.size * stride);
                obj[key].set(prev);
            }
        };

        ['position', 'rotation', 'scale'].forEach(prop => {
            ['x', 'y', 'z'].forEach(axis => {
                resize_array(this.data[prop], axis);
            });
        });

        resize_array(this.data, 'prev_world_transform', Float32Array, 16);
        resize_array(this.data, 'world_transform', Float32Array, 16);
        resize_array(this.data, 'inverse_world_transform', Float32Array, 16);
        resize_array(this.data, 'dirty', Uint8Array);
    }

    static update_entity_data(entity, data) {
        if (!this.data) {
            this.initialize();
        }

        super.update_entity_data(entity, data);
        this.data.dirty[entity] = 1;
    }

    static to_gpu_data(context) {
        if (!this.data) {
            this.initialize();
        }

        const gpu_data = new Float32Array(Math.max(this.size * 32, 32));
        for (let i = 0; i < this.size; i++) {
            const offset = i * 32;
            for (let j = 0; j < 16; j++) {
                gpu_data[offset + j] = this.data.world_transform[offset + j];
            }
            for (let j = 0; j < 16; j++) {
                gpu_data[offset + 16 + j] = this.data.inverse_world_transform[offset + j];
            }
        }

        // Resize the buffer if necessary
        if (this.data.gpu_buffer && this.data.gpu_buffer.config.size < gpu_data.byteLength) {
            this.data.gpu_buffer.destroy(context);
            this.data.gpu_buffer = null
        }

        if (!this.data.gpu_buffer) {
            this.data.gpu_buffer = Buffer.create(context, {
                name: "transform_fragment_buffer",
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                raw_data: gpu_data,
            });
        } else {
            this.data.gpu_buffer.write(context, gpu_data);
        }

        return { gpu_data, gpu_buffer: this.data.gpu_buffer };
    }
}