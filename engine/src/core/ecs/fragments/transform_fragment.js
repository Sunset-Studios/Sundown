import { Fragment } from '@/core/ecs/fragment';

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
            }
        };
    }

    static resize() {
        const resize_array = (obj, key) => {
            if (obj[key].length < this.size) {
                const prev = obj[key];
                obj[key] = new Float32Array(this.size);
                obj[key].set(prev);
            }
        };

        ['position', 'rotation', 'scale'].forEach(prop => {
            ['x', 'y', 'z'].forEach(axis => {
                resize_array(this.data[prop], axis);
            });
        });
    }

    static to_gpu_data() {
        const gpu_data = new Float32Array(this.size * 9);
        for (let i = 0; i < this.size; i++) {
            const offset = i * 9;
            gpu_data[offset] = this.data.position.x[i];
            gpu_data[offset + 1] = this.data.position.y[i];
            gpu_data[offset + 2] = this.data.position.z[i];
            gpu_data[offset + 3] = this.data.rotation.x[i];
            gpu_data[offset + 4] = this.data.rotation.y[i];
            gpu_data[offset + 5] = this.data.rotation.z[i];
            gpu_data[offset + 6] = this.data.scale.x[i];
            gpu_data[offset + 7] = this.data.scale.y[i];
            gpu_data[offset + 8] = this.data.scale.z[i];
        }
        return gpu_data;
    }
}