import SimulationLayer from '@/core/simulation_layer';
import Renderer from '@/renderer/renderer';
import { MeshTaskQueue } from '@/renderer/mesh_task_queue';
import { Mesh } from '@/renderer/mesh';
import { SharedViewBuffer } from '@/renderer/shared_data';
import { vec4, quat } from 'gl-matrix';
import { radians } from '@/utility/math';
import { InputProvider } from '@/input/input_provider';
import { InputKey } from '@/input/input_types';

export class Scene extends SimulationLayer {
    name = ''
    sphere_mesh = null

    constructor(name) {
        super();

        this.name = name;
    }

    async init() {
        super.init();

        this.sphere_mesh = await Mesh.from_gltf(
            Renderer.get().graphics_context,
            'models/sphere/sphere.gltf',
        );
        this.view = SharedViewBuffer.get().add_view_data();
    }

    update(delta_time) {
        super.update(delta_time);

        performance.mark('scene_update');

        MeshTaskQueue.get().new_task(this.sphere_mesh);

        // TODO: Set the view data somewhere else, preferably in some sort of processor for camera movement.
        const camera_position = vec4.fromValues(0, 0, -2, 1);
        const camera_rotation = quat.fromValues(0, 0, 0, 1);
        SharedViewBuffer.get().set_view_data(Renderer.get().graphics_context, this.view, {
            position: camera_position,
            rotation: camera_rotation,
            aspect_ratio: Renderer.get().graphics_context.aspect_ratio,
            fov: radians(75),
        });

        SharedViewBuffer.get().update_transforms(Renderer.get().graphics_context, this.view);
    }
}