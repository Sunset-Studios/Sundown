import Renderer from '@/renderer/renderer';
import { SimulationLayer } from '@/core/simulation_layer';
import { MeshTaskQueue } from '@/renderer/mesh_task_queue';
import { FreeformViewControlProcessor } from '@/core/subsystems/freeform_view_control_processor';
import { StaticMeshProcessor } from '@/core/subsystems/static_mesh_processor';
import { Mesh } from '@/renderer/mesh';
import { SharedViewBuffer } from '@/renderer/shared_data';
import application_state from '@/core/application_state';

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

        application_state.current_view = SharedViewBuffer.get().add_view_data();

        this.setup_default_subsystems();
    }

    setup_default_subsystems() {
        this.add_layer(FreeformViewControlProcessor);
        this.add_layer(StaticMeshProcessor);
    }

    update(delta_time) {
        super.update(delta_time);

        performance.mark('scene_update');

        MeshTaskQueue.get().new_task(this.sphere_mesh);
    }
}
