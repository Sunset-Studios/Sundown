import { Renderer } from '../../renderer/renderer.js'; 
import { SimulationLayer } from '../simulation_layer.js';
import { SharedViewBuffer } from '../shared_data.js';
import { global_dispatcher } from '../dispatcher.js';

export class ViewProcessor extends SimulationLayer {
    scene = null;

    init() {
        super.init();
        global_dispatcher.on("resolution_change", this.on_resolution_change.bind(this));
    }

    cleanup() {
        scene = null;
        global_dispatcher.off("resolution_change", this.on_resolution_change.bind(this));
        super.cleanup();
    }

    update(delta_time) {
        super.update(delta_time);
        SharedViewBuffer.update_transforms();
    }

    set_scene(scene) {
        this.scene = scene;
        this.context.current_view = scene.context.current_view;
        
        const view_data = SharedViewBuffer.get_view_data(this.scene.context.current_view);
        view_data.aspect_ratio = Renderer.get().aspect_ratio;
    }

    on_resolution_change() {
        const view_data = SharedViewBuffer.get_view_data(this.scene.context.current_view);
        view_data.aspect_ratio = Renderer.get().aspect_ratio;
    }
}