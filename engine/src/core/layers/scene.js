import SimulationLayer from '@/core/simulation_layer';

export class Scene extends SimulationLayer {
    constructor(name) {
        super();
        this.name = name;
    }

    update(delta_time) {
        performance.mark('scene_update');
    }
}

export default Scene;