import SimulationLayer from '@/core/simulation_layer';

export class Scene extends SimulationLayer {
    constructor(name) {
        super();
        this.name = name;
    }

    update(delta_time) {
        console.profile('scene_update');

        console.profileEnd('scene_update');
    }
}

export default Scene;