import { SharedFrameInfoBuffer } from "../core/shared_data.js";
import { Renderer } from "../renderer/renderer.js";

export default class SimulationCore {
    simulation_layers = [];

    constructor() {
        if (SimulationCore.instance) {
            return SimulationCore.instance;
        }
        SimulationCore.instance = this;
    }

    static get() {
        if (!SimulationCore.instance) {
            SimulationCore.instance = new SimulationCore()
        }
        return SimulationCore.instance;
    }

    async register_simulation_layer(layer) {
        this.simulation_layers.push(layer);
        await layer.init();
    }

    unregister_simulation_layer(layer) {
        layer.cleanup();
        this.simulation_layers.splice(this.simulation_layers.indexOf(layer), 1);
    }

    update(delta_time) {
        performance.mark('simulation_core_update');

        const time = SharedFrameInfoBuffer.get().get_time();
        SharedFrameInfoBuffer.get().set_time(Renderer.get().graphics_context, time + delta_time);

        for (const layer of this.simulation_layers) {
            layer.pre_update(delta_time);
        }

        for (const layer of this.simulation_layers) {
            layer.update(delta_time);
        }

        for (const layer of this.simulation_layers) {
            layer.post_update(delta_time);
        }
    }
}