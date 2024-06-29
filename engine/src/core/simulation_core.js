export default class SimulationCore {
    simulation_layers = [];
    current_time = 0;
    delta_time = 0;
    previous_time = 0;

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

    register_simulation_layer(layer) {
        this.simulation_layers.push(layer);
        layer.init();
    }

    unregister_simulation_layer(layer) {
        layer.cleanup();
        this.simulation_layers.splice(this.simulation_layers.indexOf(layer), 1);
    }

    update() {
        performance.mark('simulation_core_update');

        this.current_time = performance.now();
        this.delta_time = Math.min((this.current_time - this.previous_time, 0) / 1000, 0.1);
        this.previous_time = this.current_time;

        for (const layer of this.simulation_layers) {
            layer.pre_update();
        }

        for (const layer of this.simulation_layers) {
            layer.update(this.delta_time);
        }

        for (const layer of this.simulation_layers) {
            layer.post_update();
        }
    }
}