export class LayerContext {
    current_view = null;
}

export class SimulationLayer {
    layers = []
    context = new LayerContext();

    constructor() {
        this.name = "SimulationLayer";
    }

    init() {}
    cleanup() {}

    pre_update(delta_time) {
        for (let i = 0; i < this.layers.length; i++) {
            this.layers[i].pre_update(delta_time);
        }
    }

    update(delta_time) {
        for (let i = 0; i < this.layers.length; i++) {
            this.layers[i].update(delta_time);
        }
    }

    post_update(delta_time) {
        for (let i = 0; i < this.layers.length; i++) {
            this.layers[i].post_update(delta_time);
        }
    }

    add_layer(prototype, ...args) {
        let found_layer = this.get_layer(prototype);
        if (found_layer) {
            return found_layer;
        }

        const layer = new prototype(...args);
        layer.init();
        this.layers.push(layer);
        return layer;
    }

    remove_layer(prototype) {
        const layer = this.get_layer(prototype);
        if (layer) {
            this.layers.splice(this.layers.indexOf(layer), 1);
        }
    }

    get_layer(prototype) {
        return this.layers.find(layer => layer.constructor.name === prototype.name);
    }
}