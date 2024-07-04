export class SimulationLayer {
    layers = []
    constructor() {
        this.name = "SimulationLayer";
    }

    init() { }

    pre_update() {
        for (const layer of this.layers) {
            layer.pre_update();
        }
    }

    update(delta_time) {
        for (const layer of this.layers) {
            layer.update(delta_time);
        }
    }

    post_update() {
        for (const layer of this.layers) {
            layer.post_update();
        }
    }

    add_layer(prototype) {
        let found_layer = this.get_layer(prototype);
        if (found_layer) {
            return found_layer;
        }

        const layer = new prototype();
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
        return this.layers.find(layer => layer.constructor.name === prototype.constructor.name);
    }
}