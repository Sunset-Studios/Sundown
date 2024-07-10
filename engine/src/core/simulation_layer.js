import { EntityManager } from "./ecs/entity.js";

export class LayerContext {
    current_view = null;
    entity_manager = EntityManager.get();
}

export class SimulationLayer {
    layers = []
    context = new LayerContext();

    constructor() {
        this.name = "SimulationLayer";
    }

    init(parent_context) { }

    pre_update(parent_context) {
        for (const layer of this.layers) {
            layer.pre_update(this.context);
        }
    }

    update(delta_time, parent_context) {
        for (const layer of this.layers) {
            layer.update(delta_time, this.context);
        }
    }

    post_update(parent_context) {
        for (const layer of this.layers) {
            layer.post_update(this.context);
        }
    }

    add_layer(prototype) {
        let found_layer = this.get_layer(prototype);
        if (found_layer) {
            return found_layer;
        }

        const layer = new prototype();
        layer.init(this.context);
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