import { EntityManager } from "./ecs/entity.js";
import { SharedViewBuffer } from "./shared_data.js";
import { Renderer } from "../renderer/renderer.js";

export class LayerContext {
    current_view = null;
    entity_manager = null;
}

export class SimulationLayer {
    layers = []
    context = new LayerContext();

    constructor() {
        this.name = "SimulationLayer";
    }

    init() {
        this.context.entity_manager = EntityManager.get();
    }

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
        return this.layers.find(layer => layer.constructor.name === prototype.name);
    }
}