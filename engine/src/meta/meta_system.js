import { SimulationLayer } from "../core/simulation_layer.js";

export class MetaSystem extends SimulationLayer {
  constructor() {
    if (MetaSystem.instance) {
      return MetaSystem.instance;
    }
    super();

    MetaSystem.instance = this;
  }

  static get() {
    if (!MetaSystem.instance) {
      return new MetaSystem();
    }
    return MetaSystem.instance;
  }
}
