import { SharedFrameInfoBuffer } from "../core/shared_data.js";
import { profile_scope } from "../utility/performance.js";

const simulation_core_update_event_name = "simulation_core_update";

export default class SimulationCore {
  static simulation_layers = [];

  static async register_simulation_layer(layer) {
    this.simulation_layers.push(layer);
    await layer.init();
  }

  static unregister_simulation_layer(layer) {
    layer.cleanup();
    this.simulation_layers.splice(
      this.simulation_layers.indexOf(layer),
      1
    );
  }

  static update(delta_time) {
    profile_scope(simulation_core_update_event_name, () => {
      const time = SharedFrameInfoBuffer.get_time();
      SharedFrameInfoBuffer.set_time(time + delta_time);

      for (const layer of this.simulation_layers) {
        layer.pre_update(delta_time);
      }

      for (const layer of this.simulation_layers) {
        layer.update(delta_time);
      }

      for (const layer of this.simulation_layers) {
        layer.post_update(delta_time);
      }
    });
  }
}
