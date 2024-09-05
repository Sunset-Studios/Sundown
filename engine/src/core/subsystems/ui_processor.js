import { SimulationLayer } from "../simulation_layer.js";
import { profile_scope } from "../../utility/performance.js";

export class UIProcessor extends SimulationLayer {
  #ui_root = null;

  // Update UI before any other processing. This allows us to steal input if necessary.
  pre_update(delta_time) {
    super.pre_update(delta_time);
    profile_scope("UIProcessor.pre_update", () => {
      if (this.#ui_root) {
        this.#ui_root.update(delta_time);
      }
    });
  }

  set_ui_root(ui_root) {
    this.#ui_root = ui_root;
  }
}
