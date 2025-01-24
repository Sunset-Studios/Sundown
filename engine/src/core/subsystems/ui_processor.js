import { Element } from "../../ui/2d/element.js";
import { SimulationLayer } from "../simulation_layer.js";
import { profile_scope } from "../../utility/performance.js";

export class UIProcessor extends SimulationLayer {
  pre_update(delta_time) {
    super.pre_update(delta_time);
    profile_scope("UIProcessor.pre_update", () => {
      const view_roots = Element.get_all_view_roots();
      for (let i = 0; i < view_roots.length; i++) {
        view_roots[i]?.update(delta_time);
      }
    });
  }
}
