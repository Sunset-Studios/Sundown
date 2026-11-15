// ═══════════════════════════════════════════════════════════════════════════════
// SceneSwitcher — Sundown samples
// ═══════════════════════════════════════════════════════════════════════════════

import SimulationCore from "../engine/src/core/simulation_core.js";
import { SimulationLayer } from "../engine/src/core/simulation_layer.js";
import { InputProvider } from "../engine/src/input/input_provider.js";
import { InputKey } from "../engine/src/input/input_types.js";

export class SceneSwitcher extends SimulationLayer {
  current_scene_index = null;
  scenes = [];

  constructor(name) {
    super();
    this.name = name;
  }

  async update(delta_time) {
    super.update(delta_time);

    if (InputProvider.get_action(InputKey.K_Return)) {
      if (this.current_scene_index !== null) {
        SimulationCore.unregister_simulation_layer(this.scenes[this.current_scene_index]);
      }

      this.current_scene_index = (this.current_scene_index + 1) % this.scenes.length;

      await SimulationCore.register_simulation_layer(this.scenes[this.current_scene_index]);
    }
  }

  async add_scene(scene) {
    if (this.current_scene_index === null) {
      this.current_scene_index = 0;
      await SimulationCore.register_simulation_layer(scene);
    }
    this.scenes.push(scene);
  }
}
