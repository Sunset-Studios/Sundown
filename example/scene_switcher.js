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
      await this.set_scene_index((this.current_scene_index + 1) % this.scenes.length);
    }
  }

  async add_scene(scene) {
    this.scenes.push(scene);
    if (this.current_scene_index === null) {
      await this.set_scene_index(0);
    }
  }

  /** Activates a scene immediately, releasing and replacing the current one. */
  async set_scene(scene) {
    if (this.current_scene_index === null) {
      this.scenes.push(scene);
      await this.set_scene_index(0);
      return;
    }

    const index = this.current_scene_index;
    SimulationCore.unregister_simulation_layer(this.scenes[index]);
    this.scenes[index] = scene;
    await SimulationCore.register_simulation_layer(scene);
  }

  async set_scene_index(index) {
    if (this.current_scene_index !== null) {
      SimulationCore.unregister_simulation_layer(this.scenes[this.current_scene_index]);
    }

    this.current_scene_index = index;
    await SimulationCore.register_simulation_layer(this.scenes[index]);
  }
}
