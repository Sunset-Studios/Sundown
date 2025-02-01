import application_state from "./application_state.js";
import SimulationCore from "./simulation_core.js";
import { Renderer } from "../renderer/renderer.js";
import { DeferredShadingStrategy } from "../renderer/strategies/deferred_shading.js";
import { InputProvider } from "../input/input_provider.js";
import { MetaSystem } from "../meta/meta_system.js";
import { profile_scope } from "../utility/performance.js";
import { frame_runner } from "../utility/frame_runner.js";

export class Simulator {
  async init() {
    application_state.is_running = true;

    // Initialize meta system
    const meta_system = MetaSystem.get();
    await SimulationCore.register_simulation_layer(meta_system);

    // Initialize input provider
    const input_provider = InputProvider.get();
    await SimulationCore.register_simulation_layer(input_provider);
    input_provider.push_context(InputProvider.default_context());

    // Initialize renderer with document canvas
    const canvas = document.getElementById("gpu-canvas");
    await Renderer.create(canvas, DeferredShadingStrategy, { pointer_lock: true, use_precision_float: true });
  }

  async add_scene(scene) {
    await SimulationCore.register_simulation_layer(scene);
  }

  remove_scene(scene) {
    SimulationCore.unregister_simulation_layer(scene);
  }

  _simulate(delta_time) {
    if (application_state.is_running) {
      profile_scope("frame_loop", () => {
        SimulationCore.update(delta_time);
        Renderer.get().render(delta_time);
      });
    }
  }

  run() {
    frame_runner(this._simulate, 60);
  }

  static async create() {
    const instance = new Simulator();
    await instance.init();
    return instance;
  }
}
