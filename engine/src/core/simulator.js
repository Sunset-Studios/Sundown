import application_state from "./application_state.js";
import SimulationCore from "./simulation_core.js";
import { Renderer } from "../renderer/renderer.js";
import { DeferredShadingStrategy } from "../renderer/strategies/deferred_shading.js";
import { InputProvider } from "../input/input_provider.js";
import { MetaSystem } from "../meta/meta_system.js";
import { profile_scope } from "../utility/performance.js";
import { frame_runner } from "../utility/frame_runner.js";
import { reset_ui, flush_ui } from "../ui/2d/immediate.js";

export class Simulator {
  async init(gpu_canvas_name, ui_canvas_name = null) {
    application_state.is_running = true;

    // Initialize input provider
    InputProvider.setup();
    // Initialize meta system
    MetaSystem.setup();

    // Initialize renderer with document canvas
    const canvas = document.getElementById(gpu_canvas_name);
    const canvas_ui = ui_canvas_name ? document.getElementById(ui_canvas_name) : null;
    await Renderer.create(canvas, canvas_ui, DeferredShadingStrategy, {
      pointer_lock: true,
      use_precision_float: true,
    });
  }

  async add_sim_layer(sim_layer) {
    await SimulationCore.register_simulation_layer(sim_layer);
  }

  remove_sim_layer(sim_layer) {
    SimulationCore.unregister_simulation_layer(sim_layer);
  }

  _simulate(delta_time) {
    if (application_state.is_running) {
      profile_scope("frame_loop", () => {
        const renderer = Renderer.get();
        reset_ui(renderer.canvas_ui?.width ?? 0, renderer.canvas_ui?.height ?? 0);
        InputProvider.update(delta_time);
        SimulationCore.update(delta_time);
        renderer.render(delta_time);
        flush_ui(renderer.context_ui);
      });
    }
  }

  run() {
    frame_runner(this._simulate, 60);
  }

  static async create(gpu_canvas_name, ui_canvas_name = null) {
    const instance = new Simulator();
    await instance.init(gpu_canvas_name, ui_canvas_name);
    return instance;
  }
}
