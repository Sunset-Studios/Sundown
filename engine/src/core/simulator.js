import application_state from "./application_state.js";
import SimulationCore from "./simulation_core.js";
import { EntityManager } from "./ecs/entity.js";
import { Renderer } from "../renderer/renderer.js";
import { BufferSync } from "../renderer/buffer.js";
import { DeferredShadingStrategy } from "../renderer/strategies/deferred_shading.js";
import { InputProvider } from "../input/input_provider.js";
import { MetaSystem } from "../meta/meta_system.js";
import { profile_scope } from "../utility/performance.js";
import { frame_runner } from "../utility/frame_runner.js";
import { ProjectContext } from "./project_context.js";
import { JobSystem } from "../utility/job_system.js";
import { TextureStreamingSystem } from "../renderer/texture.js";

import { ALL_FRAGMENT_CLASSES } from "./ecs/fragment_registry.js";

export class Simulator {
  async init(gpu_canvas_name, ui_canvas_name = null, options = {}) {
    application_state.is_running = true;

    // Configure project context (if provided, otherwise use default project)
    ProjectContext.configure(options.project);
    // Initialize input provider
    InputProvider.setup();
    // Initialize meta system
    MetaSystem.setup();
    // Initialize job system
    JobSystem.install();
    // Initialize texture streaming system
    TextureStreamingSystem.install();
    
    // Initialize renderer with document canvas
    const canvas = document.getElementById(gpu_canvas_name);
    const canvas_ui = ui_canvas_name ? document.getElementById(ui_canvas_name) : null;
    await Renderer.create(canvas, canvas_ui, DeferredShadingStrategy, {
      pointer_lock: options.pointer_lock ?? true,
      use_precision_float: options.use_precision_float ?? false,
      ...(options.renderer || {}),
    });
    
    // Initialize entity manager
    EntityManager.setup(ALL_FRAGMENT_CLASSES);

    // Refresh global shader bindings
    Renderer.get().refresh_global_shader_bindings();
  }

  add_sim_layer(sim_layer) {
    SimulationCore.register_simulation_layer(sim_layer);
  }

  remove_sim_layer(sim_layer) {
    SimulationCore.unregister_simulation_layer(sim_layer);
  }

  async _simulate(delta_time) {
    if (application_state.is_running) {
      BufferSync.process_readbacks();

      profile_scope("frame_loop", () => {
        const renderer = Renderer.get();

        InputProvider.update(delta_time);
        
        SimulationCore.update(delta_time);
        
        renderer.render(delta_time);
      });
    }
  }

  run() {
    frame_runner(this._simulate, 60);
  }

  static async create(gpu_canvas_name, ui_canvas_name = null, options = {}) {
    const instance = new Simulator();
    await instance.init(gpu_canvas_name, ui_canvas_name, options);
    return instance;
  }
}
