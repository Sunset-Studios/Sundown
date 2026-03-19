import application_state from "../engine/src/core/application_state.js";
import SimulationCore from "../engine/src/core/simulation_core.js";
import { EntityManager } from "../engine/src/core/ecs/entity.js";
import { ALL_FRAGMENT_CLASSES } from "../engine/src/core/ecs/fragment_registry.js";
import { InputProvider } from "../engine/src/input/input_provider.js";
import { MetaSystem } from "../engine/src/meta/meta_system.js";
import { BufferSync } from "../engine/src/renderer/buffer.js";
import { Renderer } from "../engine/src/renderer/renderer.js";
import { DeferredShadingStrategy } from "../engine/src/renderer/strategies/deferred_shading.js";
import { reset_ui, flush_ui } from "../engine/src/ui/2d/immediate.js";
import { frame_runner } from "../engine/src/utility/frame_runner.js";

import { SnakeScene } from "./snake_scene.js";

async function bootstrap() {
  application_state.is_running = true;

  InputProvider.setup();
  MetaSystem.setup();

  const canvas = document.getElementById("gpu-canvas");
  const uiCanvas = document.getElementById("ui-canvas");

  if (!canvas || !uiCanvas) {
    throw new Error("Expected both engine canvases to exist.");
  }

  await Renderer.create(canvas, uiCanvas, DeferredShadingStrategy, {
    pointer_lock: true,
    use_precision_float: false,
  });

  EntityManager.setup(ALL_FRAGMENT_CLASSES);
  Renderer.get().refresh_global_shader_bindings();

  const snakeScene = new SnakeScene("SnakeScene");
  SimulationCore.register_simulation_layer(snakeScene);

  frame_runner(async (deltaTime) => {
    if (!application_state.is_running) {
      return;
    }

    BufferSync.process_readbacks();

    const renderer = Renderer.get();
    reset_ui(renderer.canvas_ui?.width ?? 0, renderer.canvas_ui?.height ?? 0);

    InputProvider.update(deltaTime);
    SimulationCore.update(deltaTime);
    renderer.render(deltaTime);
    flush_ui(renderer.context_ui);
  }, 60);
}

bootstrap().catch((error) => {
  console.error("Failed to start Snake example:", error);
});
