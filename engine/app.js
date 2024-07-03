import Renderer from "@/renderer/renderer.js";
import SimulationCore from "@/core/simulation_core.js";
import { InputProvider } from "@/input/input_provider.js";
import { Scene } from "@/core/layers/scene.js";
import application_state from "@/core/application_state.js";

async function init() {
  application_state.is_running = true;

  const input_provider = InputProvider.get();
  await SimulationCore.get().register_simulation_layer(input_provider);
  input_provider.push_context(InputProvider.default_context());

  const canvas = document.getElementById("gpu-canvas");
  await Renderer.get().setup(canvas);

  const scene = new Scene("TestScene");
  await SimulationCore.get().register_simulation_layer(scene);
}

function run() {
  function simulate() {
    if (application_state.is_running) {
      performance.mark("frame_start");

      SimulationCore.get().update();

      Renderer.get().render();

      requestAnimationFrame(simulate);
    }
  }
  requestAnimationFrame(simulate);
}

await init();
run();
