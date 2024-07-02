import Renderer from "@/renderer/renderer.js";
import SimulationCore from "@/core/simulation_core.js";
import { Scene } from "@/core/layers/scene.js";
import application_state from "@/core/application_state.js";

async function init() {
  application_state.is_running = true;

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
