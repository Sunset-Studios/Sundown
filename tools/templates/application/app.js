import { Simulator } from "../engine/src/core/simulator.js";
import { Renderer } from "../engine/src/renderer/renderer.js";
import { are_engine_assets_ready } from "../engine/src/utility/asset_readiness.js";
import { is_smoke_test, get_smoke_runtime } from "../tools/smoke/runtime.js";
import { SmokeTest } from "../tools/smoke/controller.js";
import { MainScene } from "./main_scene.js";

async function main() {
  const smoke = is_smoke_test();
  const runtime = get_smoke_runtime();

  if (smoke) {
    const scenes = new URLSearchParams(location.search).getAll("scene");
    if (runtime.is_sequence || scenes.length !== 1 || scenes[0] !== "MainScene") {
      throw new Error("This app supports isolated smoke runs for MainScene only");
    }
  }

  const simulator = await Simulator.create("gpu-canvas", "ui-canvas", {
    project: { name: "__APP_NAME__", root: "app" },
    pointer_lock: !smoke,
  });

  const scene = new MainScene("MainScene");
  simulator.add_sim_layer(scene);

  if (smoke) {
    new SmokeTest(Renderer.get(), runtime.frames, are_engine_assets_ready, runtime).set_scene(
      scene
    );
  }

  simulator.run();
}

void main().catch((error) => {
  console.error(error);
  const runtime = get_smoke_runtime();
  if (runtime?.is_sequence) runtime.abort(error);
  else runtime?.fail();
});
