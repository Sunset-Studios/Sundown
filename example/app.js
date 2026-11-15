import { Simulator } from "../engine/src/core/simulator.js";
import { Renderer } from "../engine/src/renderer/renderer.js";
import { are_engine_assets_ready } from "../engine/src/utility/asset_readiness.js";
import { is_smoke_test, get_smoke_runtime } from "../tools/smoke/runtime.js";
import { SmokeTest } from "../tools/smoke/controller.js";
import { create_sample_scene, sample_scenes, select_sample_scenes } from "./samples/registry.js";
import { SceneSwitcher } from "./scene_switcher.js";
import { SceneSettingsPanel } from "./scene_settings_panel.js";
import example_cvar_config from "./config/cvars.js";

async function main() {
  const simulator = await Simulator.create("gpu-canvas", "ui-canvas", {
    project: {
      name: "example",
      root: "example",
      cvar_config: example_cvar_config,
    },
    pointer_lock: !is_smoke_test(),
  });

  if (is_smoke_test()) {
    const filters = new URLSearchParams(location.search).getAll("scene");
    if (!filters.length)
      throw new Error("Smoke instances require a sample scene; use npm run smoke to run the suite");
    const selected = select_sample_scenes(filters);
    if (selected.length !== 1)
      throw new Error("Each smoke instance must select exactly one sample scene");
    const scene = await create_sample_scene(selected[0]);
    simulator.add_sim_layer(scene);
    const runtime = get_smoke_runtime();
    new SmokeTest(Renderer.get(), runtime.frames, are_engine_assets_ready, runtime).set_scene(
      scene
    );
  } else {
    const switcher = new SceneSwitcher("SceneSwitcher");
    for (const sample of sample_scenes) {
      await switcher.add_scene(await create_sample_scene(sample));
    }
    simulator.add_sim_layer(switcher);
    simulator.add_sim_layer(new SceneSettingsPanel(switcher));
  }

  simulator.run();
}

void main();
