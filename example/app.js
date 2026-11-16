import { Simulator } from "../engine/src/core/simulator.js";
import { Renderer } from "../engine/src/renderer/renderer.js";
import { are_engine_assets_ready } from "../engine/src/utility/asset_readiness.js";
import { is_smoke_test, get_smoke_runtime } from "../tools/smoke/runtime.js";
import { SmokeTest } from "../tools/smoke/controller.js";
import { create_sample_scene, sample_scenes, select_sample_scenes } from "./samples/registry.js";
import { SceneSwitcher } from "./scene_switcher.js";
import { SceneSettingsPanel } from "./scene_settings_panel.js";
import example_cvar_config from "./config/cvars.js";

// Let Electron finalize its post-completion diagnostics before the next scene begins.
const smoke_sequence_settle_ms = 300;

/**
 * Runs the selected samples through one SceneSwitcher. Each scene owns a smoke
 * controller, deadline and report; a failed scene does not prevent the remaining
 * samples from running. Rendering starts only once the first scene is active.
 */
async function run_smoke_sequence(switcher, selected, runtime, start_rendering) {
  let rendering_started = false;
  try {
    for (const sample of selected) {
      runtime.begin_scene(sample.id);
      if (runtime.finished) return;

      let smoke_test = null;
      try {
        const scene = await create_sample_scene(sample);
        await switcher.set_scene(scene);
        smoke_test = new SmokeTest(
          Renderer.get(),
          runtime.frames,
          are_engine_assets_ready,
          runtime,
          false
        );
        smoke_test.set_scene(scene);
      } catch (error) {
        runtime.fail_scene(error);
      }

      if (!rendering_started && smoke_test) {
        rendering_started = true;
        start_rendering();
      }

      await runtime.when_scene_finished();
      smoke_test?.dispose();
      if (runtime.finished) return;

      await new Promise((resolve) => setTimeout(resolve, smoke_sequence_settle_ms));
    }
    runtime.finish_sequence();
  } catch (error) {
    runtime.abort(error);
  }
}

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
    const runtime = get_smoke_runtime();

    if (runtime.is_sequence) {
      const switcher = new SceneSwitcher("SceneSwitcher");
      simulator.add_sim_layer(switcher);
      void run_smoke_sequence(switcher, selected, runtime, () => simulator.run());
      return;
    }

    if (selected.length !== 1)
      throw new Error("Each smoke instance must select exactly one sample scene");
    const scene = await create_sample_scene(selected[0]);
    simulator.add_sim_layer(scene);
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
