// ═══════════════════════════════════════════════════════════════════════════════
// Tooling-owned smoke lifecycle — count completed GPU frames only after scene
// data, GLTF hierarchy/mesh work, and streaming uploads have become quiescent.
// No sample-specific readiness signal or reduced workloads are required.
// ═══════════════════════════════════════════════════════════════════════════════

import application_state from "../../engine/src/core/application_state.js";
import { report_diagnostic } from "../../engine/src/core/diagnostics.js";

export class SmokeTest {
  constructor(renderer, frames, assets_ready, runtime) {
    if (!Number.isSafeInteger(frames) || frames < 1) {
      throw new Error("Smoke frames must be a positive integer");
    }
    if (typeof assets_ready !== "function") {
      throw new Error("SmokeTest requires an asset readiness callback");
    }
    if (typeof runtime?.complete !== "function" || typeof runtime?.fail !== "function") {
      throw new Error("SmokeTest requires a harness runtime");
    }
    this.renderer = renderer;
    this.runtime = runtime;
    this.assets_ready = assets_ready;
    this.required_frames = frames;
    this.completed_frames = 0;
    this.scene_data_ready = false;
    this.scene = null;
    this.frame_completed = this.frame_completed.bind(this);
  }

  set_scene(scene) {
    if (this.scene) throw new Error("A smoke instance can run only one scene");
    this.scene = scene;
    scene
      .when_scene_data_ready()
      .then(() => {
        this.scene_data_ready = true;
      })
      .catch((error) => {
        report_diagnostic({
          severity: "error",
          category: "scene-assets",
          message: error.message,
          source: scene.name,
          stack: error.stack,
        });
        this.runtime.fail();
      });
    this.renderer.on_post_render(this.frame_completed);
  }

  frame_completed() {
    if (this.runtime.finished) {
      this.renderer.remove_post_render(this.frame_completed);
      application_state.is_running = false;
      return;
    }
    if (!this.scene_data_ready || !this.assets_ready()) {
      this.completed_frames = 0;
      return;
    }
    if (++this.completed_frames < this.required_frames) return;
    this.renderer.remove_post_render(this.frame_completed);
    application_state.is_running = false;
    void this.runtime.complete(this.completed_frames);
  }
}
