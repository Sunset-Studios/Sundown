// ═══════════════════════════════════════════════════════════════════════════════
// Browser-side harness adapter. The engine exposes diagnostics; this tooling
// owns deadlines, fatal-error policy, readiness results and launcher transport.
// ═══════════════════════════════════════════════════════════════════════════════

import application_state from "../../engine/src/core/application_state.js";
import {
  install_diagnostics,
  subscribe_diagnostics,
  wait_for_diagnostics,
} from "../../engine/src/core/diagnostics.js";
import { SmokeSceneReport } from "./scene_report.js";

const parameters = typeof location === "undefined" ? null : new URLSearchParams(location.search);

export function is_smoke_test() {
  return parameters?.has("smoke") ?? false;
}

export class SmokeRuntime {
  get scene_finished() {
    return this.finished;
  }

  constructor(options) {
    this.frames = options.frames;
    this.bridge = options.bridge ?? null;
    this.finished = false;
    this.completing = false;
    this.timeout = null;
    this.report = new SmokeSceneReport(this.bridge ? "electron" : "browser", options.scene);
    install_diagnostics({ launcher: this.report.launcher, scene: options.scene });
    this.unsubscribe = subscribe_diagnostics(this.receive_diagnostic.bind(this));
    if (
      !Number.isSafeInteger(options.frames) ||
      options.frames < 1 ||
      !Number.isSafeInteger(options.timeout_ms) ||
      options.timeout_ms < 1
    ) {
      this.receive_diagnostic({
        severity: "error",
        category: "configuration",
        message: "Smoke frames and timeout must be positive integers",
      });
      this.fail();
    } else {
      this.timeout = setTimeout(() => {
        this.receive_diagnostic({
          severity: "error",
          category: "timeout",
          message: `Scene did not complete within ${options.timeout_ms}ms`,
          source: options.scene,
        });
        this.fail();
      }, options.timeout_ms);
    }
  }

  receive_diagnostic(diagnostic) {
    if (this.finished) return;
    this.report.add(diagnostic);
    // Electron captures native and early console output itself; avoid double counts.
    if (diagnostic.category !== "console") this.bridge?.send({ type: "diagnostic", diagnostic });
    if (
      diagnostic.category === "uncaught-exception" ||
      diagnostic.category === "unhandled-rejection"
    )
      this.fail();
  }

  finish(readiness = null) {
    if (this.finished) return;
    this.finished = true;
    clearTimeout(this.timeout);
    this.unsubscribe();
    application_state.is_running = false;
    this.report.finish(readiness);
    this.bridge?.send(readiness ? { type: "complete", readiness } : { type: "failed" });
  }

  fail() {
    this.finish();
  }

  async complete(completed_frames) {
    if (this.finished || this.completing) return;
    if (!Number.isSafeInteger(completed_frames) || completed_frames < this.frames) {
      this.receive_diagnostic({
        severity: "error",
        category: "readiness",
        message: "Scene completed without the required GPU frames",
      });
      this.fail();
      return;
    }
    this.completing = true;
    try {
      await wait_for_diagnostics();
      // GPU/console notifications can arrive after queue completion.
      await new Promise((resolve) => setTimeout(resolve, 250));
      this.finish({ assets_ready: true, completed_frames });
    } catch (error) {
      this.receive_diagnostic({
        severity: "error",
        category: "diagnostics",
        message: error.message,
        stack: error.stack,
      });
      this.fail();
    }
  }
}

/**
 * Drives an ordered scene sequence inside one application instance. Each scene
 * owns a report and deadline; diagnostics are routed to whichever scene is
 * active. The application activates its scenes and awaits
 * `when_scene_finished()` before moving on.
 */
export class SmokeSequenceRuntime {
  is_sequence = true;
  scene_finished = true;
  preliminary = true;

  constructor(options) {
    this.frames = options.frames;
    this.timeout_ms = options.timeout_ms;
    this.bridge = options.bridge ?? null;
    this.finished = false;
    this.completing = false;
    this.scene = parameters?.getAll("scene")[0] ?? "scene-sequence";
    this.report = new SmokeSceneReport(this.bridge ? "electron" : "browser", this.scene);
    this.timeout = null;
    this.resolve_scene = null;
    install_diagnostics({
      launcher: this.bridge ? "electron" : "browser",
      scene: this.scene,
    });
    this.unsubscribe = subscribe_diagnostics(this.receive_diagnostic.bind(this));
  }

  begin_scene(scene) {
    if (this.finished) return;
    const reuse_startup_report = this.preliminary && scene === this.scene;
    this.preliminary = false;
    if (!reuse_startup_report) {
      this.scene = scene;
      this.report = new SmokeSceneReport(this.bridge ? "electron" : "browser", scene);
    }
    this.scene_finished = false;
    this.completing = false;
    this.bridge?.send({ type: "scene-begin", scene });

    if (
      !Number.isSafeInteger(this.frames) ||
      this.frames < 1 ||
      !Number.isSafeInteger(this.timeout_ms) ||
      this.timeout_ms < 1
    ) {
      this.receive_diagnostic({
        severity: "error",
        category: "configuration",
        message: "Smoke frames and timeout must be positive integers",
      });
      this.abort();
      return;
    }

    this.timeout = setTimeout(() => {
      this.receive_diagnostic({
        severity: "error",
        category: "timeout",
        message: `Scene '${scene}' did not complete within ${this.timeout_ms}ms`,
        source: scene,
      });
      this.fail();
    }, this.timeout_ms);
  }

  receive_diagnostic(diagnostic) {
    if (this.finished || !this.report) return;
    this.report.add(diagnostic);
    // Electron captures native and early console output itself; avoid double counts.
    if (diagnostic.category !== "console") this.bridge?.send({ type: "diagnostic", diagnostic });
    if (
      diagnostic.category === "uncaught-exception" ||
      diagnostic.category === "unhandled-rejection"
    ) {
      this.fail();
    } else if (diagnostic.category === "webgpu-device-lost") {
      this.abort();
    }
  }

  fail_scene(error) {
    if (this.scene_finished) return;
    this.receive_diagnostic({
      severity: "error",
      category: "scene-assets",
      message: error.message,
      source: this.scene,
      stack: error.stack,
    });
    this.fail();
  }

  finish_scene(readiness = null) {
    if (this.scene_finished) return;
    this.scene_finished = true;
    clearTimeout(this.timeout);
    this.timeout = null;

    const scene = this.scene;
    const report = this.report;
    this.report = null;
    report.finish(readiness);
    this.bridge?.send(
      readiness ? { type: "scene-complete", scene, readiness } : { type: "scene-failed", scene }
    );

    const resolve_scene = this.resolve_scene;
    this.resolve_scene = null;
    resolve_scene?.(scene);
  }

  fail() {
    this.finish_scene();
  }

  async complete(completed_frames) {
    if (this.scene_finished || this.completing) return;
    if (!Number.isSafeInteger(completed_frames) || completed_frames < this.frames) {
      this.receive_diagnostic({
        severity: "error",
        category: "readiness",
        message: "Scene completed without the required GPU frames",
      });
      this.fail();
      return;
    }
    this.completing = true;
    try {
      await wait_for_diagnostics();
      // GPU/console notifications can arrive after queue completion.
      await new Promise((resolve) => setTimeout(resolve, 250));
      if (!this.scene_finished) this.finish_scene({ assets_ready: true, completed_frames });
    } catch (error) {
      this.receive_diagnostic({
        severity: "error",
        category: "diagnostics",
        message: error.message,
        stack: error.stack,
      });
      this.fail();
    }
  }

  abort(error = null) {
    if (this.finished) return;
    if (error) {
      this.receive_diagnostic({
        severity: "error",
        category: "harness",
        message: error.message,
        stack: error.stack,
      });
    }
    this.fail();
    this.finished = true;
    this.unsubscribe();
    clearTimeout(this.timeout);
    application_state.is_running = false;
    this.bridge?.send({ type: "suite-aborted" });
  }

  finish_sequence() {
    if (this.finished) return;
    this.finished = true;
    this.unsubscribe();
    clearTimeout(this.timeout);
    application_state.is_running = false;
    this.bridge?.send({ type: "suite-complete" });
  }

  when_scene_finished() {
    if (this.scene_finished) return Promise.resolve(this.scene);
    return new Promise((resolve) => {
      this.resolve_scene = resolve;
    });
  }
}

function create_smoke_runtime() {
  if (!is_smoke_test()) return null;

  const options = {
    scene: location.href,
    frames: Number(parameters.get("frames") ?? 5),
    timeout_ms: Number(parameters.get("timeout") ?? 60000),
    bridge: globalThis.sundown_smoke,
  };
  return parameters.get("switch") === "1"
    ? new SmokeSequenceRuntime(options)
    : new SmokeRuntime(options);
}

const runtime = create_smoke_runtime();

// Sequence mode keeps the engine's aggregate report; each scene report is separate.
if (runtime && !runtime.is_sequence) globalThis.sundown_diagnostics = runtime.report;

export function get_smoke_runtime() {
  return runtime;
}
