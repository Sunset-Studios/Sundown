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

const runtime = is_smoke_test()
  ? new SmokeRuntime({
      scene: location.href,
      frames: Number(parameters.get("frames") ?? 5),
      timeout_ms: Number(parameters.get("timeout") ?? 60000),
      bridge: globalThis.sundown_smoke,
    })
  : null;

if (runtime) globalThis.sundown_diagnostics = runtime.report;

export function get_smoke_runtime() {
  return runtime;
}
