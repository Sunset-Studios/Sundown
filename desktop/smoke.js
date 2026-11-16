// ═══════════════════════════════════════════════════════════════════════════════
// Per-window smoke sessions — isolated and scene-switching modes own diagnostic
// collection, IPC, timers and report transport for one Electron window.
// Callers may report a startup failure; completion remains an internal concern.
// ═══════════════════════════════════════════════════════════════════════════════

import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { SmokeSceneReport } from "../tools/smoke/scene_report.js";

export class SmokeSession {
  #app;
  #window;
  #web_contents;
  #ipc_main;
  #options;
  #report;
  #finished = false;
  #completing = false;
  #timeout = null;
  #completion_timeout = null;

  constructor(app, window, ipc_main, options) {
    this.#app = app;
    this.#window = window;
    this.#web_contents = window.webContents;
    this.#ipc_main = ipc_main;
    this.#options = options;
    this.#report = new SmokeSceneReport("electron", options.url);

    ipc_main.on("sundown:smoke", this.#receive);
    this.#web_contents.on("console-message", this.#on_console_message);
    this.#web_contents.on("did-fail-load", this.#on_load_failure);
    this.#web_contents.on("render-process-gone", this.#on_process_gone);
    window.on("unresponsive", this.#on_unresponsive);
    window.on("closed", this.#on_closed);
    this.#timeout = setTimeout(
      () => this.fail("timeout", `Scene did not complete within ${options.timeout_ms}ms`),
      options.timeout_ms
    );
  }

  fail(category, message) {
    if (this.#finished) return;
    this.#report.add({ severity: "error", category, message });
    this.#finish();
  }

  #finish(readiness = null) {
    if (this.#finished) return;
    this.#finished = true;
    this.#detach();
    const exit_code = this.#report.finish(readiness);
    try {
      mkdirSync(path.dirname(this.#options.output), { recursive: true });
      writeFileSync(this.#options.output, `${JSON.stringify(this.#report, null, 2)}\n`);
      console.log(`Smoke ${this.#report.status}: ${this.#options.url}`);
    } catch (error) {
      console.error("Failed to write smoke diagnostics:", error);
      this.#app.exit(1);
      return;
    }
    this.#app.exit(exit_code);
  }

  #detach() {
    clearTimeout(this.#timeout);
    clearTimeout(this.#completion_timeout);
    this.#ipc_main.removeListener("sundown:smoke", this.#receive);
    this.#web_contents.removeListener("console-message", this.#on_console_message);
    this.#web_contents.removeListener("did-fail-load", this.#on_load_failure);
    this.#web_contents.removeListener("render-process-gone", this.#on_process_gone);
    this.#window.removeListener("unresponsive", this.#on_unresponsive);
    this.#window.removeListener("closed", this.#on_closed);
  }

  #receive = (event, message) => {
    if (
      this.#finished ||
      event.sender !== this.#web_contents ||
      event.senderFrame !== this.#web_contents.mainFrame
    )
      return;

    if (message?.type === "diagnostic" && message.diagnostic?.message) {
      this.#report.add(message.diagnostic);
    } else if (message?.type === "failed") {
      if (!this.#report.diagnostics.length) {
        this.#report.add({
          severity: "error",
          category: "scene",
          message: "Scene reported failure",
        });
      }
      this.#finish();
    } else if (message?.type === "complete" && !this.#completing) {
      const readiness = message.readiness;
      if (
        readiness?.assets_ready !== true ||
        !Number.isSafeInteger(readiness.completed_frames) ||
        readiness.completed_frames < this.#options.frames
      ) {
        this.fail(
          "readiness",
          "Scene reported completion without the required assets and GPU frames"
        );
        return;
      }
      this.#completing = true;
      // Allow Chromium's asynchronous console/validation notifications to arrive.
      this.#completion_timeout = setTimeout(() => this.#finish(readiness), 250);
    }
  };

  #on_console_message = (details) => {
    if (this.#finished || (details.level !== "warning" && details.level !== "error")) return;
    this.#report.add({
      severity: details.level,
      category: "console",
      message: details.message,
      source: details.sourceId,
      line: details.lineNumber,
    });
  };

  #on_load_failure = (_event, code, description, url, main_frame) => {
    if (main_frame) this.fail("navigation", `${description} (${code}): ${url}`);
  };

  #on_process_gone = (_event, details) => {
    this.fail("renderer-crash", `${details.reason} (${details.exitCode})`);
  };

  #on_unresponsive = () => {
    this.fail("renderer-hang", "Renderer became unresponsive");
  };

  #on_closed = () => {
    this.fail("window-closed", "Window closed before smoke completion");
  };
}

/**
 * Drives an ordered scene sequence inside one application instance. The renderer
 * activates application scenes; this session owns per-scene reports,
 * deadlines, console routing and checkpoint writes. Missing or failed scenes do
 * not prevent later samples from running.
 */
export class SmokeSequenceSession {
  #app;
  #window;
  #web_contents;
  #ipc_main;
  #options;
  #suite;
  #expected;
  #index = 0;
  #scene = null;
  #report = null;
  #scene_started = false;
  #finished = false;
  #completing = false;
  #startup_timeout = null;
  #timeout = null;
  #completion_timeout = null;
  #pending_readiness = null;

  constructor(app, window, ipc_main, options) {
    if (!Array.isArray(options.scenes) || !options.scenes.length) {
      throw new Error("Sequence smoke sessions require an ordered list of scenes");
    }

    this.#app = app;
    this.#window = window;
    this.#web_contents = window.webContents;
    this.#ipc_main = ipc_main;
    this.#options = options;
    this.#expected = options.scenes.slice();
    this.#suite = {
      schema_version: 1,
      selected_scenes: this.#expected.slice(),
      scene: null,
      scenes: [],
      suite_diagnostics: [],
      aborted: false,
      finished: false,
    };

    ipc_main.on("sundown:smoke", this.#receive);
    this.#web_contents.on("console-message", this.#on_console_message);
    this.#web_contents.on("did-fail-load", this.#on_load_failure);
    this.#web_contents.on("render-process-gone", this.#on_process_gone);
    window.on("unresponsive", this.#on_unresponsive);
    window.on("closed", this.#on_closed);

    // Capture early console output before the renderer announces its first scene.
    this.#open_scene(this.#expected[0], false);
    this.#save();
  }

  fail(category, message) {
    this.#abort(category, message);
  }

  #open_scene(scene, started) {
    this.#scene = scene;
    this.#report = new SmokeSceneReport("electron", scene);
    this.#scene_started = started;
    this.#suite.scene = scene;
    if (started) {
      this.#start_scene_timer();
    } else {
      this.#startup_timeout = setTimeout(() => {
        this.#startup_timeout = null;
        this.#abort("timeout", `Smoke sequence did not begin within ${this.#options.timeout_ms}ms`);
      }, this.#options.timeout_ms);
    }
  }

  #start_scene_timer() {
    clearTimeout(this.#timeout);
    this.#timeout = setTimeout(() => {
      this.#timeout = null;
      if (!this.#report) return;
      this.#report.add({
        severity: "error",
        category: "timeout",
        message: `Scene '${this.#scene}' did not complete within ${this.#options.timeout_ms}ms`,
        source: this.#scene,
      });
      this.#finalize_scene(null);
    }, this.#options.timeout_ms);
  }

  #finalize_scene(readiness) {
    if (!this.#report) return;
    clearTimeout(this.#timeout);
    clearTimeout(this.#completion_timeout);
    this.#timeout = null;
    this.#completion_timeout = null;
    this.#completing = false;

    const report = this.#report;
    this.#report = null;
    this.#pending_readiness = null;
    report.finish(readiness);
    this.#suite.scenes.push(report.toJSON());
    this.#suite.scene = null;
    console.log(`Smoke ${report.status}: ${this.#scene}`);
    this.#scene = null;
    this.#scene_started = false;
    this.#index++;
    this.#save();
  }

  #abort(category, message) {
    if (this.#finished) return;
    this.#finished = true;
    this.#clear_timers();

    if (message) {
      const diagnostic = { severity: "error", category, message };
      if (this.#report) this.#report.add(diagnostic);
      else this.#suite.suite_diagnostics.push(diagnostic);
    }

    this.#finalize_scene(this.#completing ? this.#pending_readiness : null);
    this.#suite.aborted = true;
    this.#suite.finished = true;
    this.#detach();
    this.#save();
    this.#app.exit(1);
  }

  #finish_sequence() {
    if (this.#finished) return;
    if (this.#completing) this.#finalize_scene(this.#pending_readiness);
    if (this.#report || this.#index !== this.#expected.length) {
      this.#abort("sequence", "Smoke sequence completed without a result for every selected scene");
      return;
    }

    this.#finished = true;
    this.#clear_timers();
    this.#suite.finished = true;
    this.#suite.scene = null;
    this.#detach();
    this.#save();
    const passed =
      this.#suite.scenes.length === this.#expected.length &&
      this.#suite.scenes.every((scene) => scene.status === "passed");
    this.#app.exit(passed ? 0 : 1);
  }

  #clear_timers() {
    clearTimeout(this.#startup_timeout);
    clearTimeout(this.#timeout);
    clearTimeout(this.#completion_timeout);
    this.#startup_timeout = null;
    this.#timeout = null;
    this.#completion_timeout = null;
  }

  #detach() {
    this.#ipc_main.removeListener("sundown:smoke", this.#receive);
    this.#web_contents.removeListener("console-message", this.#on_console_message);
    this.#web_contents.removeListener("did-fail-load", this.#on_load_failure);
    this.#web_contents.removeListener("render-process-gone", this.#on_process_gone);
    this.#window.removeListener("unresponsive", this.#on_unresponsive);
    this.#window.removeListener("closed", this.#on_closed);
  }

  #save() {
    this.#suite.scene = this.#report ? this.#scene : null;
    try {
      mkdirSync(path.dirname(this.#options.output), { recursive: true });
      writeFileSync(this.#options.output, `${JSON.stringify(this.#suite, null, 2)}\n`);
    } catch (error) {
      console.error("Failed to write smoke sequence checkpoints:", error);
    }
  }

  #begin_scene(scene) {
    if (this.#finished) return;
    if (scene !== this.#expected[this.#index]) {
      // The renderer may begin the next scene before Electron's settle timer ends.
      if (this.#completing && scene === this.#expected[this.#index + 1]) {
        this.#finalize_scene(this.#pending_readiness);
      } else {
        this.#abort(
          "sequence",
          `Scene '${scene}' does not match the expected order (${this.#expected[this.#index]})`
        );
        return;
      }
    }
    if (!this.#report) {
      this.#open_scene(scene, true);
      this.#save();
      return;
    }
    if (this.#scene !== scene) {
      this.#abort("sequence", `Scene '${scene}' began while '${this.#scene}' was still active`);
      return;
    }
    if (this.#scene_started) return;

    clearTimeout(this.#startup_timeout);
    this.#startup_timeout = null;
    this.#scene_started = true;
    this.#start_scene_timer();
  }

  #receive = (event, message) => {
    if (
      this.#finished ||
      event.sender !== this.#web_contents ||
      event.senderFrame !== this.#web_contents.mainFrame
    )
      return;

    if (message?.type === "scene-begin" && typeof message.scene === "string") {
      this.#begin_scene(message.scene);
    } else if (message?.type === "diagnostic" && message.diagnostic?.message) {
      this.#report?.add(message.diagnostic);
    } else if (message?.type === "scene-failed") {
      if (!this.#report) return;
      if (!this.#report.diagnostics.length) {
        this.#report.add({
          severity: "error",
          category: "scene",
          message: "Scene reported failure",
        });
      }
      this.#finalize_scene(null);
    } else if (message?.type === "scene-complete") {
      this.#complete_scene(message);
    } else if (message?.type === "suite-complete") {
      this.#finish_sequence();
    } else if (message?.type === "suite-aborted") {
      this.#abort(null, null);
    }
  };

  #complete_scene(message) {
    if (!this.#report || this.#completing) return;
    if (message.scene !== this.#scene) {
      this.#abort(
        "sequence",
        `Scene completion '${message.scene}' does not match the active scene '${this.#scene}'`
      );
      return;
    }
    const readiness = message.readiness;
    if (
      readiness?.assets_ready !== true ||
      !Number.isSafeInteger(readiness.completed_frames) ||
      readiness.completed_frames < this.#options.frames
    ) {
      this.#report.add({
        severity: "error",
        category: "readiness",
        message: "Scene reported completion without the required assets and GPU frames",
      });
      this.#finalize_scene(null);
      return;
    }
    this.#completing = true;
    this.#pending_readiness = readiness;
    // Allow Chromium's asynchronous console/validation notifications to arrive.
    this.#completion_timeout = setTimeout(() => {
      this.#completion_timeout = null;
      this.#finalize_scene(readiness);
    }, 250);
  }

  #on_console_message = (details) => {
    if (this.#finished || !this.#report) return;
    if (details.level !== "warning" && details.level !== "error") return;
    this.#report.add({
      severity: details.level,
      category: "console",
      message: details.message,
      source: details.sourceId,
      line: details.lineNumber,
    });
  };

  #on_load_failure = (_event, code, description, url, main_frame) => {
    if (main_frame) this.#abort("navigation", `${description} (${code}): ${url}`);
  };

  #on_process_gone = (_event, details) => {
    this.#abort("renderer-crash", `${details.reason} (${details.exitCode})`);
  };

  #on_unresponsive = () => {
    this.#abort("renderer-hang", "Renderer became unresponsive");
  };

  #on_closed = () => {
    this.#abort("window-closed", "Window closed before smoke completion");
  };
}
