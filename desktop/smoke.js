// ═══════════════════════════════════════════════════════════════════════════════
// Per-window smoke session — owns diagnostic collection, IPC and timer lifetime.
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
