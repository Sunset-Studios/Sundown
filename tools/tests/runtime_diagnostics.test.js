import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";
import application_state from "../../engine/src/core/application_state.js";
import { DiagnosticsReport } from "../../engine/src/utility/diagnostics_report.js";

test("engine ignores harness flags, deadlines and launcher bridges", async (context) => {
  const original_warn = console.warn;
  const original_error = console.error;
  const original_timeout = globalThis.setTimeout;
  let timers = 0;
  globalThis.location = { search: "?smoke=1&timeout=1", href: "http://localhost/" };
  globalThis.addEventListener = () => {};
  globalThis.sundown_smoke = {
    send() {
      throw new Error("Engine must not contact a launcher");
    },
  };
  globalThis.setTimeout = () => {
    timers++;
  };
  context.after(() => {
    console.warn = original_warn;
    console.error = original_error;
    globalThis.setTimeout = original_timeout;
    delete globalThis.location;
    delete globalThis.addEventListener;
    delete globalThis.sundown_smoke;
    delete globalThis.sundown_diagnostics;
  });
  const diagnostics = await import("../../engine/src/core/diagnostics.js?engine-client-test");
  assert.equal(diagnostics.get_diagnostics_report(), null);
  const report = diagnostics.install_diagnostics({ launcher: "custom", scene: "scene" });
  assert.equal(diagnostics.install_diagnostics(), report);
  assert.equal(report.launcher, "custom");
  const received = [];
  const unsubscribe = diagnostics.subscribe_diagnostics((diagnostic) => received.push(diagnostic));
  console.warn("engine warning");
  assert.equal(received[0].category, "console");
  assert.equal(report.diagnostics[0].message, "engine warning");
  unsubscribe();
  diagnostics.report_diagnostic({
    severity: "warning",
    message: "unsubscribed",
    source: "engine.js",
  });
  assert.equal(received.length, 1);
  assert.equal(report.diagnostics.length, 2);
  assert.equal(report.status, "running");
  assert.equal(timers, 0);
});

test("engine collects runtime/GPU diagnostics and drains compilation without controlling execution", async (context) => {
  const listeners = new Map();
  const original_warn = console.warn;
  const original_error = console.error;
  const previous_running = application_state.is_running;
  application_state.is_running = true;
  globalThis.addEventListener = (name, callback) => listeners.set(name, callback);
  context.after(() => {
    console.warn = original_warn;
    console.error = original_error;
    application_state.is_running = previous_running;
    delete globalThis.addEventListener;
    delete globalThis.sundown_diagnostics;
  });
  const diagnostics = await import("../../engine/src/core/diagnostics.js?gpu-test");
  const report = diagnostics.install_diagnostics();
  const work = diagnostics.begin_diagnostic_work("custom asset");
  assert.equal(diagnostics.has_pending_diagnostic_work(), true);
  diagnostics.end_diagnostic_work(work);
  assert.equal(diagnostics.has_pending_diagnostic_work(), false);
  function warn() {
    console.warn("repeated warning");
  }
  warn();
  warn();
  assert.equal(report.diagnostics[0].count, 2);
  assert.ok(report.diagnostics[0].source.endsWith("runtime_diagnostics.test.js"));
  listeners.get("unhandledrejection")({ reason: new Error("async failure") });
  listeners.get("error")({ message: "runtime failure", filename: "scene.js", lineno: 8, colno: 2 });
  let uncaptured_error;
  let lose_device;
  diagnostics.observe_gpu_device({
    addEventListener: (_name, callback) => {
      uncaptured_error = callback;
    },
    lost: new Promise((resolve) => {
      lose_device = resolve;
    }),
  });
  uncaptured_error({ error: new Error("invalid GPU binding") });
  lose_device({ reason: "unknown", message: "device failed" });
  let resolve_compilation;
  diagnostics.observe_shader_compilation(
    {
      getCompilationInfo: () =>
        new Promise((resolve) => {
          resolve_compilation = resolve;
        }),
    },
    "test.wgsl",
    "variant"
  );
  let drained = false;
  const completion = diagnostics.wait_for_diagnostics().then(() => {
    drained = true;
  });
  await Promise.resolve();
  assert.equal(drained, false);
  resolve_compilation({
    messages: [{ type: "error", message: "invalid WGSL", lineNum: 3, linePos: 4 }],
  });
  await completion;
  const shader_error = report.diagnostics.find(
    (diagnostic) => diagnostic.category === "wgsl-compilation"
  );
  assert.equal(shader_error.source, "test.wgsl");
  assert.equal(shader_error.line, 3);
  assert.equal(shader_error.column, 4);
  assert.ok(report.diagnostics.some((diagnostic) => diagnostic.category === "webgpu-device-lost"));
  assert.equal(report.status, "running");
  assert.equal(application_state.is_running, true);
});

test("generic diagnostic reports have no asset or frame success requirements", () => {
  const report = new DiagnosticsReport("test", "context");
  assert.equal(report.finish(), 0);
  assert.equal("readiness" in report.toJSON(), false);
  report.add({ severity: "warning", message: "late warning" });
  assert.equal(report.status, "failed");
});

test("engine source has no smoke-test policies, imports or launcher bridge", () => {
  function inspect(directory) {
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const url = new URL(entry.name + (entry.isDirectory() ? "/" : ""), directory);
      if (entry.isDirectory()) inspect(url);
      else if (entry.name.endsWith(".js"))
        assert.doesNotMatch(readFileSync(url, "utf8"), /smoke/i, url.pathname);
    }
  }
  inspect(new URL("../../engine/src/", import.meta.url));
});
