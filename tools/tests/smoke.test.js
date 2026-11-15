import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { DiagnosticsReport } from "../../engine/src/utility/diagnostics_report.js";
import { SmokeSceneReport } from "../smoke/scene_report.js";
import { finalize_scene_launch } from "../smoke/launch_report.js";
import { SmokeSession } from "../../desktop/smoke.js";

test("diagnostics aggregate by severity, category, message and source location", () => {
  const report = new DiagnosticsReport("test", "scene");
  const diagnostic = { severity: "warning", message: "repeated", source: "test.js", line: 12 };
  report.add(diagnostic);
  report.add(diagnostic);
  report.add({ ...diagnostic, line: 13 });
  assert.equal(report.diagnostics.length, 2);
  assert.equal(report.diagnostics[0].count, 2);
  assert.equal(report.finish(), 1);
  assert.equal(JSON.parse(JSON.stringify(report)).schema_version, 1);
  assert.equal("entries" in report.toJSON(), false);
});

test("success requires assets, completed GPU work, and no diagnostics", () => {
  const report = new SmokeSceneReport("test", "scene");
  assert.equal(report.finish(), 1);
  assert.equal(report.finish({ assets_ready: false, completed_frames: 5 }), 1);
  assert.equal(report.finish({ assets_ready: true, completed_frames: 0 }), 1);
  assert.equal(report.finish({ assets_ready: true, completed_frames: 5 }), 0);
  report.add({ severity: "error", message: "late validation error" });
  assert.equal(report.status, "failed");
});

function create_launch_fixture() {
  const report = new SmokeSceneReport("electron", "sundown://app/?scene=FixtureScene");
  const process_result = {
    code: 0,
    signal: null,
    stdout: "renderer output",
    stderr: "process detail",
  };
  const options = { frames: 7, output: "/reports/FixtureScene.json" };
  return { report, process_result, options };
}

test("launcher accepts a finalized success and records process output", () => {
  const { report, process_result, options } = create_launch_fixture();
  report.finish({ assets_ready: true, completed_frames: 7 });
  const result = finalize_scene_launch(report, report.toJSON(), process_result, options);
  assert.equal(result.status, "passed");
  assert.equal(result.process.exit_code, 0);
  assert.equal(result.process.stdout, "renderer output");
  assert.equal(result.process.stderr, "process detail");
});

test("launcher preserves real failures, locations, stacks and repeat counts", () => {
  const { report, process_result, options } = create_launch_fixture();
  const diagnostic = {
    severity: "error",
    category: "webgpu-validation",
    message: "invalid binding",
    source: "test.wgsl",
    line: 12,
    column: 4,
    stack: "original stack",
  };
  report.add(diagnostic);
  report.add(diagnostic);
  report.finish({ assets_ready: true, completed_frames: 7 });
  const result = finalize_scene_launch(
    report,
    report.toJSON(),
    { ...process_result, code: 1 },
    options
  );
  assert.equal(result.status, "failed");
  assert.equal(result.diagnostics.length, 1);
  assert.equal(result.diagnostics[0].count, 2);
  assert.equal(result.diagnostics[0].source, "test.wgsl");
  assert.equal(result.diagnostics[0].stack, "original stack");
});

test("launcher preserves partial diagnostics when a process exits without finalizing", () => {
  const { report, process_result, options } = create_launch_fixture();
  const partial = report.toJSON();
  partial.diagnostics = [
    { severity: "error", message: "actual cause", source: "renderer.js", line: 42, count: 8 },
  ];
  const result = finalize_scene_launch(
    report,
    partial,
    { ...process_result, code: null, signal: "SIGSEGV" },
    options
  );
  assert.equal(result.status, "failed");
  assert.equal(result.diagnostics[0].message, "actual cause");
  assert.equal(result.diagnostics[0].count, 8);
  assert.match(result.diagnostics[1].message, /status: running/);
  assert.match(result.diagnostics[1].message, /FixtureScene.json/);
  assert.match(result.diagnostics[1].message, /SIGSEGV/);
  assert.equal(result.process.stderr, "process detail");
});

test("launcher explains missing, incomplete and inconsistent results", () => {
  for (const failure of [
    "missing",
    "timeout",
    "read-error",
    "exit",
    "empty-failure",
    "frames",
    "scene",
  ]) {
    const { report, process_result, options } = create_launch_fixture();
    report.finish({ assets_ready: true, completed_frames: 7 });
    let saved = report.toJSON();
    if (failure === "missing") saved = null;
    if (failure === "timeout") options.timed_out = true;
    if (failure === "read-error") options.error = new SyntaxError("invalid JSON");
    if (failure === "exit") process_result.code = 1;
    if (failure === "empty-failure") saved.status = "failed";
    if (failure === "frames") saved.readiness.completed_frames = 5;
    if (failure === "scene") saved.scene = "another scene";
    const result = finalize_scene_launch(
      new SmokeSceneReport("electron", report.scene),
      saved,
      process_result,
      options
    );
    assert.equal(result.status, "failed", failure);
    assert.ok(result.diagnostics.length > 0, failure);
    assert.equal(result.diagnostics.at(-1).source, options.output, failure);
    assert.equal(result.process.stderr, "process detail", failure);
  }
});

function create_runner(context, timeout_ms = 2000, ipc_main = new EventEmitter()) {
  const directory = mkdtempSync(path.join(tmpdir(), "sundown-smoke-test-"));
  context.after(() => rmSync(directory, { recursive: true, force: true }));
  const output = path.join(directory, "diagnostics.json");
  const window = new EventEmitter();
  window.webContents = new EventEmitter();
  window.webContents.mainFrame = {};
  const exit_codes = [];
  let resolve_exit;
  const exited = new Promise((resolve) => {
    resolve_exit = resolve;
  });
  const session = new SmokeSession(
    {
      exit(code) {
        exit_codes.push(code);
        resolve_exit(code);
      },
    },
    window,
    ipc_main,
    {
      output,
      frames: 3,
      timeout_ms,
      url: "sundown://app/?smoke=1&scene=BVHScene",
    }
  );
  const event = { sender: window.webContents, senderFrame: window.webContents.mainFrame };
  return { session, window, ipc_main, event, exited, exit_codes, output };
}

test("SmokeSession exposes startup failure without returning a callback", async (context) => {
  const runner = create_runner(context);
  runner.session.fail("startup", "Startup failed before navigation completed");
  assert.equal(await runner.exited, 1);
  const report = JSON.parse(readFileSync(runner.output));
  assert.equal(report.diagnostics[0].category, "startup");
  assert.equal(report.diagnostics[0].message, "Startup failed before navigation completed");
  runner.session.fail("startup", "Duplicate failure");
  runner.window.webContents.emit("console-message", { level: "error", message: "late error" });
  assert.deepEqual(runner.exit_codes, [1]);
  assert.equal(readFileSync(runner.output, "utf8"), `${JSON.stringify(report, null, 2)}\n`);
  assert.equal(runner.ipc_main.listenerCount("sundown:smoke"), 0);
  assert.equal(runner.window.eventNames().length, 0);
  assert.equal(runner.window.webContents.eventNames().length, 0);
});

test("SmokeSession keeps shared IPC subscriptions scoped to each window", async (context) => {
  const ipc_main = new EventEmitter();
  const first = create_runner(context, 2000, ipc_main);
  const second = create_runner(context, 2000, ipc_main);
  first.session.fail("startup", "First window failed");
  assert.equal(await first.exited, 1);
  assert.equal(ipc_main.listenerCount("sundown:smoke"), 1);
  ipc_main.emit("sundown:smoke", first.event, {
    type: "diagnostic",
    diagnostic: { message: "Wrong window" },
  });
  ipc_main.emit("sundown:smoke", second.event, {
    type: "complete",
    readiness: { assets_ready: true, completed_frames: 3 },
  });
  assert.equal(await second.exited, 0);
  assert.deepEqual(JSON.parse(readFileSync(second.output)).diagnostics, []);
  assert.equal(ipc_main.listenerCount("sundown:smoke"), 0);
});

test("Electron captures console warnings before readiness and repeat counts", async (context) => {
  const runner = create_runner(context);
  const details = { level: "warning", message: "warning", sourceId: "scene.js", lineNumber: 7 };
  runner.window.webContents.emit("console-message", details);
  runner.window.webContents.emit("console-message", details);
  runner.ipc_main.emit("sundown:smoke", runner.event, {
    type: "complete",
    readiness: { assets_ready: true, completed_frames: 3 },
  });
  assert.equal(await runner.exited, 1);
  const report = JSON.parse(readFileSync(runner.output));
  assert.equal(report.diagnostics[0].count, 2);
  assert.equal(report.diagnostics[0].source, "scene.js");
  assert.equal(runner.ipc_main.listenerCount("sundown:smoke"), 0);
});

test("Electron accepts readiness only from its main frame", async (context) => {
  const runner = create_runner(context);
  runner.ipc_main.emit("sundown:smoke", { sender: {} }, { type: "complete" });
  runner.ipc_main.emit("sundown:smoke", runner.event, {
    type: "complete",
    readiness: { assets_ready: true, completed_frames: 3 },
  });
  assert.equal(await runner.exited, 0);
});

test("Electron rejects premature readiness", async (context) => {
  const runner = create_runner(context);
  runner.ipc_main.emit("sundown:smoke", runner.event, {
    type: "complete",
    readiness: { assets_ready: true, completed_frames: 1 },
  });
  assert.equal(await runner.exited, 1);
  assert.equal(JSON.parse(readFileSync(runner.output)).diagnostics[0].category, "readiness");
});

test("Electron finishes promptly when the renderer harness reports a fatal failure", async (context) => {
  const runner = create_runner(context);
  runner.ipc_main.emit("sundown:smoke", runner.event, {
    type: "diagnostic",
    diagnostic: { severity: "error", category: "scene-assets", message: "asset failed" },
  });
  runner.ipc_main.emit("sundown:smoke", runner.event, { type: "failed" });
  assert.equal(await runner.exited, 1);
  const report = JSON.parse(readFileSync(runner.output));
  assert.equal(report.diagnostics.length, 1);
  assert.equal(report.diagnostics[0].message, "asset failed");
});

for (const failure of ["timeout", "navigation", "renderer-crash", "window-closed"]) {
  test(`Electron writes a report on ${failure}`, async (context) => {
    const runner = create_runner(context, failure === "timeout" ? 10 : 2000);
    if (failure === "navigation")
      runner.window.webContents.emit(
        "did-fail-load",
        {},
        -6,
        "missing",
        "sundown://app/missing",
        true
      );
    if (failure === "renderer-crash")
      runner.window.webContents.emit("render-process-gone", {}, { reason: "crashed", exitCode: 1 });
    if (failure === "window-closed") runner.window.emit("closed");
    assert.equal(await runner.exited, 1);
    const report = JSON.parse(readFileSync(runner.output));
    assert.equal(report.status, "failed");
    assert.equal(report.diagnostics[0].category, failure);
  });
}
