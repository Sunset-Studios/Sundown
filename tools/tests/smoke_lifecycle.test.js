import assert from "node:assert/strict";
import test from "node:test";
import application_state from "../../engine/src/core/application_state.js";
import { report_diagnostic } from "../../engine/src/core/diagnostics.js";
import { SmokeTest } from "../smoke/controller.js";
import { SmokeRuntime } from "../smoke/runtime.js";

function create_runtime(context, timeout_ms = 2000) {
  const original_warn = console.warn;
  const original_error = console.error;
  const previous_running = application_state.is_running;
  application_state.is_running = true;
  globalThis.addEventListener = () => {};
  const messages = [];
  let resolve_finished;
  const finished = new Promise((resolve) => {
    resolve_finished = resolve;
  });
  const runtime = new SmokeRuntime({
    frames: 3,
    timeout_ms,
    scene: "sample",
    bridge: {
      send(message) {
        messages.push(message);
        if (message.type === "complete" || message.type === "failed") resolve_finished(message);
      },
    },
  });
  context.after(() => {
    runtime.fail();
    console.warn = original_warn;
    console.error = original_error;
    application_state.is_running = previous_running;
    delete globalThis.addEventListener;
    delete globalThis.sundown_diagnostics;
  });
  return { runtime, messages, finished };
}

test("tooling waits for metadata, assets and consecutive GPU-completed frames", async (context) => {
  const { runtime, finished } = create_runtime(context);
  let resolve_scene_data;
  let assets_ready = false;
  let post_render;
  const renderer = {
    on_post_render(callback) {
      post_render = callback;
    },
    remove_post_render(callback) {
      assert.equal(callback, post_render);
      post_render = null;
    },
  };
  const scene_data = new Promise((resolve) => {
    resolve_scene_data = resolve;
  });
  const scene = { name: "sample", when_scene_data_ready: () => scene_data };
  const smoke = new SmokeTest(renderer, 3, () => assets_ready, runtime);
  smoke.set_scene(scene);
  post_render();
  assert.equal(smoke.completed_frames, 0);
  resolve_scene_data();
  await scene_data;
  post_render();
  assert.equal(smoke.completed_frames, 0);
  assets_ready = true;
  post_render();
  post_render();
  assert.equal(smoke.completed_frames, 2);
  assets_ready = false;
  post_render();
  assert.equal(smoke.completed_frames, 0);
  assets_ready = true;
  post_render();
  post_render();
  assert.equal(application_state.is_running, true);
  post_render();
  assert.equal(application_state.is_running, false);
  assert.equal(post_render, null);
  assert.deepEqual(await finished, {
    type: "complete",
    readiness: { assets_ready: true, completed_frames: 3 },
  });
  assert.equal(runtime.report.status, "passed");
});

test("tooling rejects invalid counts and multiple scenes per instance", (context) => {
  const { runtime } = create_runtime(context);
  const renderer = { on_post_render() {} };
  assert.throws(() => new SmokeTest(renderer, 0, () => true, runtime), /positive integer/);
  assert.throws(() => new SmokeTest(renderer, 1.5, () => true, runtime), /positive integer/);
  assert.throws(() => new SmokeTest(renderer, 3), /readiness callback/);
  assert.throws(() => new SmokeTest(renderer, 3, () => true), /harness runtime/);
  const smoke = new SmokeTest(renderer, 3, () => true, runtime);
  const scene = { when_scene_data_ready: () => Promise.resolve() };
  smoke.set_scene(scene);
  assert.throws(() => smoke.set_scene(scene), /only one scene/);
});

test("tooling owns deadlines and ignores late completion", async (context) => {
  const { runtime, messages, finished } = create_runtime(context, 10);
  const completion = runtime.complete(3);
  assert.deepEqual(await finished, { type: "failed" });
  await completion;
  assert.equal(runtime.report.status, "failed");
  assert.equal(runtime.report.diagnostics[0].category, "timeout");
  assert.equal(
    messages.some((message) => message.type === "complete"),
    false
  );
});

test("tooling applies fatal-error and minimum-frame policies", async (context) => {
  const { runtime, finished } = create_runtime(context);
  report_diagnostic({ severity: "error", category: "unhandled-rejection", message: "fatal" });
  assert.deepEqual(await finished, { type: "failed" });
  assert.equal(runtime.report.diagnostics[0].message, "fatal");
  const second = create_runtime(context);
  await second.runtime.complete(1);
  assert.deepEqual(await second.finished, { type: "failed" });
  assert.equal(second.runtime.report.diagnostics[0].category, "readiness");
});
