# Runtime smoke checks

Read when changing runtime diagnostics, sample selection, the smoke harness,
Electron smoke sessions, or report formats. Paths below are repository-relative.

`npm run smoke` cooks/builds once and tests **every sample scene** from
`example/samples/registry.js` in its own Electron application instance. It writes
`diagnostics.json` and returns **0 only when every selected scene passes**.
Failures do not prevent subsequent scenes from running. The coding agent owns
the inspect → targeted fix → rerun loop; the harness never edits code or retries.

```sh
npm run smoke
npm run smoke -- --list
npm run smoke -- --scene BVHScene
npm run smoke -- --scene BVHScene,UI3DTestScene --scene GLTFModelScene
npm run smoke -- --no-build --scene UI3DTestScene --frames 10 --timeout 90000
npm run smoke -- --output out/smoke/diagnostics.json
npm run smoke -- --no-build --scene BVHScene --url http://localhost:3000/
npm run test:smoke
npm run test:smoke:electron
```

`--scene` accepts comma-separated or repeated sample IDs, case-insensitively;
snake-case file names without `.js` work too (`bvh_scene`). Unknown or empty
filters fail instead of silently testing nothing. Selection follows catalog
order and removes duplicates. `--list` does not build or launch anything.

Timeouts are **per scene**, in milliseconds, covering startup, assets and rendering
(not cooking/building). The parent watchdog also covers a stalled Electron main
process. The default timeout is 60 seconds; increase it for large asset-heavy
samples. Samples keep their real workloads, including stress tests—there are no
smoke-only scenes, entity limits, software-rendering fallbacks or warning allowlists.
Hardware WebGPU is required. Process isolation avoids leaking entities, resources,
settings or failed devices between samples; it does not test interactive switching.

`--no-build` deliberately reuses the existing build. Otherwise smoke builds are
unminified with source maps, so fixes cannot be tested against stale code. Normal
build settings are unchanged.

## Engine/tooling boundary and readiness

`engine/src/core/diagnostics.js` only collects diagnostics: console/exception
events, GPU validation, device loss, shader compilation information and diagnostic
work tracking. It exposes `subscribe_diagnostics(listener)` (returning an
unsubscribe function) and `wait_for_diagnostics()` to drain compilation inspection.
It has no test flags, deadlines, completion signals, frame thresholds, process
control, or launcher bridge. Generic engine reports do not contain readiness rules.

`tools/smoke/controller.js` observes scene metadata and the engine's general
`are_engine_assets_ready()` query: in-flight GLTF hierarchy loads, cached mesh
builds/meshlet jobs, deferred GPU pipelines, and streaming requests (including
texture uploads). The controller counts consecutive **GPU-completed** frames via
`Renderer.on_post_render`; new activity resets the count.

`tools/smoke/runtime.js` subscribes to engine diagnostics and owns the deadline,
fatal-error policy, compilation drain, final notification delay and launcher
transport. Scene/suite result policies live in `tools/smoke/scene_report.js` and
`tools/smoke/suite_report.js`. `example/app.js` explicitly wires these helpers for
selected samples. `Simulator` never installs or drives a test harness. Warnings
always fail the harness, but engine collection alone does not stop an application.

On the Electron side, `desktop/smoke.js` exposes a per-window `SmokeSession`.
It owns its report, event handlers and timers, detaches them when finished, and
provides `fail(category, message)` for startup errors. No internal callback or
module-global session state is exposed.

Additional asynchronous startup work can use `begin_diagnostic_work(source)` and
`end_diagnostic_work(token)` in a `finally` block. Tracking is a no-op outside
diagnostic mode. Custom asset systems can extend `are_engine_assets_ready` or use
these tokens; scene metadata alone is not sufficient readiness. Unfinished work
fails through the deadline, never through an arbitrary successful sleep.

## Samples and the normal application

Each scene is a separate module in `example/samples`; shared label rendering is in
`world_label.js`. The registry is the single catalog used by the normal sample
switcher and Node smoke launcher. Its lazy imports keep browser-only engine
modules out of Node and load only selected scenes in smoke instances.

To add a sample, export its scene class from a new module and register a durable
ID and lazy import in `registry.js`. The exported class name must match its ID.
No new HTML entry, smoke-specific scene, or per-sample completion callback is needed.
Existing IDs, project root (`example`), scene order and saved scene-data paths
are preserved. The switcher and settings panel remain under `example/`.

## Reports

The single saved artifact, **`diagnostics.json` (schema version 2)**, includes `selected_scenes`, per-scene
results in `scenes`, suite-level diagnostics (build/launcher/interruption), and
aggregate status. It is checkpointed after each scene and finalized when done.
Suite success requires a successful result for every selection, in order.

Each entry in `scenes` contains the complete scene result (schema version 1):
`sample`, launcher, scene URL, timestamps, status, readiness (`assets_ready`,
`completed_frames`) and diagnostics. No separate `diagnostics.scenes/` directory
is created. `--output` changes the single report's path.

Electron uses a private temporary report file to send results to the Node
launcher. The launcher embeds each result in the saved report and removes its
temporary directory on completion or handled failure. Existing legacy
`diagnostics.scenes/` directories are not read, updated, or automatically deleted.

Each scene result also includes `process`: the Electron exit code/signal and the
last 32,768 characters of stdout/stderr text (with truncation flags). Launcher failures name
the expected report path and preserve any diagnostics already in a partial
report, rather than replacing the original cause with a generic exit message.

Diagnostics include severity, category, message, source, line, column, stack and
repeat count. Missing locations use empty strings/zero. GPU messages often
identify resource labels rather than JS locations. WGSL coordinates refer to
**cooked code**, with original shader path and variant, not mapped include lines.
Electron locations refer to build files; source maps live in `dist/assets`.

Navigation failures, runtime exceptions, rejected promises, streaming errors,
GPU validation/device loss, shader warnings/errors, crashes, early closure and
timeouts fail the run. Report-write failures exit nonzero. Stale suite and selected
scene results are replaced before launching so old success cannot mask failure.

## Browser inspection and future launchers

On the existing dev server, visit
`/?smoke=1&scene=BVHScene&frames=5&timeout=90000`. This is the **same application
and sample scene** as Electron. Inspect `window.sundown_diagnostics`, or serialize
it with `JSON.stringify`. Browser smoke has its own readiness watchdog.
`?diagnostics=1` collects diagnostics while retaining normal interactive scenes.
Device hooks log failures even outside diagnostic mode; WGSL inspection runs
only in diagnostic mode.

Regular engine applications can also enable collection with
`Simulator.create(..., { diagnostics: true })`, or call `install_diagnostics()`
before startup. `get_diagnostics_report()` provides the same report directly to
engine clients without depending on a window global or Electron bridge.

A future Chrome launcher can capture early console/navigation/crash events, wait
for `window.sundown_diagnostics.status !== "running"`, and serialize the same
scene report. Only the tooling adapter knows about the launcher. Electron uses a
narrow sandboxed preload bridge only in smoke mode; Node is never exposed to
scenes. These are startup/runtime smoke checks, not visual regression tests or
long-duration stability guarantees.

`test:smoke` runs fast unit/boundary tests. `test:smoke:electron` launches real
Electron windows against local transport fixtures to check custom report paths,
frame/timeout options, console capture and preload IPC. It needs a desktop
session, but no web build, scene assets or WebGPU device.
