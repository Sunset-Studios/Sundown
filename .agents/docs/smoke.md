# Runtime smoke checks

Read when changing runtime diagnostics, application scene selection, the smoke
harness, Electron smoke sessions, or report formats. Paths below are repository-relative.

`npm run smoke -- --catalog path/to/smoke.config.js` cooks/builds once and tests
**every configured application scene** in its own Electron application instance.
It writes `diagnostics.json` and returns **0 only when every selected scene passes**.
Failures do not prevent subsequent scenes from running. Add `--scene-switching`
to reuse one Electron instance and switch scenes through the application's driver;
each scene still has its own report and deadline. The coding agent owns the
inspect → targeted fix → rerun loop; the harness never edits code or retries.

For an agent-driven repair loop, load the
[smoke-repair skill](../skills/smoke-repair/SKILL.md). It defines the iteration
budget, diagnostic triage, targeted-fix safeguards and verification criteria.
The skill enables scene switching by default. For example: “Use smoke-repair for
all configured scenes, with at most three repair iterations,” or “Use smoke-repair
for MainScene, with `max_iterations=2`.” This budget is an agent instruction, not
a harness CLI option.

```sh
npm run smoke -- --catalog app/smoke.config.js --list
npm run smoke -- --catalog app/smoke.config.js --scene-switching
npm run smoke -- --catalog app/smoke.config.js --scene MainScene,EditorScene
npm run smoke -- --scene MainScene --frames 10 --timeout 90000
npm run smoke -- --scene MainScene --output out/smoke/diagnostics.json
npm run smoke -- --no-build --scene MainScene --url http://localhost:3000/
npm run test:smoke
npm run test:smoke:electron
```

The launcher has no built-in scene IDs or imports from an application directory.
`--catalog` is a trusted, Node-safe JavaScript module, resolved from the working
directory, with a default export such as:

```js
export default [{ id: "MainScene" }, { id: "EditorScene" }];
```

Only IDs are required; the launcher never constructs scenes or imports their
browser-only modules. IDs must be nonempty, trimmed and unique after ignoring
case, underscores and hyphens. `--scene` accepts comma-separated or repeated IDs
with those same matching rules. Unknown or empty filters fail. Selection follows
catalog order and removes duplicates. `--list` does not build or launch anything.
Without a catalog, explicit `--scene` IDs define the selection in request order;
the application is responsible for validating them. Omitting both options fails
with configuration guidance rather than assuming a sample application exists.

The repository's sample application is optional. Its convenience commands and
catalog live at the application boundary:
`npm run smoke:example -- --scene-switching` and `npm run test:example`. See its
own README for sample-specific details.

Timeouts are **per scene**, in milliseconds, covering startup, assets and rendering
(not cooking/building). The parent watchdog also covers a stalled Electron main
process. The default timeout is 60 seconds; increase it for large asset-heavy
scenes. Scenes keep their real workloads, including stress tests—there are no
smoke-only scenes, entity limits, software-rendering fallbacks or warning allowlists.
Hardware WebGPU is required. Process isolation avoids leaking entities, resources,
settings or failed devices between scenes; it does not test interactive switching.

`--scene-switching` runs every selected scene in one Electron process. The
application creates and activates one scene at a time using its own scene
lifecycle; each scene keeps its own deadline, diagnostics, readiness result and
checkpoint, and a failed scene does not prevent later scenes from running. Device loss aborts the
remaining sequence because the shared GPU device is no longer trustworthy. Use
the default per-process mode when a failure may contaminate engine state, or when
isolating a scene suspected of leaking resources or settings.

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
`tools/smoke/suite_report.js`. The application's entry point explicitly wires these
helpers for its selected scenes. `Simulator` never installs or drives a test
harness. Warnings always fail the harness, but engine collection alone does not
stop an application.

On the Electron side, `desktop/smoke.js` exposes a per-window `SmokeSession` for
the default isolated mode. It owns its report, event handlers and timers, detaches
them when finished, and provides `fail(category, message)` for startup errors.
`SmokeSequenceSession` implements `--scene-switching`: it validates the requested
scene order, keeps per-scene timers and reports, routes console output to the
active scene, and checkpoints the sequence transport file after each scene. No
internal callback or module-global session state is exposed.

Additional asynchronous startup work can use `begin_diagnostic_work(source)` and
`end_diagnostic_work(token)` in a `finally` block. Tracking is a no-op outside
diagnostic mode. Custom asset systems can extend `are_engine_assets_ready` or use
these tokens; scene metadata alone is not sufficient readiness. Unfinished work
fails through the deadline, never through an arbitrary successful sleep.

## Application integration and production forks

`npm run app:init -- --name my-app` generates a single-scene application with
isolated smoke integration, selects its HTML entry point and adds a `smoke:app`
shortcut. Use `--dry-run` to preview its changes. Conflicts fail instead of
overwriting app files. The optional `--remove-example` flag deletes `example/`
and `assets/example/` and removes the sample npm scripts during initial setup.
Templates live under
`tools/templates/application/` and do not depend on the optional sample app.

`engine/`, `desktop/` and the generic `tools/` tests do not import the sample app.
A production fork can remove the sample directory and its convenience npm scripts.
Replace the script entry in the root `index.html` with the production entry point;
the Vite and Electron launchers continue to serve that application's build.

For smoke support, the application reads the requested `scene` URL parameters,
constructs its normal scenes, and connects `SmokeTest`, `get_smoke_runtime()` and
`are_engine_assets_ready()`. There is no required scene switcher class. Sequence
applications call `begin_scene(id)`, activate the scene, attach a `SmokeTest` with
`stop_application_on_complete=false`, await `when_scene_finished()`, dispose the
controller, then move to the next scene. Call `finish_sequence()` after the last
scene and `abort(error)` for suite-level failures. Retain the 300ms inter-scene
settle used by the protocol to let Electron collect late diagnostics.

Production apps that do not implement sequence activation should use isolated
mode. The desktop launcher only understands generic scene IDs and IPC messages;
it never imports or chooses an application scene. Running Electron with `--smoke`
directly requires application-specific URL/options from `SUNDOWN_SMOKE_OPTIONS`.

## Reports

The single saved artifact, **`diagnostics.json` (schema version 2)**, includes
`selected_scenes`, per-scene results in `scenes`, suite-level diagnostics
(build/launcher/interruption), and aggregate status. In isolated mode it is
checkpointed after each scene. Scene switching checkpoints its private sequence
transport after each scene and writes the saved report when the run finishes.
Suite success requires a successful result for every selection, in order.

Each entry in `scenes` contains the complete scene result (schema version 1):
`sample` (the application scene ID), launcher, scene URL (the scene ID in
scene-switching mode), timestamps, status, readiness (`assets_ready`, `completed_frames`) and
diagnostics. No separate `diagnostics.scenes/` directory is created. `--output`
changes the single report's path.

Electron uses a private temporary report file to send results to the Node
launcher. The launcher embeds each result in the saved report and removes its
temporary directory on completion or handled failure. In scene-switching mode the
file is a checkpointed sequence record containing the requested order, completed
scene results and any suite-level aborted diagnostic. Existing legacy
`diagnostics.scenes/` directories are not read, updated, or automatically deleted.

Each scene result also includes `process`: the Electron exit code/signal and the
last 32,768 characters of stdout/stderr text (with truncation flags). In
scene-switching mode all scene results carry the same shared process record; a
completed scene keeps its own status even when the shared process later exits
nonzero. Launcher failures name the expected report path and preserve any
diagnostics already in a partial report, rather than replacing the original cause
with a generic exit message.

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
`/?smoke=1&scene=MainScene&frames=5&timeout=90000`. This is the **same application
and scene** as Electron. Add `switch=1` and repeated `scene` parameters to
exercise the sequence runtime in a browser. Inspect the engine's aggregate
`window.sundown_diagnostics`, or serialize it with `JSON.stringify`; per-scene
readiness is delivered through the launcher bridge. Browser smoke has its own
readiness watchdog. `?diagnostics=1` collects diagnostics while retaining normal
interactive scenes. Device hooks log failures even outside diagnostic mode; WGSL
inspection runs only in diagnostic mode.

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
frame/timeout options, console capture, preload IPC and one-process scene
switching. It needs a desktop session, but no web build, scene assets or WebGPU
device.
