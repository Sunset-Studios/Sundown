# Sundown sample application

This directory owns the sample scene registry, scene-switching driver, settings
panel and sample-specific tests. It is optional; production forks can remove it.
Initially, the root `index.html` selects `example/app.js` as its application entry.

```sh
npm run smoke:example -- --list
npm run smoke:example -- --scene-switching
npm run smoke:example -- --scene-switching --scene BVHScene,UI3DTestScene
npm run test:example
```

`smoke.config.js` exposes the Node-safe `samples/registry.js` catalog to the generic
launcher. The registry's lazy imports keep browser-only scene modules out of Node.
`app.js` uses `SceneSwitcher` to activate the selected scenes in sequence mode.
These are application choices, not engine or desktop requirements. The sample
smoke commands expect the page to still select the sample application's entry.

To add a sample, export its scene class in `samples/` and register its durable ID
and lazy import in `samples/registry.js`. The exported class name must match the ID.
The registry supplies both interactive scene order and smoke selection. Shared
label rendering uses `world_label` from `engine/src/ui/3d/common.js`.

## Start your own application

Run these commands from the **repository root**, not inside `example/`. You need
Node.js/npm and a machine with hardware WebGPU support.

### 1. Generate the starter

```sh
npm install
npm run app:init -- --name my-game --dry-run
npm run app:init -- --name my-game
```

The preview is optional. Omit `--name` to use `my-app`; names must start with a
lowercase letter and contain only lowercase letters, digits and hyphens.

The command automatically:

- Creates `app/app.js` with simulator startup and isolated smoke-test integration.
- Creates `app/main_scene.js` with a cube, directional light, sky and entity cleanup.
- Creates `app/smoke.config.js`, a Node-safe catalog containing `MainScene`.
- Creates `assets/app/.gitkeep` for your own runtime assets.
- Points the existing module script in `index.html` at `app/app.js`, preserving
  the canvases, stylesheet, security policy and other page content.
- Adds `npm run smoke:app` to `package.json`, preserving other scripts and metadata.

It refuses to overwrite `app/`, `assets/app/` or an existing `smoke:app` script.
The page must contain both existing canvas IDs (`gpu-canvas` and `ui-canvas`) and
exactly one external module script. Setup reports conflicts before writing; move
existing work aside yourself or integrate the templates manually. There is no
force-overwrite option.

**Setup changes which application the page launches.** By default it keeps the
sample content. It does not install dependencies, run builds, commit changes or replace packaging
metadata. `--name` sets the engine project name, not the package name or titles.
Review the changes with `git diff` before continuing. Do not rerun setup to update
an application you have already customized.

To remove the samples **during initial setup**, preview and then run:

```sh
npm run app:init -- --name my-game --remove-example --dry-run
npm run app:init -- --name my-game --remove-example
```

**This deletes `example/` and `assets/example/` in full, including local changes
and this guide**, and removes `smoke:example` and `test:example` from
`package.json`. Save anything you want to keep first. If you already initialized
your app, use the manual removal steps below instead of rerunning setup.

### 2. Run your application

```sh
npm run dev
```

Open the local URL printed by the dev server. You should see the starter's lit
cube and sky. For an Electron development window, stop the browser dev server and
run `npm run devtop` instead. No Vite or desktop launcher changes are required.

### 3. Verify smoke testing

```sh
npm run smoke:app -- --list
npm run smoke:app
```

The first command should print `MainScene`. The second cooks assets, builds your
app and launches Electron. Expect exit code `0` and a fresh `diagnostics.json`
with `status: "passed"`, ready assets, at least five completed GPU frames and no
scene or suite diagnostics. Inspect the report if it fails; do not disable checks
or remove required assets to get a passing result.

The equivalent generic command is:

```sh
npm run smoke -- --catalog app/smoke.config.js
```

**Do not pass `--scene-switching` to the starter.** It implements isolated smoke
runs. A catalog lists IDs; it does not implement scene switching. To support
one-process sequences later, add your own scene replacement/cleanup lifecycle.
Before deleting the sample app, consult `run_smoke_sequence()` in `example/app.js`
and the optional `example/scene_switcher.js`, or read the
[generic protocol guide](../.agents/docs/smoke.md#application-integration-and-production-forks).

### 4. Make it your application

- **Scene content:** edit `app/main_scene.js`. Keep `super.init()`, release entities
  your scene creates, and call `super.cleanup()` when unloading it.
- **Startup and additional scenes:** edit `app/app.js`. Register scenes before
  calling `simulator.run()`. Extend its scene-ID selection together with
  `app/smoke.config.js`; do not import browser-only scene modules into that catalog.
- **Assets:** add textures/models under `assets/app/`. The generated `project.root`
  is `app`, an asset namespace rather than a JavaScript directory. Saved scene
  packages use paths such as `assets/app/scenes/MainScene.scene.bin`; none is
  required for the code-defined starter scene. Keep `assets/engine/`.
- **Smoke behavior:** the generated entry runs the same scene normally and during
  smoke tests. The tooling waits for assets and completed GPU frames; no reduced
  smoke-only scene is needed. Smoke integration remains outside engine code.
- **Branding and distribution:** update `package.json`'s name/version,
  `build.appId`, `build.productName`, icons and platform settings; update the page
  title in `index.html` and window title in `desktop/main.js` yourself.

### 5. Build, then remove the sample app when ready

```sh
npm run test:app-init
npm run test:smoke
npm run test:smoke:electron
npm run build
```

`test:smoke:electron` needs a desktop session. These tests do not require the sample
app. `npm run build` creates the web app in `dist/`; use `npm run buildtop` for an
unpacked Electron build or `npm run make` for installers.

If you kept the samples during setup, remove them once your app passes smoke testing:

1. Remove the `smoke:example` and `test:example` scripts from `package.json`.
2. Save any sample reference material you want, including this guide, outside
   `example/`, then remove `example/` and `assets/example/` yourself.
3. Keep `engine/`, `desktop/`, `tools/` and the built-in engine assets. Setup and
   its templates live under `tools/` and do not depend on `example/`.
4. Run `npm run smoke:app` again to verify your application with the sample directory gone.

Keep the generic `smoke` npm script alongside `smoke:app` so commands such as
`npm run smoke -- --scene MainScene` continue to work.
