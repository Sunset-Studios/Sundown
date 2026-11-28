# Sundown Engine 🕹️

An extensible WebGPU game and simulation engine for fun, games, and research.

## Quick start

Install a current LTS release of [Node.js](https://nodejs.org/) with npm and use a
machine with hardware WebGPU support. Run all commands from the repository root.

```sh
git clone https://github.com/Sunset-Studio/Sundown.git
cd Sundown
npm install
```

### Start your own application

```sh
npm run app:init -- --name my-game --dry-run
npm run app:init -- --name my-game
npm run dev
```

The dry run is an optional preview. Setup creates `app/app.js`, a lit cube and sky
in `app/main_scene.js`, a smoke catalog in `app/smoke.config.js`, and an
`assets/app/` directory for your assets. It switches `index.html` from the sample
app to your starter and adds `npm run smoke:app`. It keeps the samples by default
and refuses to overwrite existing starter paths or an existing `smoke:app` script.
`--name` sets the engine project name (default: `my-app`), not packaging metadata
or window titles. Start the name with a lowercase letter and use only lowercase
letters, digits, and hyphens.

Open the local URL printed by `npm run dev` to see the starter. To run it in
Electron instead, stop the browser dev server and run `npm run devtop`.

```sh
npm run smoke:app -- --list
npm run smoke:app
```

The first command lists `MainScene`. The second cooks assets, builds the app, and
runs that scene in Electron. A successful run exits with code `0` and writes
`diagnostics.json` with `status: "passed"`, ready assets, and at least five
completed GPU frames. Runtime warnings or errors fail the run. The starter
supports isolated smoke runs, not `--scene-switching`.

To omit the samples from your fork at setup time, preview with
`npm run app:init -- --name my-game --remove-example --dry-run`. Running without
`--dry-run` **deletes `example/` and `assets/example/`, including local changes**,
and removes the sample test scripts. If you've already initialized your app, don't
rerun setup; see the [full starter guide](example/README.md#start-your-own-application)
for customization and manual sample removal.

### Try the sample application

On a fresh checkout, skip `app:init` and run `npm run dev` for the browser or
`npm run devtop` for an Electron window. The default entry point is `example/app.js`.
To try the tray-first sample, run `npm run devtop -- --tray`, then click the tray
icon on macOS or Windows, or choose **Open panel** from the tray menu on Linux.
See the [desktop tray guide](.agents/docs/desktop-tray.md) to ship your own tray app.

```sh
npm run smoke:example -- --list
npm run smoke:example -- --scene-switching
```

The sample suite supports both isolated runs and scene switching in one Electron
process. See the [sample guide](example/README.md) for sample-specific commands and
the [smoke testing guide](.agents/docs/smoke.md) for custom catalogs, scene filters,
and report details.

## Features

- **WebGPU rendering:** GPU resource abstractions, a render graph for graphics and
  compute passes, a compute task queue, and a configurable post-processing stack.
- **GPU-driven geometry:** entity-based instancing and automatic draw batching,
  meshlet culling, a visibility-buffer deferred renderer, and mesh LOD generation.
- **Materials and assets:** built-in PBR shaders, custom shader materials,
  pseudo-bindless texture pools and material tables, glTF scene loading, and KTX2
  texture cooking and streaming.
- **Lighting and ray tracing:** Adaptive Sparse Virtual Shadow Maps (AS-VSM),
  Surface Cache GI (SCGI), probe-based Dynamic Diffuse GI (DDGI), and experimental
  baked GI with streamed SVLM tiles. A multi-bounce compute reference path tracer,
  dynamic scene TLAS, and per-mesh triangle BLAS support software ray tracing and
  ray casting.
- **Simulation and UI:** modular simulation layers, the TypedArray-backed Solar
  archetype/chunk ECS, context-based input, immediate-mode screen-space and
  world-space UI, and MSDF text.
- **Developer tools:** starter app scaffolding, GPU timing and memory views, BVH and
  mesh LOD inspection, and a WGSL formatter.
- **Smoke testing:** application-owned scene catalogs, isolated Electron runs or
  optional in-process scene switching, asset and GPU-frame readiness checks, and
  per-scene runtime diagnostics in `diagnostics.json`.
- **Desktop:** sandboxed Electron builds with standard windowed and optional
  tray-first launch modes.
- **Machine learning experiments:** layer-based subnet graphs with backpropagation,
  built-in activations and loss functions, the Adam optimizer, and `MasterMind`
  for coordinating live training and optional weight sharing across models.

## Showcase

![sundown_demo](https://github.com/user-attachments/assets/fb001dca-66dd-4ba5-b307-428cad52e441)

<img width="1587" height="883" alt="Screenshot 2025-11-18 140745" src="https://github.com/user-attachments/assets/c3083a5f-aec8-42ca-bcd1-c32702930670" />

<img width="1583" height="885" alt="Screenshot 2025-11-18 132838" src="https://github.com/user-attachments/assets/9131c374-b200-49c0-940a-ad52afdacda4" />

<img width="1582" height="897" alt="Screenshot 2025-11-18 133035" src="https://github.com/user-attachments/assets/30e32ad7-1714-4aa1-8e38-667f0eee3aab" />

https://github.com/user-attachments/assets/3671857c-6eab-422b-81d8-8d6095874c5a

https://github.com/user-attachments/assets/2925ccb8-4484-48e6-a049-143ed0b5c944

## Packaging

| Command | Output |
| --- | --- |
| `npm run build` | Web build in `dist/` |
| `npm run buildtop` | Unpacked Electron app in `executables/` |
| `npm run make` | Desktop packages in `executables/`: DMG and ZIP on macOS, NSIS installer on Windows, AppImage on Linux |

For web deployment, serve the entire contents of `dist/` over HTTPS and configure
the cross-origin isolation headers used in [`vite.config.mjs`](vite.config.mjs).

Desktop builds bundle [Electron](https://www.electronjs.org/) and use Chromium's
WebGPU implementation, not WKWebView. Sundown checks required GPU capabilities at
startup. The desktop renderer is sandboxed with Node integration disabled and uses
the same WebGPU API as the web build. Build on the target OS for distribution;
configure electron-builder for Apple code signing and notarization for public
macOS releases.

## Contributing

Sundown is available for free under the MIT license. You can use and modify the engine for individual or commercial use (a reference or mention is still appreciated!) If you want to contribute features or fixes, please fork this repository and submit PRs. I am a one man team but will check any promising PRs as soon as I can. If you want to become a regular contributor feel free to DM me on [X](https://x.com/SunsetLearn) or shoot me an email at adrians.sanchez@sunsetlearn.com.
