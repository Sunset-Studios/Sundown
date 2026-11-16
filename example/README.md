# Sundown sample application

This directory owns the sample scene registry, scene-switching driver, settings
panel and sample-specific tests. It is optional; production forks can remove it.
The root `index.html` currently selects `example/app.js` as its application entry.

```sh
npm run smoke:example -- --list
npm run smoke:example -- --scene-switching
npm run smoke:example -- --scene-switching --scene BVHScene,UI3DTestScene
npm run test:example
```

`smoke.config.js` exposes the Node-safe `samples/registry.js` catalog to the generic
launcher. The registry's lazy imports keep browser-only scene modules out of Node.
`app.js` uses `SceneSwitcher` to activate the selected scenes in sequence mode.
These are application choices, not engine or desktop requirements.

To add a sample, export its scene class in `samples/` and register its durable ID
and lazy import in `samples/registry.js`. The exported class name must match the ID.
The registry supplies both interactive scene order and smoke selection. Shared
label rendering uses `world_label` from `engine/src/ui/3d/common.js`.

For a production application, replace the root HTML script entry with your own
entry point, provide a Node-safe smoke catalog (or explicit `--scene` IDs), and
wire the generic smoke helpers into your application's lifecycle. Remove the
`smoke:example` and `test:example` convenience scripts if this directory is removed.
Generic `test:smoke` and `test:smoke:electron` do not require the sample app.
