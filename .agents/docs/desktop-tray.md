# Desktop tray mode

Use `npm run devtop -- --tray` to try tray mode with the Vite development server.
After a web build, `npx electron . --tray` also works without a dev server. A
packaged executable accepts `--tray` as well. Set `enabled: true` in
[`desktop/config.js`](../../desktop/config.js) before `npm run buildtop` or
`npm run make` to make tray mode the default for your release. Regular desktop
launches remain windowed by default. Smoke launches always use their normal test
window, regardless of tray settings.

## Panel behavior

- No renderer starts until the panel is opened. On macOS the tray app does not
  keep a Dock icon; on Windows the panel does not get a taskbar button.
- On macOS and Windows, clicking the icon opens the panel near it, within that
  display's usable area. Right-click the icon for the native menu.
- On Linux, choose **Open panel** from the native tray menu. Linux needs a desktop
  tray host. Electron doesn't expose Linux tray-icon coordinates, so Sundown
  uses the pointer as an anchor on X11 and leaves placement to the compositor
  in a Wayland session.
- Clicking away or pressing Escape hides the panel without discarding its state.
  If the panel is closed, the tray stays active and the next open creates it again.
  The native menu's **Quit** action exits the app.

## Application integration

The panel loads the same application entry point as a regular desktop window,
with `?tray=1`. Check that parameter before starting your ordinary scene and
mount your panel instead. Width and height in `desktop/config.js` are
device-independent pixels. Set `icon` to an app-relative image path (if you
change the path, include the new file in `build.files` in `package.json`).
`open_label` and `quit_label` set the text of the native tray menu.

[`example/tray_panel.js`](../../example/tray_panel.js) draws with Sundown's
immediate-mode 2D widgets. It uses Canvas 2D and redraws on demand, without
starting a scene or requesting a WebGPU device. You can preview it in a browser
by adding `?tray=1` to the dev server URL. Your panel can also run a normal
Sundown `Simulator` if it needs WebGPU rendering; disable pointer lock for a
menu and avoid running background work when the panel is hidden. The panel
retains the desktop renderer's sandbox and has no tray-specific IPC or Node API.

Run `npm run test:tray` for popup placement tests. To verify interactions on a
target desktop, open the panel from the tray, use its button, dismiss it with
Escape and by clicking away, reopen to confirm state is retained, then quit
through the tray's native menu.
