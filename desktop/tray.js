import { app, dialog, Menu, nativeImage, screen, Tray } from "electron";
import path from "node:path";
import { create_app_window } from "./window.js";

const icon_gap = 6;

function get_panel_placement(tray, config, linux, wayland) {
  if (wayland) {
    // Wayland does not expose the global cursor position or allow moving windows.
    const area = screen.getPrimaryDisplay().workArea;
    return {
      width: Math.min(config.width, area.width),
      height: Math.min(config.height, area.height),
    };
  }

  const icon_bounds = linux ? null : tray.getBounds();
  const anchor =
    icon_bounds?.width > 0 && icon_bounds.height > 0
      ? icon_bounds
      : { ...screen.getCursorScreenPoint(), width: 0, height: 0 };
  const display = anchor.width
    ? screen.getDisplayMatching(anchor)
    : screen.getDisplayNearestPoint(anchor);
  return get_tray_panel_bounds(anchor, display.workArea, config.width, config.height);
}

// All inputs and outputs are Electron device-independent pixels. Work areas can
// have negative origins when the tray is on a secondary display.
export function get_tray_panel_bounds(anchor, work_area, requested_width, requested_height) {
  const width = Math.min(requested_width, work_area.width);
  const height = Math.min(requested_height, work_area.height);
  const right = work_area.x + work_area.width;
  const bottom = work_area.y + work_area.height;
  const anchor_right = anchor.x + anchor.width;
  const anchor_bottom = anchor.y + anchor.height;

  let x = anchor.x + (anchor.width - width) / 2;
  let y;
  if (anchor_right <= work_area.x || anchor.x >= right) {
    // Side-mounted taskbars leave the nearest usable space beside the icon.
    x = anchor_right <= work_area.x ? anchor_right + icon_gap : anchor.x - width - icon_gap;
    y = anchor.y + (anchor.height - height) / 2;
  } else {
    const below = bottom - anchor_bottom - icon_gap;
    const above = anchor.y - work_area.y - icon_gap;
    y = below >= height || below >= above ? anchor_bottom + icon_gap : anchor.y - height - icon_gap;
  }

  return {
    x: Math.round(Math.max(work_area.x, Math.min(x, right - width))),
    y: Math.round(Math.max(work_area.y, Math.min(y, bottom - height))),
    width,
    height,
  };
}

async function open_panel(state) {
  if (state.loading) return;

  const { tray, config, linux, wayland, url } = state;
  const bounds = get_panel_placement(tray, config, linux, wayland);
  let panel = state.panel;

  if (!panel) {
    panel = create_app_window(url, {
      ...bounds,
      show: false,
      paintWhenInitiallyHidden: false,
      frame: false,
      resizable: false,
      fullscreenable: false,
      skipTaskbar: true,
      alwaysOnTop: true,
      backgroundColor: "#121923",
      autoHideMenuBar: true,
    });
    state.panel = panel;
    if (process.platform !== "darwin") panel.removeMenu();
    panel.on("closed", () => {
      state.panel = null;
    });
    panel.on("blur", () => panel.hide());
    panel.webContents.on("before-input-event", (event, input) => {
      if (input.type === "keyDown" && input.key === "Escape") {
        event.preventDefault();
        panel.hide();
      }
    });

    state.loading = true;
    try {
      await panel.loadURL(url);
    } catch (error) {
      if (panel.isDestroyed()) return;
      panel.destroy();
      throw error;
    } finally {
      state.loading = false;
    }
    if (panel.isDestroyed()) return;
  }

  if (!wayland) panel.setBounds(bounds);
  panel.show();
  panel.focus();
}

function request_open_panel(state) {
  open_panel(state).catch((error) => {
    console.error("Failed to open Sundown tray panel:", error);
    dialog.showErrorBox("Sundown could not open the tray panel", error.message);
  });
}

export function start_tray(app_url, config) {
  const panel_url = new URL(app_url);
  panel_url.searchParams.set("tray", "1");
  const url = panel_url.href;

  const icon_path = path.join(app.getAppPath(), config.icon);
  const image = nativeImage.createFromPath(icon_path);
  if (image.isEmpty()) throw new Error(`Unable to load the tray icon: ${icon_path}`);

  const linux = process.platform === "linux";
  const wayland = linux && Boolean(process.env.WAYLAND_DISPLAY);
  const icon = image.resize({ width: linux ? 22 : 16 });
  icon.addRepresentation({
    scaleFactor: 2,
    buffer: image.resize({ width: linux ? 44 : 32 }).toPNG(),
  });

  if (process.platform === "darwin") {
    icon.setTemplateImage(image.isTemplateImage());
    app.setActivationPolicy("accessory");
  }

  const tray = new Tray(icon);
  tray.setToolTip(app.getName());

  const state = { tray, config, linux, wayland, url, panel: null, loading: false };
  const open = request_open_panel.bind(null, state);
  const menu = Menu.buildFromTemplate([
    { label: config.open_label, click: open },
    { type: "separator" },
    { label: config.quit_label, click: () => app.quit() },
  ]);

  if (linux) {
    // StatusNotifier hosts may consume activation themselves instead of emitting click.
    tray.setContextMenu(menu);
  } else {
    tray.on("click", open);
    tray.on("right-click", () => {
      state.panel?.hide();
      tray.popUpContextMenu(menu, screen.getCursorScreenPoint());
    });
  }
  app.once("will-quit", () => tray.destroy());
  return tray;
}
