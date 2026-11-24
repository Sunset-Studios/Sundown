import assert from "node:assert/strict";
import test from "node:test";
import { get_tray_panel_bounds } from "../../desktop/tray.js";

const work_area = { x: 0, y: 24, width: 1440, height: 836 };

test("tray panel opens below a top menu bar and clamps to the right edge", () => {
  const icon = { x: 1380, y: 0, width: 24, height: 24 };
  assert.deepEqual(get_tray_panel_bounds(icon, work_area, 320, 220), {
    x: 1120,
    y: 30,
    width: 320,
    height: 220,
  });
});

test("tray panel opens above a bottom taskbar on a display with a negative origin", () => {
  const display = { x: -1280, y: -200, width: 1280, height: 760 };
  const icon = { x: -125, y: 560, width: 24, height: 24 };
  assert.deepEqual(get_tray_panel_bounds(icon, display, 320, 220), {
    x: -320,
    y: 334,
    width: 320,
    height: 220,
  });
});

test("tray panel opens beside a side taskbar", () => {
  const display = { x: 40, y: 0, width: 1240, height: 800 };
  const icon = { x: 8, y: 360, width: 24, height: 24 };
  assert.deepEqual(get_tray_panel_bounds(icon, display, 320, 220), {
    x: 40,
    y: 262,
    width: 320,
    height: 220,
  });
});

test("tray panel fits a small display even when only a cursor anchor is available", () => {
  const display = { x: 0, y: 0, width: 300, height: 200 };
  const cursor = { x: 260, y: 170, width: 0, height: 0 };
  assert.deepEqual(get_tray_panel_bounds(cursor, display, 320, 220), display);
});
