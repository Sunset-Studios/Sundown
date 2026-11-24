import * as UI from "../engine/src/ui/2d/immediate.js";

const panel_style = {
  width: "100%",
  height: "100%",
  layout: "column",
  gap: 8,
  padding: 16,
  background_color: "#121923",
};
const title_style = {
  x: 0,
  y: 0,
  width: "100%",
  height: 30,
  font: "600 20px system-ui",
  text_color: "#f1f8ff",
  text_align: "left",
};
const caption_style = {
  x: 0,
  y: 0,
  width: "100%",
  height: 24,
  font: "13px system-ui",
  text_color: "#aabdd0",
  text_align: "left",
};
const button_style = {
  x: 0,
  y: 0,
  width: "100%",
  height: 40,
  font: "600 14px system-ui",
  background_color: "#243a53",
  hover_color: "#31577c",
  text_color: "#f1f8ff",
  corner_radius: 8,
};

// Sundown's 2D widgets can paint directly to a canvas without a Simulator.
export function mount_tray_panel() {
  document.getElementById("gpu-canvas").remove();
  const canvas = document.getElementById("ui-canvas");
  canvas.style.pointerEvents = "auto";
  canvas.tabIndex = 0;
  canvas.setAttribute("role", "button");
  canvas.setAttribute("aria-label", "Pause example activity");
  const ctx = canvas.getContext("2d");
  const input = UI.UIContext.input_state;
  let paused = false;
  let paint_request = null;

  function toggle_status() {
    paused = !paused;
    canvas.setAttribute("aria-pressed", String(paused));
    request_paint();
  }

  function paint() {
    paint_request = null;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const scale = window.devicePixelRatio;
    const pixel_width = Math.round(width * scale);
    const pixel_height = Math.round(height * scale);
    if (canvas.width !== pixel_width) canvas.width = pixel_width;
    if (canvas.height !== pixel_height) canvas.height = pixel_height;
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    UI.reset_ui(width, height);
    UI.panel(panel_style, draw_widgets);
    input.clicked = false;
    UI.flush_ui(ctx);
  }

  function draw_widgets() {
    UI.label("Sundown", title_style);
    UI.label(paused ? "Example activity is paused" : "Example activity is running", caption_style);
    const button = UI.button(paused ? "Resume activity" : "Pause activity", button_style);
    canvas.style.cursor = button.hovered ? "pointer" : "";
    if (button.clicked) toggle_status();
    UI.label("Esc to dismiss", caption_style);
  }

  function request_paint() {
    if (paint_request === null) paint_request = requestAnimationFrame(paint);
  }

  function update_pointer(event) {
    input.x = event.offsetX;
    input.y = event.offsetY;
  }

  function clear_pointer() {
    input.x = -1;
    input.y = -1;
    input.clicked = false;
    request_paint();
  }

  canvas.addEventListener("pointermove", (event) => {
    update_pointer(event);
    request_paint();
  });
  canvas.addEventListener("pointerleave", clear_pointer);
  canvas.addEventListener("click", (event) => {
    update_pointer(event);
    input.clicked = true;
    // Hit-test the click now; a later pointer event can update the same UI state.
    if (paint_request !== null) cancelAnimationFrame(paint_request);
    paint();
  });
  canvas.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    if (!event.repeat) toggle_status();
  });
  window.addEventListener("blur", clear_pointer);
  window.addEventListener("resize", request_paint);
  canvas.setAttribute("aria-pressed", "false");
  clear_pointer();
}
