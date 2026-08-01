import { Renderer } from "../../renderer/renderer.js";
import { InputProvider } from "../../input/input_provider.js";
import { InputKey, InputRange, InputKeyToPrintableString } from "../../input/input_types.js";
import { FrameAllocator, FrameStackAllocator } from "../../memory/allocator.js";
import { profile_scope } from "../../utility/performance.js";
import { clamp } from "../../utility/math.js";
import {
  SliderMode,
  format_slider_value,
  resolve_slider_value,
  slider_ratio,
  update_slider_interaction,
} from "../slider.js";

export { SliderMode } from "../slider.js";

export const UI_Z_ORDER_CURSOR = 10000;

const corner_radius_top_left = "corner_radius_top_left";
const corner_radius_top_right = "corner_radius_top_right";
const corner_radius_bottom_right = "corner_radius_bottom_right";
const corner_radius_bottom_left = "corner_radius_bottom_left";
const left = "left";
const right = "right";
const top = "top";
const bottom = "bottom";
const center = "center";
const middle = "middle";
const width_name = "width";
const height_name = "height";
const absolute = "absolute";
const row = "row";
const column = "column";
const row_reversed = "row_reversed";
const column_reversed = "column_reversed";

const shift_char_map = {
  1: "!",
  2: "@",
  3: "#",
  4: "$",
  5: "%",
  6: "^",
  7: "&",
  8: "*",
  9: "(",
  0: ")",
  "-": "_",
  "=": "+",
  "[": "{",
  "]": "}",
  "\\": "|",
  ";": ":",
  "'": '"',
  ",": "<",
  ".": ">",
  "/": "?",
  "`": "~",
};

class LayoutStackContainer {
  x = 0;
  y = 0;
  width = 0;
  height = 0;
  layout = absolute;
  gap = 0;
  padding_left = 0;
  padding_top = 0;
  padding_right = 0;
  padding_bottom = 0;
  cursor = { x: 0, y: 0 };
  config = {};
  auto_width = false;
  auto_height = false;
  content_max_x = 0;
  content_max_y = 0;
  z_order = 0;
}

const KeyboardKey = {
  key: InputKey.K_None,
  first: false,
  held: false,
  consumed: false,
  last_change_time: 0,
};

/**
 * Immediate Mode UI Framework for Sundown
 *

 * This file replaces all persistent UI element classes with a per–frame,
 * functional immediate–mode API. Every frame the UI is rebuilt with a
 * series of function calls that push drawing commands (and perform hit–testing,
 * layout, drag–drop and events) into a per–frame buffer. At the end of the frame,
 * the commands are flushed to the canvas.
 *
 * The API supports:
 *   - Parenting & auto–layout via beginContainer() / endContainer()
 *   - Button widgets that support icons, text, drag–and–drop and callbacks
 *   - Label, Input (text field) and Cursor widgets
 *   - A Panel widget, which is simply a container.
 *
 * To use this system, external code must update the global input state and keyboard
 * events. (See the sample render loop in main.js for one example.)
 */

// ------------------------------
// Global UI Context & State
// ------------------------------

export const UIContext = {
  /**
   * Global draw commands array.
   */
  draw_commands: [],
  /**
   * Stable insertion order for draw commands. Used as a tie-breaker when z-order matches.
   */
  draw_command_order: 0,
  /**
   * Global layout allocator.
   */
  layout_allocator: new FrameAllocator(1024, LayoutStackContainer),
  /**
   * Global layout stack.
   */
  layout_stack: new FrameStackAllocator(100, 0), // TODO: Turn this into a TypedStack since it's just storing a single int per entry.
  /**
   * Global unique ID counter.
   */
  id_counter: 0,
  /**
   * Global delta time.
   */
  delta_time: 0,
  /**
   * Global drag state.
   */
  drag_state: {
    active: false,
    started: false,
    widget_id: null,
    timer: 0,
  },
  /**
   * Global input state must be updated externally by event handlers.
   * x, y are current mouse coordinates; pressed is true while the mouse button is down.
   * clicked is true for the frame in which the primary pointer action occurs.
   */
  input_state: {
    x: 0,
    y: 0,
    prev_x: 0,
    prev_y: 0,
    depth: 200,
    pressed: false,
    clicked: false,
    world_position: null, // assume this is a vec3 from a math library
    wheel: 0,
  },
  /**
   * Global scroll state.
   */
  scroll_state: {},
  /**
   * Pointer capture shared by 2D and 3D sliders.
   */
  slider_state: {
    active_widget_id: null,
    drag_start_x: 0,
    drag_start_value: 0,
  },
  /**
   * Global keyboard events array.

   * External code should append key events (i.e. key strings like "a", "Backspace", etc.)
   * at the beginning of a frame.
   */
  keyboard_events: new FrameAllocator(256, KeyboardKey),
  /**
   * Global cache for images (used by buttons and cursors to load icons).
   */
  image_cache: {},
  /**
   * Global state storage for input fields.
   * Keys are given by the input field's provided name.
   */
  input_field_state: {},
  /**
   * Global canvas size.
   */
  canvas_size: {
    width: 0,
    height: 0,
  },

  // Generate a unique widget id.
  get_unique_id() {
    return this.id_counter++;
  },

  // Consume a given key from the keyboard events array.
  consume_key(key) {
    for (let i = 0; i < this.keyboard_events.length; i++) {
      if (this.keyboard_events.get(i).key === key) {
        this.keyboard_events.splice(i, 1);
      }
    }
  },
  /**
   * Global measure context for text measurement.
   */
  measure_context: null,
  get_measure_context() {
    if (!this.measure_context) {
      this.measure_context = document.createElement("canvas").getContext("2d");
    }
    return this.measure_context;
  },
};

// ------------------------------
// Utility Functions
// ------------------------------

function parse_dimension(value, base = 0) {
  if (typeof value === "string") {
    if (value.endsWith("%")) {
      // Convert percentage to pixels using the given base.
      return parseFloat(value) * 0.01 * base;
    }
    return parseFloat(value);
  }
  return Number(value) || 0;
}

function resolve_z_order(config = {}, inherited_z_order = 0) {
  const configured_z_order = config.z_order ?? config.z_index ?? inherited_z_order;
  const z_order = Number(configured_z_order);
  return Number.isFinite(z_order) ? z_order : inherited_z_order;
}

function get_current_z_order() {
  const container_index = UIContext.layout_stack.peek();
  if (container_index === null) {
    return 0;
  }

  const container = UIContext.layout_allocator.get(container_index.value);
  return container?.z_order ?? 0;
}

function push_draw_command(draw, config = {}, inherited_z_order = get_current_z_order()) {
  const command = {
    draw,
    z_order: resolve_z_order(config, inherited_z_order),
    order: UIContext.draw_command_order++,
  };
  UIContext.draw_commands.push(command);
  return UIContext.draw_commands.length - 1;
}

function set_draw_command(index, draw, config = {}, inherited_z_order = get_current_z_order()) {
  const existing_command = UIContext.draw_commands[index];
  UIContext.draw_commands[index] = {
    draw,
    z_order: existing_command?.z_order ?? resolve_z_order(config, inherited_z_order),
    order: existing_command?.order ?? UIContext.draw_command_order++,
  };
}

function rounded_rect_path_corners(
  ctx,
  x,
  y,
  width,
  height,
  radius_top_left,
  radius_top_right,
  radius_bottom_right,
  radius_bottom_left
) {
  // Clamp each radius so it doesn't exceed half the min(width, height)
  const max_radius = Math.min(width, height) / 2;
  radius_top_left = Math.min(radius_top_left, max_radius);
  radius_top_right = Math.min(radius_top_right, max_radius);
  radius_bottom_right = Math.min(radius_bottom_right, max_radius);
  radius_bottom_left = Math.min(radius_bottom_left, max_radius);

  ctx.moveTo(x + radius_top_left, y);
  ctx.lineTo(x + width - radius_top_right, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius_top_right);
  ctx.lineTo(x + width, y + height - radius_bottom_right);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius_bottom_right, y + height);
  ctx.lineTo(x + radius_bottom_left, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius_bottom_left);
  ctx.lineTo(x, y + radius_top_left);
  ctx.quadraticCurveTo(x, y, x + radius_top_left, y);
  ctx.closePath();
}

function rounded_rect_path(ctx, x, y, width, height, radius) {
  // Ensure the radius does not exceed half of width or height
  radius = Math.min(radius, width / 2, height / 2);
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

function base_draw(ctx, x, y, width, height, config) {
  // Check if any individual corner options are provided.
  const has_per_corner =
    corner_radius_top_left in config ||
    corner_radius_top_right in config ||
    corner_radius_bottom_right in config ||
    corner_radius_bottom_left in config;

  // Compute the effective corner radii.
  let should_round = false;
  let tl = 0,
    tr = 0,
    br = 0,
    bl = 0;
  if (has_per_corner) {
    tl =
      config.corner_radius_top_left !== undefined
        ? config.corner_radius_top_left
        : config.corner_radius || 0;
    tr =
      config.corner_radius_top_right !== undefined
        ? config.corner_radius_top_right
        : config.corner_radius || 0;
    br =
      config.corner_radius_bottom_right !== undefined
        ? config.corner_radius_bottom_right
        : config.corner_radius || 0;
    bl =
      config.corner_radius_bottom_left !== undefined
        ? config.corner_radius_bottom_left
        : config.corner_radius || 0;
    should_round = tl > 0 || tr > 0 || br > 0 || bl > 0;
  } else {
    // Use the older global setting.
    let radius = config.corner_radius || 0;
    should_round = radius > 0;
  }

  // Create the clipping region.
  ctx.beginPath();
  if (has_per_corner && should_round) {
    rounded_rect_path_corners(ctx, x, y, width, height, tl, tr, br, bl);
  } else if (should_round) {
    // Use the single global radius if no per-corner values are provided.
    rounded_rect_path(ctx, x, y, width, height, config.corner_radius);
  } else {
    ctx.rect(x, y, width, height);
  }
  ctx.clip();

  // Draw background using the appropriate shape.
  if (config.background_color) {
    ctx.fillStyle = config.background_color || "#ccc";
    ctx.beginPath();
    if (has_per_corner && should_round) {
      rounded_rect_path_corners(ctx, x, y, width, height, tl, tr, br, bl);
    } else if (should_round) {
      rounded_rect_path(ctx, x, y, width, height, config.corner_radius);
    } else {
      ctx.rect(x, y, width, height);
    }
    ctx.fill();
  }

  // Draw the border if provided.
  if (config.border_top || config.border_right || config.border_bottom || config.border_left) {
    // -------------------------------
    // Draw individual borders if specified.
    // -------------------------------
    // Top border
    if (config.border_top) {
      const parts = config.border_top.split(" ");
      const border_width_top = parseFloat(parts[0]);
      const border_color_top = parts[2] || "#000";
      ctx.strokeStyle = border_color_top;
      ctx.lineWidth = border_width_top;
      ctx.beginPath();
      if (has_per_corner && should_round) {
        const tl_radius =
          config.corner_radius_top_left !== undefined
            ? config.corner_radius_top_left
            : config.corner_radius || 0;
        const tr_radius =
          config.corner_radius_top_right !== undefined
            ? config.corner_radius_top_right
            : config.corner_radius || 0;
        ctx.moveTo(x + tl_radius, y);
        ctx.lineTo(x + width - tr_radius, y);
      } else {
        ctx.moveTo(x, y);
        ctx.lineTo(x + width, y);
      }
      ctx.stroke();
    }

    // Right border
    if (config.border_right) {
      const parts = config.border_right.split(" ");
      const border_width_right = parseFloat(parts[0]);
      const border_color_right = parts[2] || "#000";
      ctx.strokeStyle = border_color_right;
      ctx.lineWidth = border_width_right;
      ctx.beginPath();
      if (has_per_corner && should_round) {
        const tr_radius =
          config.corner_radius_top_right !== undefined
            ? config.corner_radius_top_right
            : config.corner_radius || 0;
        const br_radius =
          config.corner_radius_bottom_right !== undefined
            ? config.corner_radius_bottom_right
            : config.corner_radius || 0;
        ctx.moveTo(x + width, y + tr_radius);
        ctx.lineTo(x + width, y + height - br_radius);
      } else {
        ctx.moveTo(x + width, y);
        ctx.lineTo(x + width, y + height);
      }
      ctx.stroke();
    }

    // Bottom border
    if (config.border_bottom) {
      const parts = config.border_bottom.split(" ");
      const border_width_bottom = parseFloat(parts[0]);
      const border_color_bottom = parts[2] || "#000";
      ctx.strokeStyle = border_color_bottom;
      ctx.lineWidth = border_width_bottom;
      ctx.beginPath();
      if (has_per_corner && should_round) {
        const br_radius =
          config.corner_radius_bottom_right !== undefined
            ? config.corner_radius_bottom_right
            : config.corner_radius || 0;
        const bl_radius =
          config.corner_radius_bottom_left !== undefined
            ? config.corner_radius_bottom_left
            : config.corner_radius || 0;
        ctx.moveTo(x + width - br_radius, y + height);
        ctx.lineTo(x + bl_radius, y + height);
      } else {
        ctx.moveTo(x, y + height);
        ctx.lineTo(x + width, y + height);
      }
      ctx.stroke();
    }

    // Left border
    if (config.border_left) {
      const parts = config.border_left.split(" ");
      const border_width_left = parseFloat(parts[0]);
      const border_color_left = parts[2] || "#000";
      ctx.strokeStyle = border_color_left;
      ctx.lineWidth = border_width_left;
      ctx.beginPath();
      if (has_per_corner && should_round) {
        const tl_radius =
          config.corner_radius_top_left !== undefined
            ? config.corner_radius_top_left
            : config.corner_radius || 0;
        const bl_radius =
          config.corner_radius_bottom_left !== undefined
            ? config.corner_radius_bottom_left
            : config.corner_radius || 0;
        ctx.moveTo(x, y + height - bl_radius);
        ctx.lineTo(x, y + tl_radius);
      } else {
        ctx.moveTo(x, y);
        ctx.lineTo(x, y + height);
      }
      ctx.stroke();
    }
  } else if (config.border) {
    // -------------------------------
    // Global border (fallback to the existing implementation).
    // -------------------------------
    const parts = config.border.split(" ");
    const border_width = parseFloat(parts[0]);
    const border_color = parts[2] || "#000";
    ctx.strokeStyle = border_color;
    ctx.lineWidth = border_width;
    ctx.beginPath();
    if (has_per_corner && should_round) {
      rounded_rect_path_corners(ctx, x, y, width, height, tl, tr, br, bl);
    } else if (should_round) {
      rounded_rect_path(ctx, x, y, width, height, config.corner_radius);
    } else {
      ctx.rect(x, y, width, height);
    }
    ctx.stroke();
  }

  if (config.box_shadow) {
    ctx.save();
    const parts = config.box_shadow.split(" ");
    ctx.shadowOffsetX = parse_dimension(parts[0], 10);
    ctx.shadowOffsetY = parse_dimension(parts[1], 10);
    ctx.shadowBlur = parse_dimension(parts[2], 10);
    ctx.shadowColor = parts[3] || "rgba(0, 0, 0, 0.5)";
    ctx.fillStyle = "none";
    ctx.fill();
    ctx.restore();
  }

  if (config.filter) {
    ctx.filter = config.filter;
  }
}

function is_input_within(x, y, width, height) {
  return (
    UIContext.input_state.x >= x &&
    UIContext.input_state.x <= x + width &&
    UIContext.input_state.y >= y &&
    UIContext.input_state.y <= y + height
  );
}

function child_container_layout_update(container, x, y, width, height) {
  // Auto–layout: update container cursor only for non–anchored positioning.
  if (container.layout !== absolute) {
    if (container.layout === row) {
      container.cursor.x += width + container.gap;
    } else if (container.layout === row_reversed) {
      container.cursor.x -= width + container.gap;
    } else if (container.layout === column) {
      container.cursor.y += height + container.gap;
    } else if (container.layout === column_reversed) {
      container.cursor.y -= height + container.gap;
    }

    // ---------- Auto–size update ----------
    // Now that the container's cursor has been advanced,
    // use its new position to update the content sizing.
    if (container.auto_width) {
      if (container.layout === row_reversed) {
        // For reversed row, track the minimum (leftmost) x position
        container.content_max_x = Math.max(
          container.content_max_x,
          container.x + container.width + container.padding_right - container.cursor.x
        );
      } else {
        container.content_max_x = Math.max(
          container.content_max_x,
          container.cursor.x - (container.x + container.padding_left + container.gap)
        );
      }
    }
    if (container.auto_height) {
      if (container.layout === column_reversed) {
        // For reversed column, track the minimum (topmost) y position
        container.content_max_y = Math.max(
          container.content_max_y,
          container.y + container.height + container.padding_bottom - container.cursor.y
        );
      } else {
        container.content_max_y = Math.max(
          container.content_max_y,
          container.cursor.y - (container.y + container.padding_top + container.gap)
        );
      }
    }
    // ---------- End auto–size update ----------
  } else {
    // For absolute positioning, use the original widget bounds.
    if (container.auto_width) {
      const rel_x = x - (container.x + container.padding_left);
      container.content_max_x = Math.max(container.content_max_x, rel_x + width);
    }
    if (container.auto_height) {
      const rel_y = y - (container.y + container.padding_top);
      container.content_max_y = Math.max(container.content_max_y, rel_y + height);
    }
  }
}

function element_handle_input(x, y, width, height, config, container) {
  // If the parent container is scrollable, adjust the widget's y coordinate
  // to account for the scrolling translation applied during drawing.
  if (container && container.config && container.config.scrollable) {
    y += container.scroll_offset || 0;
  }

  // Hit testing against current input state.
  const hovered = is_input_within(x, y, width, height);
  const clicked = hovered && UIContext.input_state.clicked;
  const pressed = hovered && UIContext.input_state.pressed;
  let dragged = false;

  if (
    !UIContext.drag_state.active &&
    UIContext.slider_state.active_widget_id === null &&
    (clicked || pressed)
  ) {
    InputProvider.consume_action(InputKey.B_mouse_left);
  }

  // ----------------------
  // Drag-and-Drop Handling
  // ----------------------
  if (config.draggable) {
    if (
      !UIContext.drag_state.active &&
      UIContext.slider_state.active_widget_id === null &&
      hovered &&
      UIContext.input_state.pressed
    ) {
      UIContext.drag_state.active = true;
      UIContext.drag_state.widget_id = config.widget_id;
    }

    if (
      UIContext.drag_state.active &&
      !UIContext.drag_state.started &&
      UIContext.drag_state.timer > (config.drag_delay || 0) &&
      UIContext.drag_state.widget_id === config.widget_id
    ) {
      if (config.on_drag_start) config.on_drag_start();
      UIContext.drag_state.started = true;
    }

    if (UIContext.drag_state.active && UIContext.drag_state.started) {
      if (UIContext.drag_state.widget_id === config.widget_id) {
        const offset_x = UIContext.input_state.prev_x - UIContext.input_state.x;
        const offset_y = UIContext.input_state.y - UIContext.input_state.prev_y;
        if (UIContext.input_state.pressed) {
          if (config.on_drag) config.on_drag(offset_x, offset_y);
          dragged = true;
        } else {
          if (config.on_drop) config.on_drop(x, y);
        }
        InputProvider.consume_action(InputKey.B_mouse_left);
      } else if (UIContext.input_state.pressed && hovered) {
        if (config.on_drag_over) config.on_drag_over(x, y, width, height);
      }
    }
  }

  // ----------------------
  // Scroll Handling Implementation
  // ----------------------
  if (config.scrollable && UIContext.input_state.wheel !== 0) {
    let scroll_delta = -UIContext.input_state.wheel * (config.scroll_speed || 20);
    let scroll_id = config.widget_id;
    let current_scroll = UIContext.scroll_state[scroll_id] || 0;
    let new_scroll = current_scroll + scroll_delta;
    if (container) {
      const extra_scroll = Math.max(0, container.content_max_y - container.height);
      new_scroll = Math.max(new_scroll, -extra_scroll);
      new_scroll = Math.min(new_scroll, 0);
    }

    if (scroll_id) {
      UIContext.scroll_state[scroll_id] = new_scroll;
    } else {
      config.scroll_offset = new_scroll;
    }
    InputProvider.consume_range(InputRange.M_wheel);
  }

  return { hovered, clicked, pressed, dragged, widget_id: config.widget_id };
}

function wrap_text(text, font, max_width) {
  const words = text.split(/\s+|_/);
  let lines = [];
  let current_line = words[0];

  const measure_context = UIContext.get_measure_context();
  measure_context.font = font;
  for (let i = 1; i < words.length; i++) {
    const word = words[i];
    const test_line = current_line + " " + word;
    if (measure_context.measureText(test_line).width > max_width) {
      lines.push(current_line);
      current_line = word;
    } else {
      current_line = test_line;
    }
  }
  lines.push(current_line);
  return lines;
}

// ------------------------------
// Reset & Flush
// ------------------------------

/**
 * Call at the start of every frame to clear the previous UI state.
 */
export function reset_ui(canvas_width = 0, canvas_height = 0) {
  UIContext.draw_commands = [];
  UIContext.draw_command_order = 0;
  UIContext.layout_allocator.reset();
  UIContext.layout_stack.reset();
  UIContext.keyboard_events.reset();
  UIContext.id_counter = 0;
  UIContext.canvas_size.width = canvas_width;
  UIContext.canvas_size.height = canvas_height;
}

/**
 * Call after all immediate UI calls in the frame to execute the drawing commands.
 * @param {CanvasRenderingContext2D} ctx - The canvas 2D context.
 */
export function flush_ui(ctx) {
  profile_scope("flush_ui", () => {
    UIContext.draw_commands.sort((a, b) => {
      const z_delta = (a?.z_order ?? 0) - (b?.z_order ?? 0);
      return z_delta !== 0 ? z_delta : (a?.order ?? 0) - (b?.order ?? 0);
    });

    for (let i = 0; i < UIContext.draw_commands.length; i++) {
      const command = UIContext.draw_commands[i];
      if (typeof command === "function") {
        command(ctx);
      } else if (typeof command?.draw === "function") {
        command.draw(ctx);
      }
    }
    UIContext.draw_commands.length = 0;

    const renderer = Renderer.get();
    reset_ui(renderer.canvas_ui?.width ?? 0, renderer.canvas_ui?.height ?? 0);
  });
}

// ------------------------------
// Parenting & Layout
// ------------------------------

/**
 * Begin a new container. A container defines an area (with position, width, height)
 * plus a layout mode (e.g., "absolute", "row", "column"). Once begun its context
 * is pushed on the layout stack; all child widget calls will use that context.
 *
 * @param {object} config - Container configuration:
 *    x, y: offset relative to parent (default 0)
 *    width, height: of the container (default 100)
 *    layout: "absolute", "row", "column", "row_reversed", "column_reversed", or "absolute"
 *    gap: gap between children in auto–layout (default 0)
 *    padding: inner padding (default 0)
 *    (Other style settings such as background_color, border, etc. may also be provided.)
 */
export function begin_container(config = {}) {
  config.widget_id = UIContext.get_unique_id();

  let parent_index = UIContext.layout_stack.peek();
  let parent =
    parent_index !== null
      ? UIContext.layout_allocator.get(parent_index.value)
      : {
          x: 0,
          y: 0,
          cursor: { x: 0, y: 0 },
          width: UIContext.canvas_size.width,
          height: UIContext.canvas_size.height,
          z_order: 0,
        };

  // Determine auto-sizing flags for each dimension.
  const auto_width = !(width_name in config);
  const auto_height = !(height_name in config);

  // Compute width and height.
  // If auto sizing is enabled for a dimension, start with 0.
  const width = auto_width ? 0 : parse_dimension(config.width, parent.width);
  const height = auto_height ? 0 : parse_dimension(config.height, parent.height);

  // Determine offsets using auto-sizing flags.
  let offset_x;
  if (config.x !== undefined) {
    offset_x = parse_dimension(config.x, parent.width);
  } else {
    offset_x = parent.auto_width ? 0 : (parent.width - width) / 2;
  }

  let offset_y;
  if (config.y !== undefined) {
    offset_y = parse_dimension(config.y, parent.height);
  } else {
    offset_y = parent.auto_height ? 0 : (parent.height - height) / 2;
  }

  let x;
  if (config.anchor_x === right) {
    x = parent.x + parent.width - offset_x - width;
  } else {
    x = parent.cursor.x + offset_x;
  }

  let y;
  if (config.anchor_y === bottom) {
    y = parent.y + parent.height - offset_y - height;
  } else {
    y = parent.cursor.y + offset_y;
  }

  let gap = parse_dimension(config.gap || 0, parent.width);
  let layout = config.layout || absolute; // "row", "column", "row_reversed", "column_reversed", or "absolute"

  // Parse padding values.
  const padding_left = config.padding_left || config.padding || 0;
  const padding_top = config.padding_top || config.padding || 0;
  const padding_right = config.padding_right || config.padding || 0;
  const padding_bottom = config.padding_bottom || config.padding || 0;

  // Push a new container onto the layout stack.
  const container_index = UIContext.layout_stack.push();
  container_index.value = UIContext.layout_allocator.length;

  const container = UIContext.layout_allocator.allocate();
  container.x = x;
  container.y = y;
  // The content area size (without padding) is based on the explicit sizes (if any).
  container.width = auto_width ? 0 : width - padding_left - padding_right;
  container.height = auto_height ? 0 : height - padding_top - padding_bottom;
  container.layout = layout;
  container.gap = gap;
  container.padding_left = padding_left;
  container.padding_top = padding_top;
  container.padding_right = padding_right;
  container.padding_bottom = padding_bottom;

  // Set initial cursor position based on layout type
  if (layout === row_reversed) {
    // For reversed row, start from the right side
    container.cursor = {
      x: x + width - padding_right,
      y: y + padding_top,
    };
  } else if (layout === column_reversed) {
    // For reversed column, start from the bottom
    container.cursor = {
      x: x + padding_left,
      y: y + padding_bottom,
    };
  } else {
    // Default cursor position (top-left corner)
    container.cursor = {
      x: x + padding_left,
      y: y + padding_top,
    };
  }

  container.config = config;
  container.z_order = resolve_z_order(config, parent.z_order ?? 0);

  container._has_clip = false;
  // Set auto-sizing flags and prepare to track children extents.
  container.auto_width = auto_width;
  container.auto_height = auto_height;
  // For auto sizing, we track the maximum extent relative to the container's content origin.
  // (For simplicity we assume children are not placed with negative offsets.)
  if (container.auto_width) {
    container.content_max_x = 0;
  }
  if (container.auto_height) {
    container.content_max_y = 0;
  }

  // Record a background draw command for the container.
  // Note that we reference container properties so that if they later change due to auto-sizing,
  // the final drawn background will use the updated dimensions.
  container._base_command_index = UIContext.draw_commands.length;
  push_draw_command(null, config, container.z_order);

  // --- Begin Clipping (for "clip") ---
  // If requested, add a clip region so that children drawn afterward will be clipped
  // to this container's bounds.
  if (config.clip) {
    // Mark that this container activated a clip.
    container._has_clip = true;
  }
  // --- End Clipping (for "clip") ---

  // --- Begin Scroll Translation (for scrollable containers) ---
  if (config.scrollable) {
    // Determine a persistent scroll offset.
    let scroll_id = config.widget_id;
    if (scroll_id) {
      if (!(scroll_id in UIContext.scroll_state)) {
        UIContext.scroll_state[scroll_id] = 0;
      }
      container.scroll_offset = UIContext.scroll_state[scroll_id];
    } else {
      container.scroll_offset = config.scroll_offset || 0;
    }

    // Push a translation draw command so that children are drawn shifted.
    push_draw_command(
      (ctx) => {
        ctx.save();
        ctx.translate(0, container.scroll_offset);
      },
      config,
      container.z_order
    );
  }
  // --- End Scroll Translation ---
}

/**
 * End the current container.
 */
export function end_container() {
  // Before popping the container, update its size if auto-sizing is enabled.
  const container_index = UIContext.layout_stack.peek();
  const container = UIContext.layout_allocator.get(container_index.value);

  if (container.auto_width) {
    container.width = container.content_max_x;
  } else if (container.config.scrollable) {
    // When width is fixed but the container is scrollable,
    // calculate the content_max_x from the current cursor.
    // (This assumes that container.cursor.x represents the maximum X coordinate reached by its children.)
    if (container.layout === row_reversed) {
      container.content_max_x =
        container.x + container.width + container.padding_right - container.cursor.x;
    } else {
      container.content_max_x = container.cursor.x - (container.x + container.padding_left);
    }
  }

  if (container.auto_height) {
    container.height = container.content_max_y;
  } else if (container.config.scrollable) {
    // When height is fixed but the container is scrollable,
    // calculate the content_max_y from the current cursor.
    // (This assumes that container.cursor.y represents the maximum Y coordinate reached by its children.)
    if (container.layout === column_reversed) {
      container.content_max_y =
        container.y + container.height + container.padding_bottom - container.cursor.y;
    } else {
      container.content_max_y = container.cursor.y - (container.y + container.padding_top);
    }
  }

  // --- Push base draw command for container ---
  set_draw_command(
    container._base_command_index,
    (ctx) => {
      ctx.save();
      let total_width = container.width + container.padding_left + container.padding_right;
      let total_height = container.height + container.padding_top + container.padding_bottom;
      base_draw(ctx, container.x, container.y, total_width, total_height, container.config);

      if (!container._has_clip) {
        ctx.restore();
      }
    },
    container.config,
    container.z_order
  );

  // Remove the current container from the layout stack.
  UIContext.layout_stack.pop();

  // If there is a parent container, treat the ended container as a widget
  // and update the parent's layout cursor and auto-size bounds accordingly.
  const parent_index = UIContext.layout_stack.peek();
  if (parent_index !== null) {
    const parent = UIContext.layout_allocator.get(parent_index.value);
    if (parent.layout !== absolute) {
      if (parent.layout === row) {
        parent.cursor.x += container.width + parent.gap;
        if (parent.auto_width) {
          const rel_x = container.x - (parent.x + parent.padding_left);
          parent.content_max_x = Math.max(
            parent.content_max_x,
            rel_x + container.width + container.padding_left + container.padding_right
          );
        }
      } else if (parent.layout === row_reversed) {
        parent.cursor.x -= container.width + parent.gap;
        if (parent.auto_width) {
          const rel_x =
            parent.x +
            parent.width +
            parent.padding_right -
            (container.x + container.width + container.padding_right);
          parent.content_max_x = Math.max(
            parent.content_max_x,
            rel_x + container.width + container.padding_left + container.padding_right
          );
        }
      } else if (parent.layout === column) {
        parent.cursor.y += container.height + parent.gap;
        if (parent.auto_height) {
          const rel_y = container.y - (parent.y + parent.padding_top);
          parent.content_max_y = Math.max(
            parent.content_max_y,
            rel_y + container.height + container.padding_top + container.padding_bottom
          );
        }
      } else if (parent.layout === column_reversed) {
        parent.cursor.y -= container.height + parent.gap;
        if (parent.auto_height) {
          const rel_y =
            parent.y +
            parent.height +
            parent.padding_bottom -
            (container.y + container.height + container.padding_bottom);
          parent.content_max_y = Math.max(
            parent.content_max_y,
            rel_y + container.height + container.padding_top + container.padding_bottom
          );
        }
      }
    }
  }

  // --- End Scroll Translation for scrollable containers ---
  if (container.config.scrollable) {
    push_draw_command(
      (ctx) => {
        ctx.restore();
      },
      container.config,
      container.z_order
    );
  }
  // --- End Clipping for "clip" ---
  // If this container activated a clip, push draw command to restore the context.
  if (container._has_clip) {
    push_draw_command(
      (ctx) => {
        ctx.restore();
      },
      container.config,
      container.z_order
    );
  }

  return element_handle_input(
    container.x,
    container.y,
    container.width,
    container.height,
    container.config,
    container
  );
}

/**
 * A simple panel is just a container with an optional visual style.
 * This helper wraps begin_container()/end_container() around a callback.
 *
 * @param {object} config - Same as begin_container().
 * @param {function} callback - A function in which you call child widget functions.
 */
export function panel(config, callback) {
  begin_container(config);
  if (callback) callback();
  return end_container();
}

// ------------------------------
// Widget Primitives
// ------------------------------

/**
 * Button widget.
 *
 * The button supports:
 *   - Icon drawing (using config.icon)
 *   - Text drawing (using config.text or the label parameter)
 *   - Hit–testing and drag–and–drop (if config.draggable is true)
 *   - Event callbacks (onClick, onDragStart, onDrag, onDrop)
 *
 * @param {string} label - The default label text.
 * @param {object} config - Widget config overrides:
 *   width, height, offset_x, offset_y,
 *   background_color, hover_color, border, text_color, font, etc.
 *   icon: URL for an icon image.
 *   draggable: boolean flag.
 *   on_drag_start / on_drag / on_drop: drag callbacks.
 *
 * @returns {object} An object containing { hovered, clicked, widgetId }.
 */
export function button(label, config = {}) {
  config.widget_id = UIContext.get_unique_id();

  // Get the current container (or default)
  const container_index = UIContext.layout_stack.peek();
  const container =
    container_index !== null
      ? UIContext.layout_allocator.get(container_index.value)
      : {
          x: 0,
          y: 0,
          cursor: { x: 0, y: 0 },
          width: UIContext.canvas_size.width,
          height: UIContext.canvas_size.height,
          z_order: 0,
        };

  const font = config.font || "16px sans-serif";
  const text_padding = config.text_padding || 0;

  // ------------------------------
  // Compute Dimensions: fit-content vs. fixed
  // ------------------------------
  let width, height;

  if (config.width === "fit-content") {
    // Use the shared measurement context for text measurement.
    const mc = UIContext.get_measure_context();
    mc.font = font;
    const metrics = mc.measureText(label);
    width = metrics.width + text_padding * 2;
  } else {
    width = parse_dimension(config.width, container.width);
  }

  let lines = [label];
  if (config.wrap) {
    const max_text_width = width - text_padding * 2;
    lines = wrap_text(label, font, max_text_width);
  }

  if (config.height === "fit-content") {
    // There is no built–in text height measurement; extract font size and use a heuristic.
    const font_match = font.match(/(\d+)px/);
    const font_size = font_match ? parseInt(font_match[1], 10) : 16;
    height = (config.wrap ? lines.length : 1) * font_size * 1.2;
  } else {
    height = parse_dimension(config.height, container.height);
  }

  // Determine offsets using auto-sizing flags.
  let offset_x;
  if (config.x !== undefined) {
    offset_x = parse_dimension(config.x, container.width);
  } else {
    offset_x = container.auto_width ? 0 : (container.width - width) / 2;
  }

  let offset_y;
  if (config.y !== undefined) {
    offset_y = parse_dimension(config.y, container.height);
  } else {
    offset_y = container.auto_height ? 0 : (container.height - height) / 2;
  }

  let x;
  if (config.anchor_x === right) {
    x = container.x + container.width - offset_x - width;
  } else {
    x = container.cursor.x + offset_x;
  }

  let y;
  if (config.anchor_y === bottom) {
    y = container.y + container.height - offset_y - height;
  } else {
    y = container.cursor.y + offset_y;
  }

  child_container_layout_update(container, x, y, width, height);

  // ----------------------
  // Assemble the Draw Command
  // ----------------------
  push_draw_command(
    (ctx) => {
      ctx.save();
      base_draw(ctx, x, y, width, height, config);

      // Button content: render icon if provided; fall back to text otherwise.
      if (config.icon) {
        if (!UIContext.image_cache[config.icon]) {
          const img = new Image();
          img.src = config.icon;
          UIContext.image_cache[config.icon] = img;
        }
        const img = UIContext.image_cache[config.icon];
        if (img.complete) {
          const icon_width = width * 0.8;
          const icon_height = height * 0.8;
          const icon_x = x + (width - icon_width) / 2;
          const icon_y = y + (height - icon_height) / 2;
          ctx.drawImage(img, icon_x, icon_y, icon_width, icon_height);
        } else if (config.text || label) {
          // Fallback to text drawing with alignment properties.
          ctx.fillStyle = config.text_color || "#fff";
          ctx.font = font;
          const text_align = config.text_align || center;
          const text_valign = config.text_valign || middle;
          const text_padding = config.text_padding || 10;
          let text_x, text_y;

          if (text_align === left) {
            text_x = x + text_padding;
          } else if (text_align === right) {
            text_x = x + width - text_padding;
          } else {
            text_x = x + width / 2;
          }

          if (config.wrap) {
            const font_size_match = font.match(/(\d+)px/);
            const font_size = font_size_match ? parseInt(font_size_match[1], 10) : 16;
            const line_height = font_size * 1.2;
            const total_text_height = lines.length * line_height;

            let start_y;
            if (text_valign === top) {
              start_y = y + text_padding + line_height;
            } else if (text_valign === bottom) {
              start_y = y + height - total_text_height + line_height;
            } else {
              start_y = y + (height - total_text_height) / 2 + line_height;
            }

            for (let i = 0; i < lines.length; i++) {
              ctx.fillText(lines[i], text_x, start_y + i * line_height);
            }
          } else {
            if (text_valign === top) {
              text_y = y + text_padding;
            } else if (text_valign === bottom) {
              text_y = y + height - text_padding;
            } else {
              text_y = y + height / 2;
            }
            ctx.textAlign = text_align;
            ctx.textBaseline = text_valign === center ? middle : text_valign;
            ctx.fillText(config.text || label, text_x, text_y);
          }
        }
      } else {
        if (config.text || label) {
          // Fallback to text drawing with alignment properties.
          ctx.fillStyle = config.text_color || "#fff";
          ctx.font = font;
          const text_align = config.text_align || center;
          const text_valign = config.text_valign || middle;
          const text_padding = config.text_padding || 10;
          let text_x, text_y;

          if (text_align === left) {
            text_x = x + text_padding;
          } else if (text_align === right) {
            text_x = x + width - text_padding;
          } else {
            text_x = x + width / 2;
          }

          if (config.wrap) {
            const font_size_match = font.match(/(\d+)px/);
            const font_size = font_size_match ? parseInt(font_size_match[1], 10) : 16;
            const line_height = font_size * 1.2;
            const total_text_height = lines.length * line_height;

            let start_y;
            if (text_valign === top) {
              start_y = y + text_padding + line_height;
            } else if (text_valign === bottom) {
              start_y = y + height - total_text_height + line_height;
            } else {
              start_y = y + (height - total_text_height) / 2 + line_height;
            }

            for (let i = 0; i < lines.length; i++) {
              ctx.fillText(lines[i], text_x, start_y + i * line_height);
            }
          } else {
            if (text_valign === top) {
              text_y = y + text_padding;
            } else if (text_valign === bottom) {
              text_y = y + height - text_padding;
            } else {
              text_y = y + height / 2;
            }
            ctx.textAlign = text_align;
            ctx.textBaseline = text_valign === center ? middle : text_valign;
            ctx.fillText(config.text || label, text_x, text_y);
          }
        }
      }
      ctx.restore();
    },
    config,
    container.z_order
  );

  return element_handle_input(x, y, width, height, config, container);
}

/**
 * A clamped horizontal slider.
 *
 * `mode: "bar"` maps the pointer directly onto the visual bar. `mode: "numeric"`
 * presents the formatted number as the primary visual and scrubs it horizontally.
 *
 * @param {number} value Current controlled value.
 * @param {object} config Widget, range and style configuration.
 * @returns {object} Interaction state including the current `value` and `changed`.
 *
 * @example
 * speed = slider(speed, { id: "speed", min: 0, max: 10, step: 0.1 }).value;
 * exposure = slider(exposure, { id: "exposure", mode: "numeric", min: -5, max: 5 }).value;
 */
export function slider(value, config = {}) {
  config.widget_id = UIContext.get_unique_id();

  const container_index = UIContext.layout_stack.peek();
  const container =
    container_index !== null
      ? UIContext.layout_allocator.get(container_index.value)
      : {
          x: 0,
          y: 0,
          cursor: { x: 0, y: 0 },
          width: UIContext.canvas_size.width,
          height: UIContext.canvas_size.height,
          z_order: 0,
        };

  const mode =
    (config.mode ?? config.slider_type) === SliderMode.Numeric
      ? SliderMode.Numeric
      : SliderMode.Bar;
  const width = parse_dimension(config.width ?? 160, container.width);
  const height = parse_dimension(config.height ?? 24, container.height);
  const offset_x =
    config.x !== undefined
      ? parse_dimension(config.x, container.width)
      : container.auto_width
        ? 0
        : (container.width - width) / 2;
  const offset_y =
    config.y !== undefined
      ? parse_dimension(config.y, container.height)
      : container.auto_height
        ? 0
        : (container.height - height) / 2;
  const x =
    config.anchor_x === right
      ? container.x + container.width - offset_x - width
      : container.cursor.x + offset_x;
  const y =
    config.anchor_y === bottom
      ? container.y + container.height - offset_y - height
      : container.cursor.y + offset_y;

  const input = element_handle_input(
    x,
    y,
    width,
    height,
    { ...config, draggable: false },
    container
  );
  const resolved = resolve_slider_value(value, config);
  const interaction_id = `2d:${String(config.id ?? config.widget_id)}`;
  const interaction = update_slider_interaction({
    state: UIContext.slider_state,
    widget_id: interaction_id,
    mode,
    value: resolved.value,
    resolved,
    config,
    pointer: UIContext.input_state,
    hovered: input.hovered && !UIContext.drag_state.active,
    axis_start: { x, y: y + height * 0.5 },
    axis_end: { x: x + width, y: y + height * 0.5 },
  });
  if (interaction.active && UIContext.input_state.pressed) {
    InputProvider.consume_action(InputKey.B_mouse_left);
  }
  const ratio = slider_ratio(interaction.value, resolved);
  const display_text = format_slider_value(interaction.value, resolved, config);

  child_container_layout_update(container, x, y, width, height);

  push_draw_command(
    (ctx) => {
      ctx.save();

      const background_color =
        interaction.active && config.active_color
          ? config.active_color
          : input.hovered && config.hover_color
            ? config.hover_color
            : (config.background_color ?? "#252a31");
      const draw_config = {
        corner_radius: config.corner_radius ?? Math.min(4, height * 0.25),
        ...config,
        background_color,
      };
      base_draw(ctx, x, y, width, height, draw_config);

      const track_padding = Math.max(
        0,
        Number(config.track_padding ?? (mode === SliderMode.Bar ? 3 : 0))
      );
      const track_x = x + track_padding;
      const track_y = y + track_padding;
      const track_width = Math.max(0, width - track_padding * 2);
      const track_height = Math.max(0, height - track_padding * 2);
      const show_fill = mode === SliderMode.Bar || config.show_fill === true;

      if (show_fill && track_width > 0 && track_height > 0 && ratio > 0) {
        ctx.fillStyle = config.fill_color ?? "#4eaaff";
        ctx.beginPath();
        rounded_rect_path(
          ctx,
          track_x,
          track_y,
          track_width * ratio,
          track_height,
          Math.min(
            Number(config.fill_corner_radius ?? draw_config.corner_radius),
            track_height * 0.5
          )
        );
        ctx.fill();
      }

      if (mode === SliderMode.Bar && config.show_handle !== false && track_width > 0) {
        const handle_width = Math.min(track_width, Math.max(1, Number(config.handle_width ?? 8)));
        const handle_height = Math.max(1, Number(config.handle_height ?? track_height));
        const handle_x = Math.min(
          track_x + track_width - handle_width,
          Math.max(track_x, track_x + track_width * ratio - handle_width * 0.5)
        );
        const handle_y = y + (height - handle_height) * 0.5;
        ctx.fillStyle = config.handle_color ?? "#f4f7fb";
        ctx.beginPath();
        rounded_rect_path(
          ctx,
          handle_x,
          handle_y,
          handle_width,
          handle_height,
          Math.min(Number(config.handle_corner_radius ?? handle_width * 0.5), handle_height * 0.5)
        );
        ctx.fill();
      }

      if (mode === SliderMode.Numeric || config.show_value === true) {
        ctx.fillStyle = config.text_color ?? "#f4f7fb";
        ctx.font = config.font ?? "16px sans-serif";
        ctx.textAlign = config.text_align ?? center;
        ctx.textBaseline = middle;
        const text_padding = Number(config.text_padding ?? 8);
        const text_x =
          (config.text_align ?? center) === left
            ? x + text_padding
            : (config.text_align ?? center) === right
              ? x + width - text_padding
              : x + width * 0.5;
        ctx.fillText(display_text, text_x, y + height * 0.5);
      }

      ctx.restore();
    },
    config,
    container.z_order
  );

  const result = {
    ...input,
    ...interaction,
    ratio,
    mode,
    formatted_value: display_text,
    dragged: interaction.active && UIContext.input_state.pressed,
  };
  if (result.changed) {
    config.on_change?.(result.value, result);
  }
  return result;
}

/**
 * A compact row of numeric sliders for vector-like values.
 *
 * The helper owns the common label/component/slider layout while allowing each
 * layer to be styled independently through config objects. Component changes are
 * only committed by the actively scrubbed slider, avoiding idle quantization from
 * overwriting neighboring values.
 *
 * @param {ArrayLike<number>} value Current vector value.
 * @param {object} config Vector field configuration. Supports id, label,
 *   components, on_change, group_config, label_config, row_config,
 *   component_config(s), component_label_config/styles, slider_config and
 *   component_slider_configs.
 * @returns {object} Interaction state with { value, changed, changed_index, active, components }.
 *
 * @example
 * const result = vector_field(direction, {
 *   id: "light.direction",
 *   label: "Direction",
 *   components: ["X", "Y", "Z"],
 *   slider_config: { min: -1, max: 1, step: 0.01, precision: 2 },
 * });
 * if (result.changed) direction = result.value;
 */
export function vector_field(value, config = {}) {
  const source_value = Array.from(value ?? [], Number);
  const default_components = ["X", "Y", "Z", "W"];
  const component_labels = config.components ?? default_components.slice(0, source_value.length);
  const component_count = Math.min(source_value.length, component_labels.length);
  const next_value = source_value.slice();
  const component_states = [];
  const vector_id = String(config.id ?? `vector_field.${UIContext.id_counter}`);
  const has_label = config.label !== undefined && config.label !== null && config.label !== "";
  let changed = false;
  let changed_index = -1;

  const group_config = {
    layout: "column",
    gap: 4,
    width: "100%",
    height: has_label ? 52 : 30,
    x: 0,
    y: 0,
    ...(config.group_config ?? {}),
  };
  const label_config = {
    width: "100%",
    height: 18,
    x: 0,
    y: 0,
    font: "600 10px monospace",
    text_color: "#8d9aa5",
    text_align: "left",
    text_valign: "middle",
    text_padding: 2,
    ...(config.label_config ?? {}),
  };
  const row_config = {
    layout: "row",
    gap: 6,
    width: "100%",
    height: 30,
    x: 0,
    y: 0,
    ...(config.row_config ?? {}),
  };

  panel(group_config, () => {
    if (has_label) {
      label(String(config.label), label_config);
    }

    panel(row_config, () => {
      const row_index = UIContext.layout_stack.peek();
      const row_container = UIContext.layout_allocator.get(row_index.value);
      const available_width = Math.max(0, row_container.width);
      const component_gap = row_container.gap;
      const default_component_width =
        component_count > 0
          ? Math.max(0, (available_width - component_gap * (component_count - 1)) / component_count)
          : 0;

      for (let index = 0; index < component_count; index++) {
        const component_override = config.component_configs?.[index] ?? {};
        const configured_component_width =
          component_override.width ?? config.component_config?.width;
        const component_width =
          configured_component_width !== undefined
            ? parse_dimension(configured_component_width, available_width)
            : default_component_width;
        const component_config = {
          layout: "row",
          gap: 0,
          width: component_width,
          height: row_container.height,
          x: 0,
          y: 0,
          background_color: "rgba(255, 255, 255, 0.04)",
          border: "1px solid rgba(255, 255, 255, 0.08)",
          corner_radius: 5,
          ...(config.component_config ?? {}),
          ...component_override,
          width: component_width,
        };
        const component_label_config = {
          width: 24,
          height: row_container.height,
          x: 0,
          y: 0,
          font: "700 11px monospace",
          text_color: "#aebbc5",
          text_align: "center",
          text_valign: "middle",
          ...(config.component_label_config ?? {}),
          ...(config.component_label_styles?.[index] ?? {}),
        };
        const component_label_width = parse_dimension(
          component_label_config.width,
          component_width
        );
        const component_slider_override = config.component_slider_configs?.[index] ?? {};
        const component_slider_config = {
          width: Math.max(0, component_width - component_label_width),
          height: row_container.height,
          x: 0,
          y: 0,
          mode: SliderMode.Numeric,
          background_color: "rgba(255, 255, 255, 0.05)",
          hover_color: "rgba(78, 170, 255, 0.14)",
          active_color: "rgba(78, 170, 255, 0.22)",
          text_color: "#f4f7fb",
          font: "11px monospace",
          corner_radius: 4,
          ...(config.slider_config ?? {}),
          ...component_slider_override,
          id: `${vector_id}.${String(component_labels[index]).toLowerCase()}`,
        };

        panel(component_config, () => {
          label(String(component_labels[index]), component_label_config);
          const state = slider(source_value[index], component_slider_config);
          component_states.push(state);

          if (state.active && state.changed) {
            next_value[index] = state.value;
            changed = true;
            changed_index = index;
          }
        });
      }
    });
  });

  const result = {
    value: next_value,
    changed,
    changed_index,
    active: component_states.some((state) => state.active),
    components: component_states,
  };
  if (changed) {
    config.on_change?.(next_value, changed_index, result);
  }
  return result;
}

/**
 * Label widget.
 * Simply draws text using the provided config.
 *
 * @param {string} text - The label text.
 * @param {object} config - Optional configuration:
 *    offsetX, offsetY, font, textColor, etc.
 */
export function label(text, config = {}) {
  config.widget_id = UIContext.get_unique_id();

  const container_index = UIContext.layout_stack.peek();
  const container =
    container_index !== null
      ? UIContext.layout_allocator.get(container_index.value)
      : {
          x: 0,
          y: 0,
          cursor: { x: 0, y: 0 },
          width: UIContext.canvas_size.width,
          height: UIContext.canvas_size.height,
          z_order: 0,
        };

  // Determine the effective font, padding, etc.
  const font = config.font || "16px sans-serif";
  const text_padding = config.text_padding || 0;

  // ------------------------------
  // Compute Dimensions: fit-content vs. fixed
  // ------------------------------
  let width, height;

  if (config.width === "fit-content") {
    // Use the shared measurement context for text measurement.
    const mc = UIContext.get_measure_context();
    mc.font = font;
    const metrics = mc.measureText(text);
    width = metrics.width + text_padding * 2;
  } else {
    width = parse_dimension(config.width, container.width);
  }

  let lines = [text];
  if (config.wrap) {
    const max_text_width = width - text_padding * 2;
    lines = wrap_text(text, font, max_text_width);
  }

  if (config.height === "fit-content") {
    // There is no built–in text height measurement; extract font size and use a heuristic.
    const font_match = font.match(/(\d+)px/);
    const font_size = font_match ? parseInt(font_match[1], 10) : 16;
    height = (config.wrap ? lines.length : 1) * font_size * 1.2;
  } else {
    height = parse_dimension(config.height, container.height);
  }

  // ------------------------------
  // Determine Offsets Using Auto-sizing Flags
  // ------------------------------
  let offset_x;
  if (config.x !== undefined) {
    offset_x = parse_dimension(config.x, container.width);
  } else {
    offset_x = container.auto_width ? 0 : (container.width - width) / 2;
  }

  let offset_y;
  if (config.y !== undefined) {
    offset_y = parse_dimension(config.y, container.height);
  } else {
    offset_y = container.auto_height ? 0 : (container.height - height) / 2;
  }

  let x;
  if (config.anchor_x === right) {
    x = container.x + container.width - offset_x - width;
  } else {
    x = container.cursor.x + offset_x;
  }

  let y;
  if (config.anchor_y === bottom) {
    y = container.y + container.height - offset_y - height;
  } else {
    y = container.cursor.y + offset_y;
  }

  // Update the parent's layout (for auto–layout on containers).
  child_container_layout_update(container, x, y, width, height);

  // ------------------------------
  // Add the Draw Command
  // ------------------------------
  push_draw_command(
    (ctx) => {
      ctx.save();
      ctx.font = font;

      // Draw the background, border, etc.
      base_draw(ctx, x, y, width, height, config);

      // Set up tracking for text color, alignment, and padding.
      ctx.fillStyle = config.text_color || "#000";
      const text_align = config.text_align || left;
      const text_valign = config.text_valign || top;

      let text_x;
      if (text_align === left) {
        text_x = x + text_padding;
      } else if (text_align === right) {
        text_x = x + width - text_padding;
      } else {
        text_x = x + width / 2;
      }

      let text_y;
      if (text_valign === top) {
        text_y = y + text_padding;
      } else if (text_valign === bottom) {
        text_y = y + height - text_padding;
      } else {
        text_y = y + height / 2;
      }

      if (config.wrap) {
        // Estimate a line height by extracting the font size (assumes font in px).
        const font_size_match = font.match(/(\d+)px/);
        const font_size = font_size_match ? parseInt(font_size_match[1], 10) : 16;
        const line_height = font_size;
        const total_text_height = lines.length * (line_height + text_padding);

        // Determine the starting y based on vertical alignment.
        let start_y;
        if (text_valign === top) {
          start_y = y + text_padding + line_height;
        } else if (text_valign === bottom) {
          start_y = y + height - text_padding - total_text_height + line_height;
        } else {
          start_y = y + (height - total_text_height) / 2 + line_height;
        }

        for (let i = 0; i < lines.length; i++) {
          ctx.fillText(lines[i], text_x, start_y + i * line_height);
        }
      } else {
        // Fall-back: no wrapping, single-line text.
        ctx.textAlign = text_align;
        ctx.textBaseline = text_valign === center ? middle : text_valign;
        ctx.fillText(text, text_x, text_y);
      }

      // ------------------------------
      // Underline on Hover (optional)
      // ------------------------------
      if (config.underline_on_hover && is_input_within(x, y, width, height)) {
        const measure_ctx = UIContext.get_measure_context();
        measure_ctx.font = font;
        const underline_width = measure_ctx.measureText(text).width;
        ctx.beginPath();
        ctx.strokeStyle = config.underline_color || "#4eaaff";
        ctx.lineWidth = 2;
        // Position the underline a couple pixels below the text baseline.
        ctx.moveTo(text_x, text_y + height / 2);
        ctx.lineTo(text_x + underline_width, text_y + height / 2);
        ctx.stroke();
      }

      ctx.restore();
    },
    config,
    container.z_order
  );

  // Return the input handling results.
  return element_handle_input(x, y, width, height, config, container);
}

/**
 * Image widget.
 *
 * Draws an image at the specified position.
 *
 * @param {object} config - Configuration for the image:
 *   x, y: offset from top–left of container (default 0)
 *   width, height: size of the image (default 100)
 *   icon: URL for an image file.
 */
export function image(config = {}) {
  config.widget_id = UIContext.get_unique_id();

  const container_index = UIContext.layout_stack.peek();
  const container =
    container_index !== null
      ? UIContext.layout_allocator.get(container_index.value)
      : {
          x: 0,
          y: 0,
          cursor: { x: 0, y: 0 },
          width: UIContext.canvas_size.width,
          height: UIContext.canvas_size.height,
          z_order: 0,
        };

  let width = parse_dimension(config.width, container.width);
  let height = parse_dimension(config.height, container.height);

  // If the image is provided, adjust the width and height to fit the image.
  if (config.src) {
    if (!UIContext.image_cache[config.src]) {
      const img = new Image();
      img.src = config.src;
      UIContext.image_cache[config.src] = img;
    }
    const img = UIContext.image_cache[config.src];
    if (img && img.complete) {
      const aspect = img.width / img.height;
      if (width > 0 && height <= 0) {
        height = width / aspect;
      } else if (height > 0 && width <= 0) {
        width = height * aspect;
      }
    }
  }

  // ------------------------------
  // Determine Offsets Using Auto-sizing Flags
  // ------------------------------
  let offset_x;
  if (config.x !== undefined) {
    offset_x = parse_dimension(config.x, container.width);
  } else {
    offset_x = container.auto_width ? 0 : (container.width - width) / 2;
  }

  let offset_y;
  if (config.y !== undefined) {
    offset_y = parse_dimension(config.y, container.height);
  } else {
    offset_y = container.auto_height ? 0 : (container.height - height) / 2;
  }

  let x;
  if (config.anchor_x === right) {
    x = container.x + container.width - offset_x - width;
  } else {
    x = container.cursor.x + offset_x;
  }

  let y;
  if (config.anchor_y === bottom) {
    y = container.y + container.height - offset_y - height;
  } else {
    y = container.cursor.y + offset_y;
  }

  // Update the parent's layout (for auto–layout on containers).
  child_container_layout_update(container, x, y, width, height);

  push_draw_command(
    (ctx) => {
      ctx.save();

      base_draw(ctx, x, y, width, height, config);

      if (config.src) {
        const img = UIContext.image_cache[config.src];
        if (img.complete) {
          if (config.cover) {
            // Calculate scaling factors for both dimensions
            const scale_x = width / img.width;
            const scale_y = height / img.height;
            // Use the larger scaling factor to ensure coverage
            const scale = Math.max(scale_x, scale_y);

            // Calculate dimensions at this scale
            const scaled_width = img.width * scale;
            const scaled_height = img.height * scale;

            // Center the image
            const offset_x = (width - scaled_width) / 2;
            const offset_y = (height - scaled_height) / 2;

            ctx.drawImage(img, x + offset_x, y + offset_y, scaled_width, scaled_height);
          } else {
            ctx.drawImage(img, x, y, width, height);
          }
        }
      }
      ctx.restore();
    },
    config,
    container.z_order
  );

  return element_handle_input(x, y, width, height, config, container);
}

/**
 * Input field widget.
 *
 * Supports a simple text input.
 * (Focus is activated when the input is clicked. Keyboard events are processed
 * from the global keyboardEvents array. In a full implementation you might hook
 * into a real keyboard input system.)
 *
 * @param {string} name - A unique name to use for storing persistent input state.
 * @param {object} config - Configuration for appearance and initial value.
 *   placeholder: Optional placeholder text shown while the value is empty.
 *   placeholder_color: Optional color override for the placeholder text.
 *   blur_on_click_away: Remove focus when a click lands outside the field.
 *
 * @returns {object} The updated state { value, isFocused }.
 */
export function input(name, config = {}) {
  config.widget_id = UIContext.get_unique_id();

  if (!UIContext.input_field_state[name]) {
    UIContext.input_field_state[name] = {
      value: config.value || "",
      is_focused: false,
    };
  }

  const field_state = UIContext.input_field_state[name];

  const container_index = UIContext.layout_stack.peek();
  const container =
    container_index !== null
      ? UIContext.layout_allocator.get(container_index.value)
      : {
          x: 0,
          y: 0,
          cursor: { x: 0, y: 0 },
          width: UIContext.canvas_size.width,
          height: UIContext.canvas_size.height,
          z_order: 0,
        };

  const input_state = UIContext.input_state;
  let width = parse_dimension(config.width, container.width);
  let height = parse_dimension(config.height, container.height);

  // Determine offsets using auto-sizing flags.
  let offset_x;
  if (config.x !== undefined) {
    offset_x = parse_dimension(config.x, container.width);
  } else {
    offset_x = container.auto_width ? 0 : (container.width - width) / 2;
  }

  let offset_y;
  if (config.y !== undefined) {
    offset_y = parse_dimension(config.y, container.height);
  } else {
    offset_y = container.auto_height ? 0 : (container.height - height) / 2;
  }

  let x;
  if (config.anchor_x === right) {
    x = container.x + container.width - offset_x - width;
  } else {
    x = container.cursor.x + offset_x;
  }

  let y;
  if (config.anchor_y === bottom) {
    y = container.y + container.height - offset_y - height;
  } else {
    y = container.cursor.y + offset_y;
  }

  child_container_layout_update(container, x, y, width, height);

  const font = config.font || "14px sans-serif";

  // Hit testing: clicks can focus the field or optionally blur it when they land elsewhere.
  const hovered =
    input_state.x >= x &&
    input_state.x <= x + width &&
    input_state.y >= y &&
    input_state.y <= y + height;
  if (input_state.clicked) {
    if (hovered) {
      field_state.is_focused = true;
    } else if (config.blur_on_click_away) {
      field_state.is_focused = false;
    }
  }

  if (field_state.is_focused) {
    const now = performance.now();
    const repeat_delay = config.repeat_delay || 500;
    // Detect modifier keys and set up Shift transformations
    const shift_down =
      InputProvider.get_state(InputKey.K_LShift) || InputProvider.get_state(InputKey.K_RShift);
    const ctrl_down =
      InputProvider.get_state(InputKey.K_LControl) || InputProvider.get_state(InputKey.K_RControl);
    const alt_down =
      InputProvider.get_state(InputKey.K_LAlt) || InputProvider.get_state(InputKey.K_RAlt);

    for (let i = 0; i < UIContext.keyboard_events.length; i++) {
      const key = UIContext.keyboard_events.get(i);
      if (key.consumed) continue;

      let action_callback = null;
      if (key.key === InputKey.K_Backspace) {
        action_callback = () => {
          field_state.value = field_state.value.slice(0, -1);
        };
      } else if (InputKeyToPrintableString[key.key]) {
        action_callback = () => {
          const base_str = InputKeyToPrintableString[key.key];
          if (!base_str) return;

          // Avoid inserting characters when ctrl/alt are held (reserved for shortcuts)
          if (ctrl_down || alt_down) return;

          let out = base_str;
          if (out.length === 1 && shift_down) {
            out = shift_char_map[out] || out.toUpperCase();
          }

          field_state.value += out;
        };
      }

      if (action_callback) {
        if (key.first || (key.held && now - key.last_change_time >= repeat_delay)) {
          action_callback();
          key.consumed = true;
        }
      }

      InputProvider.consume_action(key.key);
      InputProvider.consume_state(key.key);
    }
  }

  push_draw_command(
    (ctx) => {
      ctx.save();

      base_draw(ctx, x, y, width, height, config);

      const display_text =
        field_state.value.length > 0 ? field_state.value : (config.placeholder ?? "");
      const showing_placeholder = field_state.value.length === 0 && display_text.length > 0;

      ctx.fillStyle = showing_placeholder
        ? (config.placeholder_color ?? "rgba(255, 255, 255, 0.5)")
        : config.text_color || "#fff";
      ctx.font = font;
      ctx.textAlign = left;
      ctx.textBaseline = middle;
      const text_x = x + 10;
      const text_y = y + height / 2;
      ctx.fillText(display_text, text_x, text_y);

      if (field_state.is_focused) {
        const text_metrics = ctx.measureText(field_state.value);
        const caret_x = text_x + text_metrics.width;
        ctx.beginPath();
        ctx.moveTo(caret_x, y + 5);
        ctx.lineTo(caret_x, y + height - 5);
        ctx.strokeStyle = config.cursor_color || "#fff";
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      ctx.restore();
    },
    config,
    container.z_order
  );

  return field_state;
}

/**
 * Cursor widget.
 *
 * Updates the cursor's screen–position and world coordinates based on input.
 * It draws an icon if provided (config.icon) or fallback text / outline.
 *
 * Note: This widget is special: it's meant to be invoked every frame so that it tracks
 * the latest mouse position.
 *
 * @param {object} config - Configuration for style, optional icon, text, etc.
 */
export function cursor(config = {}) {
  config.widget_id = UIContext.get_unique_id();
  config.z_order ??= UI_Z_ORDER_CURSOR;

  const input_state = UIContext.input_state;
  input_state.depth += input_state.wheel;

  push_draw_command((ctx) => {
    ctx.save();

    base_draw(ctx, input_state.x, input_state.y, config.width || 20, config.height || 20, config);

    // Draw icon if provided.
    let width = config.width || 20;
    let height = config.height || 20;

    if (config.icon) {
      if (!UIContext.image_cache[config.icon]) {
        const img = new Image();
        img.src = config.icon;
        UIContext.image_cache[config.icon] = img;
      }
      const img = UIContext.image_cache[config.icon];

      if (img.complete) {
        ctx.drawImage(img, input_state.x, input_state.y, width, height);
      } else if (config.text) {
        ctx.fillStyle = config.text_color || "#fff";
        ctx.font = config.font || "16px sans-serif";
        const text_metrics = ctx.measureText(config.text);
        const text_width = text_metrics.width;
        const text_x = input_state.x + (width - text_width) / 2;
        const text_y = input_state.y + height / 2 + 6;
        ctx.fillText(config.text, text_x, text_y);
      }
    } else if (config.text) {
      // Draw text if no icon.
      ctx.fillStyle = config.text_color || "#fff";
      ctx.font = config.font || "16px sans-serif";
      const text_metrics = ctx.measureText(config.text);
      const text_width = text_metrics.width;
      const text_x = input_state.x + (width - text_width) / 2;
      const text_y = input_state.y + height / 2 + 6;
      ctx.fillText(config.text, text_x, text_y);
    }

    ctx.restore();
  }, config);
}

/**
 * Immediate UI updater.
 *
 * This class is used as a base to implement immediate mode UI updates.
 *
 */
export class ImmediateUIUpdater {
  static all_updaters = [];

  constructor() {
    ImmediateUIUpdater.all_updaters.push(this);
  }

  update(delta_time) {
    throw new Error("Not implemented");
  }

  static update_all(delta_time) {
    for (let i = 0; i < ImmediateUIUpdater.all_updaters.length; i++) {
      ImmediateUIUpdater.all_updaters[i].update(delta_time);
    }
  }

  static remove_of_type(type) {
    for (let i = 0; i < ImmediateUIUpdater.all_updaters.length; i++) {
      if (ImmediateUIUpdater.all_updaters[i] instanceof type) {
        ImmediateUIUpdater.all_updaters.splice(i, 1);
      }
    }
  }
}

/**
 * Property animator.
 *
 * Helper class used to animate properties over time.
 *
 */
export class PropertyAnimator {
  #current_value = null;
  #target_value = null;
  #animation_speed = 5;
  #is_array = false;

  constructor(initial_value, animation_speed = 5) {
    this.#is_array = Array.isArray(initial_value);
    if (this.#is_array) {
      this.#current_value = [...initial_value];
      this.#target_value = [...initial_value];
    } else {
      this.#current_value = initial_value;
      this.#target_value = initial_value;
    }
    this.#animation_speed = animation_speed;
  }

  set_target(target_value) {
    if (this.#is_array) {
      for (let i = 0; i < this.#current_value.length; i++) {
        this.#target_value[i] = target_value[i];
      }
    } else {
      this.#target_value = target_value;
    }
  }

  update(delta_time) {
    if (this.#is_array) {
      // Handle array values (like RGB colors)
      for (let i = 0; i < this.#current_value.length; i++) {
        const diff = this.#target_value[i] - this.#current_value[i];
        this.#current_value[i] = clamp(
          this.#current_value[i] + diff * this.#animation_speed * delta_time,
          diff > 0 ? this.#current_value[i] : this.#target_value[i],
          diff < 0 ? this.#current_value[i] : this.#target_value[i]
        );
      }
    } else {
      // Handle single numeric values
      const diff = this.#target_value - this.#current_value;
      this.#current_value = clamp(
        this.#current_value + diff * this.#animation_speed * delta_time,
        diff > 0 ? this.#current_value : this.#target_value,
        diff < 0 ? this.#current_value : this.#target_value
      );
    }
    return this.#current_value;
  }

  get value() {
    return this.#current_value;
  }
}
