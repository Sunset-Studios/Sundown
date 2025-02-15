import { InputProvider } from "../../input/input_provider.js";
import { InputKey, InputRange, InputKeyToPrintableString } from "../../input/input_types.js";
import { FrameAllocator, FrameStackAllocator } from "../../memory/allocator.js";
import { profile_scope } from "../../utility/performance.js";

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
   * Global layout allocator.
   */
  layout_allocator: new FrameAllocator(100, LayoutStackContainer),
  /**
   * Global layout stack.
   */
  layout_stack: new FrameStackAllocator(100, 0),
  /**
   * Global unique ID counter.
   */
  id_counter: 0,
  /**
   * Global drag state.
   */
  drag_state: {
    active: false,
    widget_id: null,
    offset_x: 0,
    offset_y: 0,
  },
  /**
   * Global input state must be updated externally by event handlers.
   * x, y are current mouse coordinates; pressed is true while the mouse button is down.
   * wasClicked is toggled true for one frame after mouse–up.
   */
  input_state: {
    x: 0,
    y: 0,
    prev_x: 0,
    prev_y: 0,
    depth: 2,
    pressed: false,
    was_clicked: false,
    world_position: null, // assume this is a vec3 from a math library
    wheel: 0,
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

  ctx.save();

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
  if (config.border) {
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

  // Additional style settings (e.g. box-shadow, filter)
  if (config.box_shadow) {
    const parts = config.box_shadow.split(" ");
    ctx.shadowOffsetX = parse_dimension(parts[0], 0);
    ctx.shadowOffsetY = parse_dimension(parts[1], 0);
    ctx.shadowBlur = parse_dimension(parts[2], 10);
    ctx.shadowColor = parts[3] || "rgba(0, 0, 0, 0.5)";
  }

  if (config.filter) {
    ctx.filter = config.filter;
  }
  ctx.restore();
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
    } else if (container.layout === column) {
      container.cursor.y += height + container.gap;
    }

    // ---------- Auto–size update ----------
    // Now that the container's cursor has been advanced,
    // use its new position to update the content sizing.
    if (container.auto_width) {
      container.content_max_x = Math.max(
        container.content_max_x,
        container.cursor.x - (container.x + container.padding_left + container.gap)
      );
    }
    if (container.auto_height) {
      container.content_max_y = Math.max(
        container.content_max_y,
        container.cursor.y - (container.y + container.padding_top + container.gap)
      );
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

// ------------------------------
// Reset & Flush
// ------------------------------

/**
 * Call at the start of every frame to clear the previous UI state.
 */
export function reset_ui(canvas_width = 0, canvas_height = 0) {
  UIContext.draw_commands = [];
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
    for (let i = 0; i < UIContext.draw_commands.length; i++) {
      UIContext.draw_commands[i](ctx);
    }
    UIContext.draw_commands.length = 0;
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
 *    layout: "absolute", "row", or "column" (default "absolute")
 *    gap: gap between children in auto–layout (default 0)
 *    padding: inner padding (default 0)
 *    (Other style settings such as background_color, border, etc. may also be provided.)
 */
export function begin_container(config = {}) {
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
  let layout = config.layout || absolute; // "row", "column", or "absolute"

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
  container.cursor = { x: x + padding_left, y: y + padding_top };
  container.config = config;
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
  UIContext.draw_commands.push((ctx) => {
    ctx.save();
    let total_width = container.width + container.padding_left + container.padding_right;
    let total_height = container.height + container.padding_top + container.padding_bottom;
    base_draw(ctx, container.x, container.y, total_width, total_height, config);
    ctx.restore();
  });
}

/**
 * End the current container.
 */
export function end_container() {
  // Before popping the container, update its size if auto-sizing is enabled.
  const container_index = UIContext.layout_stack.peek();
  const container = UIContext.layout_allocator.get(container_index.value);
  const input_state = UIContext.input_state;

  if (container.auto_width) {
    container.width = container.content_max_x;
  }
  if (container.auto_height) {
    container.height = container.content_max_y;
  }

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
      } else if (parent.layout === column) {
        parent.cursor.y += container.height + parent.gap;
        if (parent.auto_height) {
          const rel_y = container.y - (parent.y + parent.padding_top);
          parent.content_max_y = Math.max(
            parent.content_max_y,
            rel_y + container.height + container.padding_top + container.padding_bottom
          );
        }
      }
    }
  }

  const hovered = is_input_within(
    container.x,
    container.y,
    container.width + container.padding_left + container.padding_right,
    container.height + container.padding_top + container.padding_bottom
  );
  const clicked = hovered && input_state.clicked;
  const pressed = hovered && input_state.pressed;

  return { hovered, clicked, pressed };
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
        };
  const input_state = UIContext.input_state;
  const widget_id = UIContext.get_unique_id();

  // Determine widget width and height.
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

  // Hit testing against current input state.
  const hovered = is_input_within(x, y, width, height);
  const clicked = hovered && input_state.clicked;
  const pressed = hovered && input_state.pressed;

  if (!UIContext.drag_state.active && (clicked || pressed)) {
    InputProvider.consume_action(InputKey.B_mouse_left);
  }

  // ----------------------
  // Drag-and-Drop Handling
  // ----------------------
  if (config.draggable) {
    if (!UIContext.drag_state.active && hovered && input_state.pressed) {
      UIContext.drag_state.active = true;
      UIContext.drag_state.widget_id = widget_id;
      UIContext.drag_state.offset_x = input_state.x - x;
      UIContext.drag_state.offset_y = input_state.y - y;
      if (config.on_drag_start) config.on_drag_start(widget_id);
      InputProvider.consume_action(InputKey.B_mouse_left);
    }
    if (UIContext.drag_state.active && UIContext.drag_state.widget_id === widget_id) {
      if (input_state.pressed) {
        x = input_state.x - UIContext.drag_state.offset_x;
        y = input_state.y - UIContext.drag_state.offset_y;
        if (config.on_drag) config.on_drag(widget_id, x, y);
      } else {
        UIContext.drag_state.active = false;
        if (config.on_drop) config.on_drop(widget_id, x, y);
      }
    }
  }

  if (config.scrollable && input_state.wheel !== 0) {
    InputProvider.consume_range(InputRange.M_wheel);
  }

  child_container_layout_update(container, x, y, width, height);

  // ----------------------
  // Assemble the Draw Command
  // ----------------------
  UIContext.draw_commands.push((ctx) => {
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
        ctx.font = config.font || "16px sans-serif";
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
    } else {
      if (config.text || label) {
        ctx.fillStyle = config.text_color || "#fff";
        ctx.font = config.font || "16px sans-serif";
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
    ctx.restore();
  });

  return { hovered, clicked, pressed, widget_id };
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
        };

  // Use provided width/height to calculate anchoring; default width/height may be provided via config.
  const width = parse_dimension(config.width, container.width);
  const height = parse_dimension(config.height, container.height);

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

  // Hit testing against current input state.
  const hovered = is_input_within(x, y, width, height);
  const clicked = hovered && UIContext.input_state.clicked;
  const pressed = hovered && UIContext.input_state.pressed;

  const font = config.font || "16px sans-serif";
  const text_color = config.text_color || "#000";

  child_container_layout_update(container, x, y, width, height);

  UIContext.draw_commands.push((ctx) => {
    ctx.save();

    base_draw(ctx, x, y, width, height, config);

    ctx.fillStyle = text_color;
    ctx.font = font;
    const text_align = config.text_align || left;
    const text_valign = config.text_valign || top;
    const text_padding = config.text_padding || 0;
    let text_x, text_y;
    if (text_align === left) {
      text_x = x + text_padding;
    } else if (text_align === right) {
      text_x = x + width - text_padding;
    } else {
      text_x = x + width / 2;
    }
    if (text_valign === top) {
      text_y = y + text_padding;
    } else if (text_valign === bottom) {
      text_y = y + height - text_padding;
    } else {
      text_y = y + height / 2;
    }
    ctx.textAlign = text_align;
    ctx.textBaseline = text_valign === center ? middle : text_valign;
    ctx.fillText(text, text_x, text_y);

    ctx.restore();
  });

  return { hovered, clicked, pressed };
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
        };

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

  // Hit testing against current input state.
  const hovered = is_input_within(x, y, width, height);
  const clicked = hovered && input_state.clicked;
  const pressed = hovered && input_state.pressed;

  child_container_layout_update(container, x, y, width, height);

  UIContext.draw_commands.push((ctx) => {
    ctx.save();

    base_draw(ctx, x, y, width, height, config);

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
      }
    }
    ctx.restore();
  });

  return { hovered, clicked, pressed };
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
 *
 * @returns {object} The updated state { value, isFocused }.
 */
export function input(name, config = {}) {
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

  // Hit testing: if clicked, mark this widget as focused.
  const hovered =
    input_state.x >= x &&
    input_state.x <= x + width &&
    input_state.y >= y &&
    input_state.y <= y + height;
  if (hovered && input_state.was_clicked) {
    field_state.is_focused = true;
  }

  if (field_state.is_focused) {
    const now = performance.now();
    const repeat_delay = config.repeat_delay || 500;
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
          field_state.value += InputKeyToPrintableString[key.key];
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

  UIContext.draw_commands.push((ctx) => {
    ctx.save();

    base_draw(ctx, x, y, width, height, config);

    ctx.fillStyle = config.text_color || "#fff";
    ctx.font = font;
    ctx.textAlign = left;
    ctx.textBaseline = middle;
    const text_x = x + 10;
    const text_y = y + height / 2;
    ctx.fillText(field_state.value, text_x, text_y);

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
  });

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
  const input_state = UIContext.input_state;
  input_state.depth += input_state.mouse_wheel;

  UIContext.draw_commands.push((ctx) => {
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
  });
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
