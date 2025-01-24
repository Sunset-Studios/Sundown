import { Renderer } from "../../renderer/renderer.js";
import { InputProvider } from "../../input/input_provider.js";
import { InputRange, InputKey } from "../../input/input_types.js";
import { profile_scope } from "../../utility/performance.js";

const element_update_scope_name = "Element.update";
const selected_event_name = "selected";
const pressed_event_name = "pressed";
const root_name = "root";
const hidden_name = "hidden";
const none_name = "none";
const block_name = "block";
const element_name = "div";
const panel_name = "panel";
const button_name = "button";
const label_name = "label";
const input_name = "input";

const root_config = {
  style: {
    position: "absolute",
    top: "0",
    left: "0",
    width: "100%",
    height: "100%",
    pointerEvents: "none",
  },
};

const was_cursor_inside_bit = 1 << 0;
const is_cursor_inside_bit = 1 << 1;
const was_clicked_bit = 1 << 2;
const is_clicked_bit = 1 << 3;
const was_pressed_bit = 1 << 4;
const is_pressed_bit = 1 << 5;
const is_being_dragged_bit = 1 << 6;
const child_is_dragging_bit = 1 << 7;
const is_visible_bit = 1 << 8;

export function element_type_to_dom_type(element_type) {
  switch (element_type) {
    case panel_name:
      return element_name;
    case button_name:
      return button_name;
    case label_name:
      return label_name;
    case input_name:
      return input_name;
    default:
      return element_name;
  }
}

export class Element {
  static view_roots = [];

  name = "";
  children = [];
  events = {};
  config = {};
  dom = null;
  client_rect = null;

  state_flags = 0;

  drag_start_x = 0;
  drag_start_y = 0;
  drag_offset_x = 0;
  drag_offset_y = 0;
  drag_hold_start_time = 0;
  original_position = null;
  drag_target = null;

  get is_cursor_inside() {
    return (this.state_flags & is_cursor_inside_bit) !== 0;
  }

  set is_cursor_inside(value) {
    this.state_flags =
      (this.state_flags & ~is_cursor_inside_bit) | (value ? is_cursor_inside_bit : 0);
  }

  get was_cursor_inside() {
    return (this.state_flags & was_cursor_inside_bit) !== 0;
  }

  set was_cursor_inside(value) {
    this.state_flags =
      (this.state_flags & ~was_cursor_inside_bit) | (value ? was_cursor_inside_bit : 0);
  }

  get was_clicked() {
    return (this.state_flags & was_clicked_bit) !== 0;
  }

  set was_clicked(value) {
    this.state_flags = (this.state_flags & ~was_clicked_bit) | (value ? was_clicked_bit : 0);
  }

  get is_clicked() {
    return (this.state_flags & is_clicked_bit) !== 0;
  }

  set is_clicked(value) {
    this.state_flags = (this.state_flags & ~is_clicked_bit) | (value ? is_clicked_bit : 0);
  }

  get was_pressed() {
    return (this.state_flags & was_pressed_bit) !== 0;
  }

  set was_pressed(value) {
    this.state_flags = (this.state_flags & ~was_pressed_bit) | (value ? was_pressed_bit : 0);
  }

  get is_pressed() {
    return (this.state_flags & is_pressed_bit) !== 0;
  }

  set is_pressed(value) {
    this.state_flags = (this.state_flags & ~is_pressed_bit) | (value ? is_pressed_bit : 0);
  }

  get is_being_dragged() {
    return (this.state_flags & is_being_dragged_bit) !== 0;
  }

  set is_being_dragged(value) {
    this.state_flags =
      (this.state_flags & ~is_being_dragged_bit) | (value ? is_being_dragged_bit : 0);
  }

  get child_is_dragging() {
    return (this.state_flags & child_is_dragging_bit) !== 0;
  }

  set child_is_dragging(value) {
    this.state_flags =
      (this.state_flags & ~child_is_dragging_bit) | (value ? child_is_dragging_bit : 0);
  }

  get is_visible() {
    return (this.state_flags & is_visible_bit) !== 0;
  }

  set is_visible(value) {
    this.state_flags = (this.state_flags & ~is_visible_bit) | (value ? is_visible_bit : 0);
    this.apply_style();
  }

  init(name, config, children = [], element_type = element_name) {
    this.name = name;
    this.config = config;

    this.dom = document.createElement(element_type_to_dom_type(element_type));
    this.dom.id = `${element_type}-${this.name}`;
    this.dom.classList.add(element_type);

    this.is_visible = true;

    if (this.config.style) {
      Object.assign(this.dom.style, this.config.style);
    }

    if (this.config.is_draggable && this.config.drag_hold_delay === undefined) {
      this.config.drag_hold_delay = 0;
    }

    if (children.length > 0) {
      children.forEach((child) => {
        this.add_child(child);
      });
    }

    this.recalculate_client_rect();
  }

  update(delta_time) {
    profile_scope(element_update_scope_name, () => {
      for (let i = 0; i < this.children.length; i++) {
        this.children[i].update(delta_time);
      }

      if (this.dom.style.visibility === hidden_name || this.dom.style.display === none_name) {
        return;
      }

      this.recalculate_client_rect();

      const input = InputProvider.get();

      const x = input.get_range(InputRange.M_xabs);
      const y = input.get_range(InputRange.M_yabs);
      const clicked = input.get_action(InputKey.B_mouse_left);
      const pressed = input.get_state(InputKey.B_mouse_left);

      const current_rect = this.rect;

      if (this.config.allows_cursor_events) {
        this.was_cursor_inside = this.is_cursor_inside;
        this.is_cursor_inside =
          x >= current_rect.left &&
          x <= current_rect.right &&
          y >= current_rect.top &&
          y <= current_rect.bottom;

        this.was_clicked = this.is_clicked;
        this.is_clicked = this.is_cursor_inside && clicked;

        this.was_pressed = this.is_pressed;
        this.is_pressed = this.is_cursor_inside && pressed;

        this.child_is_dragging = this.children.some(
          (child) => child.is_being_dragged || child.child_is_dragging
        );

        if (this.config.is_draggable) {
          if (!this.child_is_dragging && !this.is_being_dragged && this.is_pressed) {
            const current_time = performance.now();

            // Start tracking hold time when mouse is first pressed
            if (this.drag_hold_start_time === 0) {
              this.drag_hold_start_time = current_time;
            }

            // Check if we've held long enough to start dragging
            const hold_duration = current_time - this.drag_hold_start_time;
            if (hold_duration >= this.config.drag_hold_delay) {
              // Start drag
              this.is_being_dragged = true;
              this.drag_start_x = x;
              this.drag_start_y = y;
              this.drag_offset_x = x - this.client_rect.left;
              this.drag_offset_y = y - this.client_rect.top;
              this.original_position = {
                left: this.dom.style.left,
                top: this.dom.style.top,
              };

              if (this.config.on_drag_start) {
                this.config.on_drag_start(this);
              }

              input.consume_action(InputKey.B_mouse_left);
            }
          }

          if (!pressed) {
            // Reset hold timer when mouse is released
            this.drag_hold_start_time = 0;
          }

          if (this.is_being_dragged) {
            // Update position while dragging
            if (pressed) {
              this.dom.style.left = `${x - this.drag_offset_x}px`;
              this.dom.style.top = `${y - this.drag_offset_y}px`;

              // Find potential drop target
              this.drag_target = this._find_drop_target(x, y);

              if (this.config.on_drag) {
                this.config.on_drag(this, this.drag_target);
              }
            } else {
              // End drag
              this.is_being_dragged = false;

              if (this.drag_target && this.config.on_drop) {
                this.config.on_drop(this, this.drag_target);
              } else if (this.config.on_drag_end) {
                this.config.on_drag_end(this);
              }

              this.drag_target = null;
            }
          }
        }

        if (!this.was_cursor_inside && this.is_cursor_inside && this.config.hover_style) {
          this.apply_style(this.config.hover_style);
        }
        if (
          this.was_cursor_inside &&
          !this.is_cursor_inside &&
          this.config.style &&
          this.config.hover_style
        ) {
          this.apply_style(this.config.style, true);
        }

        if (!this.child_is_dragging && this.is_clicked) {
          this.trigger(selected_event_name);
          if (!this.config.dont_consume_cursor_events) {
            input.consume_action(InputKey.B_mouse_left);
          }
        }
        if (!this.child_is_dragging && this.is_pressed) {
          this.trigger(pressed_event_name);
          if (!this.config.dont_consume_cursor_events) {
            input.consume_action(InputKey.B_mouse_left);
          }
        }

        if (this.config.is_scrollable && this.is_cursor_inside) {
          const wheel_delta = input.get_range(InputRange.M_wheel);
          if (wheel_delta !== 0) {
            let scroll_top = this.dom.scrollTop;
            scroll_top = Math.max(
              0,
              Math.min(
                this.dom.scrollHeight - this.dom.clientHeight,
                scroll_top + wheel_delta * this.config.scroll_amount
              )
            );
            this.dom.scrollTop = scroll_top;

            input.consume_range(InputRange.M_wheel);
          }
        }
      }
    });
  }

  destroy() {
    this.dom.remove();
  }

  get is_hovered() {
    return this.is_cursor_inside;
  }

  get rect() {
    return this.client_rect;
  }

  recalculate_client_rect() {
    this.client_rect = this.dom.getBoundingClientRect();
  }

  apply_style(style, reset = false) {
    if (!style) {
      style = this.config.style;
    }
    if (reset) {
      this.dom.style = {};
    }
    Object.assign(this.dom.style, style);

    this.dom.style.display = this.is_visible ? (style.display || block_name) : none_name;
  }

  add_child(child) {
    this.children.push(child);
    this.dom.appendChild(child.dom);
  }

  remove_child(child) {
    this.children = this.children.filter((c) => c !== child);
    this.dom.removeChild(child.dom);
  }

  get_child(name) {
    return this.children.find((child) => child.name === name);
  }

  clear_children() {
    for (let i = 0; i < this.children.length; i++) {
      this.children[i].destroy();
    }
    this.children = [];
    this.dom.innerHTML = "";
  }

  on(event, callback) {
    if (!this.events[event]) {
      this.events[event] = [];
    }
    if (!this.events[event].includes(callback)) {
      this.events[event].push(callback);
    }
  }

  trigger(event, ...args) {
    if (this.events[event]) {
      for (let i = 0; i < this.events[event].length; i++) {
        this.events[event][i](...args);
      }
    }
  }

  is_inside(x, y) {
    return (
      x >= this.client_rect.left &&
      x <= this.client_rect.right &&
      y >= this.client_rect.top &&
      y <= this.client_rect.bottom
    );
  }

  _find_drop_target(x, y) {
    // Get the root element to search through all UI elements
    const root = Element.get_view_root();
    return this._find_drop_target_recursive(root, x, y);
  }

  _find_drop_target_recursive(element, x, y) {
    if (element === this) return null; // Skip self

    // Check children first (top-most elements)
    for (let i = element.children.length - 1; i >= 0; i--) {
      const child = element.children[i];
      const target = this._find_drop_target_recursive(child, x, y);
      if (target) return target;
    }

    // Then check the element itself
    if (element.config.accepts_drops && element.is_inside(x, y)) {
      return element;
    }

    return null;
  }

  static new_view_root(view_index) {
    const ui_root = Element.create(root_name, root_config);

    this.view_roots.length = Math.max(this.view_roots.length, view_index + 1);
    if (this.view_roots[view_index]?.dom) {
      this.view_roots[view_index].dom.remove();
    }
    this.view_roots[view_index] = ui_root;

    Renderer.get().canvas.after(ui_root.dom);

    return ui_root;
  }

  static get_view_root(view_index = 0) {
    return this.view_roots[view_index];
  }

  static get_all_view_roots() {
    return this.view_roots;
  }

  static create(name, config, children = []) {
    const element = new Element();
    element.init(name, config, children, element_name);
    return element;
  }
}
