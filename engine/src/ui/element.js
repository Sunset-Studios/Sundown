import { InputProvider } from "../input/input_provider.js";
import { InputRange, InputKey } from "../input/input_types.js";
import { profile_scope } from "../utility/performance.js";

export function element_type_to_dom_type(element_type) {
  switch (element_type) {
    case "panel":
      return "div";
    case "button":
      return "button";
    case "label":
      return "label";
    case "input":
      return "input";
    default:
      return "div";
  }
}

export class Element {
  name = "";
  children = [];
  events = {};
  config = {};
  dom = null;
  client_rect = null;

  was_cursor_inside = false;
  is_cursor_inside = false;
  was_clicked = false;
  is_clicked = false;
  was_pressed = false;
  is_pressed = false;

  init(context, name, config, children = [], element_type = "div") {
    this.name = name;
    this.config = config;

    this.dom = document.createElement(element_type_to_dom_type(element_type));
    this.dom.id = `${element_type}-${this.name}`;
    this.dom.classList.add(element_type);

    if (this.config.style) {
      Object.assign(this.dom.style, this.config.style);
    }

    if (children.length > 0) {
      children.forEach((child) => {
        this.add_child(child);
      });
    }

    this.recalculate_client_rect();
  }

  update(delta_time) {
    profile_scope("Element.update", () => {
      this.recalculate_client_rect();

      for (let i = 0; i < this.children.length; i++) {
        this.children[i].update(delta_time);
      }

      const x = InputProvider.get().get_range(InputRange.M_xabs);
      const y = InputProvider.get().get_range(InputRange.M_yabs);

      const current_rect = this.rect;

      if (this.config.allows_cursor_events) {
        this.was_cursor_inside = this.is_cursor_inside;
        this.is_cursor_inside =
          x >= current_rect.left &&
          x <= current_rect.right &&
          y >= current_rect.top &&
          y <= current_rect.bottom;

        this.was_clicked = this.is_clicked;
        this.is_clicked =
          this.is_cursor_inside &&
          InputProvider.get().get_action(InputKey.B_mouse_left);

        this.was_pressed = this.is_pressed;
        this.is_pressed =
          this.is_cursor_inside &&
          InputProvider.get().get_state(InputKey.B_mouse_left);

        if (
          !this.was_cursor_inside &&
          this.is_cursor_inside &&
          this.config.hover_style
        ) {
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

        if (this.is_clicked) {
          this.trigger("selected");
          InputProvider.get().consume_action(InputKey.B_mouse_left);
        }
        if (this.is_pressed) {
          this.trigger("pressed");
          InputProvider.get().consume_action(InputKey.B_mouse_left);
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

  get is_clicked() {
    return this.is_clicked;
  }

  get is_pressed() {
    return this.is_pressed;
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
  }

  add_child(child) {
    this.children.push(child);
    this.dom.appendChild(child.dom);
  }

  remove_child(child) {
    this.children = this.children.filter((c) => c !== child);
    this.dom.removeChild(child.dom);
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
}
