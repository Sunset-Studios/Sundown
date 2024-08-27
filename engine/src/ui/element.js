import { InputProvider } from "../input/input_provider.js";
import { InputRange, InputKey } from "../input/input_types.js";

export function element_type_to_dom_type(element_type) {
  switch (element_type) {
    case "panel":
      return "div";
    case "button":
      return "button";
    default:
      return "div";
  }
}

export class Element {
  name = "";
  children = [];
  config = {};
  dom = null;

  allows_cursor_events = true;
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
  }

  update(delta_time) {
    const x = InputProvider.get().get_range(InputRange.M_xabs);
    const y = InputProvider.get().get_range(InputRange.M_yabs);

    const rect = this.dom.getBoundingClientRect();

    if (this.allows_cursor_events) {
      this.was_cursor_inside = this.is_cursor_inside;
      this.is_cursor_inside =
        x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom;

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
        this.config.style
      ) {
        this.apply_style(this.config.style, true);
      }
    }

    this.children.forEach((child) => {
      child.update(delta_time);
    });
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

  apply_style(style, reset = false) {
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
}
