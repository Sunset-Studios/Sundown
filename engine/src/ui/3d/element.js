import { InputProvider } from "../../input/input_provider.js";
import { InputRange, InputKey } from "../../input/input_types.js";
import { profile_scope } from "../../utility/performance.js";

export class Element3D {
  name = "";
  parent = null;
  children = [];
  config = {};
  events = {};

  was_cursor_inside = false;
  is_cursor_inside = false;
  was_clicked = false;
  is_clicked = false;
  was_pressed = false;
  is_pressed = false;

  init(context, name, config, parent = null, children = []) {
    this.name = name;
    this.parent = parent;
    this.config = config;

    if (children.length > 0) {
      children.forEach((child) => {
        this.add_child(child);
      });
    }

    this.recalculate_transform();
  }

  update(delta_time) {
    profile_scope("Element.update", () => {
      this.recalculate_transform();

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
    // TODO: Implement
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

  recalculate_transform() {
    // TODO: Implement
  }

  add_child(child) {
    this.children.push(child);
  }

  remove_child(child) {
    this.children = this.children.filter((c) => c !== child);
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
