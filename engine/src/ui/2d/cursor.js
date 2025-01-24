import { Renderer } from "../../renderer/renderer.js";
import { Element } from "./element.js";
import { InputProvider } from "../../input/input_provider.js";
import { InputRange } from "../../input/input_types.js";
import { SharedViewBuffer, SharedFrameInfoBuffer } from "../../core/shared_data.js";
import { screen_pos_to_world_pos } from "../../utility/camera.js";
import { vec3 } from "gl-matrix";

const pixel_suffix = "px";
const cursor_name = "cursor";
const img_name = "img";
const default_cursor_transition = "all 0.2s ease";

export class Cursor extends Element {
  icon = null;
  icon_change_stack = [];
  world_position = vec3.create();
  prev_x = 0;
  prev_y = 0;
  current_depth = 2;

  init(name, config, children = []) {
    super.init(name, config, children, cursor_name);

    this.config.allows_cursor_events = false;

    if (this.config.icon) {
      this.set_icon(this.config.icon);
    }
  }

  update(delta_time) {
    super.update(delta_time);

    const x = InputProvider.get().get_range(InputRange.M_xabs);
    const y = InputProvider.get().get_range(InputRange.M_yabs);
    const mouse_wheel =
      InputProvider.get().get_range(InputRange.M_wheel) * 0.01;

    if (this.prev_x !== x || this.prev_y !== y || mouse_wheel !== 0) {
      this.current_depth += mouse_wheel;

      this.dom.style.left = x + pixel_suffix;
      this.dom.style.top = y + pixel_suffix;

      const renderer = Renderer.get();
      this.world_position = screen_pos_to_world_pos(
        SharedViewBuffer.get_view_data(0),
        x,
        y,
        renderer.canvas.width,
        renderer.canvas.height,
        this.current_depth
      );

      SharedFrameInfoBuffer.set_cursor_world_position(this.world_position);

      this.prev_x = x;
      this.prev_y = y;
    }
  }

  set_icon(path) {
    const url = new URL(`${path}`, window.location.href);
    this.icon_change_stack.push(url.href);
    if (!this.icon) {
      this.icon = document.createElement(img_name);
      this.icon.src = url.href;
      this.icon.style.width = this.config.style.width;
      this.icon.style.height = this.config.style.height;
      this.icon.style.transition = default_cursor_transition;
      this.dom.appendChild(this.icon);
    }
    this.icon.src = url.href;
  }

  reset_icon() {
    if (this.icon_change_stack.length <= 1) {
      return;
    }
    this.icon_change_stack.pop();
    this.set_icon(this.icon_change_stack[this.icon_change_stack.length - 1]);
  }

  apply_style(style, reset = false) {
    super.apply_style(style, reset);
    if (!style) {
      style = this.config.style;
    }
    if (this.icon) {
      this.icon.style.width = style.width;
      this.icon.style.height = style.height;
    }
  }

  static create(name, config, children = []) {
    const cursor = new Cursor();
    cursor.init(name, config, children);
    return cursor;
  }
}
