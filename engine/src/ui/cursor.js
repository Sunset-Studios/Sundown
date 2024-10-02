import { Element } from "./element.js";
import { InputProvider } from "../input/input_provider.js";
import { InputRange } from "../input/input_types.js";
import { SharedViewBuffer, SharedFrameInfoBuffer } from "../core/shared_data.js";
import { screen_pos_to_world_pos } from "../utility/camera.js";
import { vec3 } from "gl-matrix";

export class Cursor extends Element {
  context = null;
  icon = null;
  icon_change_stack = [];
  world_position = vec3.create();
  prev_x = 0;
  prev_y = 0;
  current_depth = 2;

  init(context, name, config, children = []) {
    super.init(context, name, config, children, "cursor");

    this.context = context;
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

      this.dom.style.left = x + "px";
      this.dom.style.top = y + "px";

      this.world_position = screen_pos_to_world_pos(
        SharedViewBuffer.get().get_view_data(0),
        x,
        y,
        this.context.canvas.width,
        this.context.canvas.height,
        this.current_depth
      );

      SharedFrameInfoBuffer.get().set_cursor_world_position(
        this.context,
        this.world_position
      );

      this.prev_x = x;
      this.prev_y = y;
    }
  }

  set_icon(path) {
    const url = new URL(`${path}`, window.location.href);
    this.icon_change_stack.push(url.href);
    if (!this.icon) {
      this.icon = document.createElement("img");
      this.icon.src = url.href;
      this.icon.style.width = this.config.style.width;
      this.icon.style.height = this.config.style.height;
      this.icon.style.transition = "all 0.2s ease";
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
    if (this.icon) {
      this.icon.style.width = style.width;
      this.icon.style.height = style.height;
    }
  }

  static create(context, name, config, children = []) {
    const cursor = new Cursor();
    cursor.init(context, name, config, children);
    return cursor;
  }
}
