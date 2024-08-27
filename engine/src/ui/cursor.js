import { Element } from "./element.js";
import { InputProvider } from "../input/input_provider.js";
import { InputRange } from "../input/input_types.js";

export class Cursor extends Element {
  context = null;
  icon = null;
  icon_change_stack = [];

  init(context, name, config, children = []) {
    super.init(context, name, config, children, "cursor");

    this.context = context;
    this.allows_cursor_events = false;

    if (this.config.icon) {
      this.set_icon(this.config.icon);
    }
  }

  update(delta_time) {
    super.update(delta_time);

    const x = InputProvider.get().get_range(InputRange.M_xabs);
    const y = InputProvider.get().get_range(InputRange.M_yabs);

    this.dom.style.left = x + "px";
    this.dom.style.top = y + "px";
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
