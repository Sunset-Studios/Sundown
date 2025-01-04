import { Element } from "./element.js";

export class Input extends Element {
  init(name, config, children = []) {
    super.init(name, config, children, "input");
      
    this.dom.type = this.config.type ?? "text";
    this.dom.value = this.config.value ?? "";
    this.config.allows_cursor_events = true;

    this.dom.addEventListener("change", (e) => {
      let new_value;
      if (this.dom.type === "checkbox") {
        new_value = e.target.checked;
      } else if (this.dom.type === "number") {
        new_value = parseFloat(e.target.value);
      } else {
        new_value = e.target.value;
      }
      this.trigger("changed", new_value);
    });
  }

  update(delta_time) {
    super.update(delta_time);
    if (this.is_clicked) {
      this.dom.focus();
    }
  }

  static create(name, config, children = []) {
    const input = new Input();
    input.init(name, config, children);
    return input;
  }
}
