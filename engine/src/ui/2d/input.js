import { Element } from "./element.js";

const input_name = "input";
const input_type_text = "text";
const input_type_number = "number";
const input_type_checkbox = "checkbox";
const change_event_name = "change";
const changed_event_name = "changed";

export class Input extends Element {
  init(name, config, children = []) {
    super.init(name, config, children, input_name);

    this.dom.type = this.config.type ?? input_type_text;
    this.dom.value = this.config.value ?? "";
    this.config.allows_cursor_events = true;

    this.dom.addEventListener(change_event_name, (e) => {
      let new_value;
      if (this.dom.type === input_type_checkbox) {
        new_value = e.target.checked;
      } else if (this.dom.type === input_type_number) {
        new_value = parseFloat(e.target.value);
      } else {
        new_value = e.target.value;
      }
      this.trigger(changed_event_name, new_value);
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
