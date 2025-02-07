import { Element } from "./element.js";

const label_name = "label";

export class Label extends Element {
  init(name, config, children = []) {
    super.init(name, config, children, label_name);
    if (this.config.text) {
      this.dom.textContent = this.config.text;
    }
  }

  set_text(text) {
    this.config.text = text;
    this.dom.textContent = text;
  }

  static create(name, config, children = []) {
    const label = new Label();
    label.init(name, config, children);
    return label;
  }

}
