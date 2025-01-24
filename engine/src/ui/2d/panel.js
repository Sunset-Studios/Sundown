import { Element } from "./element.js";

const panel_name = "panel";

export class Panel extends Element {
  static create(name, config, children = []) {
    const panel = new Panel();
    panel.init(name, config, children, panel_name);
    return panel;
  }
}
