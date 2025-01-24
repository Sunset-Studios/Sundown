export class DevConsoleTool {
  is_open = false;

  init() {}
  update() {}
  execute() {}
  show() {
    this.is_open = true;
  }
  hide() {
    this.is_open = false;
  }
  toggle() {
    if (this.is_open) {
      this.hide();
    } else {
      this.show();
    }
  }
}
