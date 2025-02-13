import { DevConsoleTool } from "./dev_console_tool.js";
import { MasterMind } from "../ml/mastermind.js";
import { InputProvider } from "../input/input_provider.js";
import { InputKey, InputRange } from "../input/input_types.js";
import { panel, label, UIContext } from "../ui/2d/immediate.js";

/*
  Panel and label configurations adapted for immediate mode UI.
  Adjust these values as needed.
*/
const stats_panel_config = {
  dont_consume_cursor_events: true,
  background_color: "rgba(0, 0, 0, 0.7)",
  padding: "10px",
  border: "1px solid rgb(68, 68, 68)",
  corner_radius: "5px",
  font: "16px monospace",
};

const stats_label_config = {
  text_color: "#fff",
  font: "16px monospace",
  // Note: Immediate mode UI's label() may not use margin settings.
};

/**
 * MLStats displays machine learning model statistics using the
 * immediate mode UI framework.
 */
export class MLStats extends DevConsoleTool {
  is_open = false;
  scene = null;

  /**
   * Called each frame to update the MLStats state. If the user clicks
   * outside the stats panel, the panel is hidden.
   */
  update(delta_time) {
    if (!this.is_open) return;

    if (InputProvider.get_action(InputKey.B_mouse_left)) {
      const mouse_x = InputProvider.get_range(InputRange.M_xabs);
      const mouse_y = InputProvider.get_range(InputRange.M_yabs);
      const panel_rect = this._get_panel_rect();
      if (!this._is_inside_panel(mouse_x, mouse_y, panel_rect)) {
        this.hide();
      }
      InputProvider.consume_action(InputKey.B_mouse_left);
    }

    if (!this.is_open) return;

    this.render();
  }

  /**
   * Renders the ML statistics panel.
   * This method should be called every frame as part of the render loop.
   */
  render() {
    if (!this.is_open) return;

    const panel_rect = this._get_panel_rect();

    panel(
      {
        ...stats_panel_config,
        x: `${panel_rect.x}px`,
        y: `${panel_rect.y}px`,
        width: `${panel_rect.width}px`,
        height: `${panel_rect.height}px`,
      },
      () => {
        // Gather the aggregated stats from all MasterMind instances.
        const all_masterminds = MasterMind.all_masterminds;
        let all_stats = [];
        for (let i = 0; i < all_masterminds.length; i++) {
          const mastermind = all_masterminds[i];
          const stats = mastermind.get_model_stats();
          all_stats.push(...stats);
        }

        // For each model, render one label per statistic.
        for (let i = 0; i < all_stats.length; i++) {
          const entry = all_stats[i];
          const model_name = entry.name || `model_${i}`;
          const model_stats = entry.stats;
          for (let j = 0; j < model_stats.length; j++) {
            const stat = model_stats[j];
            label(
              `${model_name} - ${stat.name}: ${stat.loss.data[0]}`,
              stats_label_config
            );
          }
        }
      }
    );
  }

  /**
   * Toggles the display of the ML stats panel.
   */
  execute() {
    this.toggle();
  }

  toggle() {
    this.is_open = !this.is_open;
    if (this.scene) {
      if (this.is_open && typeof this.scene.show_dev_cursor === "function") {
        this.scene.show_dev_cursor();
      } else if (!this.is_open && typeof this.scene.hide_dev_cursor === "function") {
        this.scene.hide_dev_cursor();
      }
    }
  }

  show() {
    this.is_open = true;
    if (this.scene && typeof this.scene.show_dev_cursor === "function") {
      this.scene.show_dev_cursor();
    }
  }

  hide() {
    this.is_open = false;
    if (this.scene && typeof this.scene.hide_dev_cursor === "function") {
      this.scene.hide_dev_cursor();
    }
  }

  set_scene(scene) {
    this.scene = scene;
  }

  /**
   * Computes the on-screen rectangle for the stats panel.
   * For this example, we center the panel.
   */
  _get_panel_rect() {
    const canvas_width = UIContext.canvas_size.width || window.innerWidth;
    const canvas_height = UIContext.canvas_size.height || window.innerHeight;
    const width = 600;
    const height = 200;
    const x = (canvas_width - width) / 2;
    const y = (canvas_height - height) / 2;
    return { x, y, width, height };
  }

  /**
   * Determines whether the given (x, y) coordinate lies within the panel.
   */
  _is_inside_panel(x, y, panel_rect) {
    return (
      x >= panel_rect.x &&
      x <= panel_rect.x + panel_rect.width &&
      y >= panel_rect.y &&
      y <= panel_rect.y + panel_rect.height
    );
  }
}
