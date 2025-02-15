import { DevConsoleTool } from "./dev_console_tool.js";
import { MasterMind } from "../ml/mastermind.js";
import { InputProvider } from "../input/input_provider.js";
import { InputKey } from "../input/input_types.js";
import { panel, label } from "../ui/2d/immediate.js";

/*
  Panel and label configurations adapted for immediate mode UI.
  Adjust these values as needed.
*/
const stats_panel_config = {
  layout: "column",
  gap: 4,
  y: "20%",
  dont_consume_cursor_events: true,
  background_color: "rgba(0, 0, 0, 0.7)",
  width: 600,
  padding: 10,
  border: "1px solid rgb(68, 68, 68)",
  corner_radius: 5,
};

const stats_label_config = {
  text_color: "#fff",
  font: "16px monospace",
  height: 20,
  width: "100%",
  text_valign: "middle",
  text_align: "left",
  text_padding: 5,
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
    this.render();
  }

  /**
   * Renders the ML statistics panel.
   * This method should be called every frame as part of the render loop.
   */
  render() {
    let panel_state = panel(stats_panel_config,
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

    if (this.is_open && InputProvider.get_action(InputKey.B_mouse_left)) {
      if (!panel_state.hovered) {
        this.hide();
      }
      InputProvider.consume_action(InputKey.B_mouse_left);
    }
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
}
