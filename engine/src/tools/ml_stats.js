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
  y: 25,
  x: 25,
  anchor_x: "right",
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
          const stats = mastermind.get_subnet_stats();
          all_stats.push(...stats);
        }

        // For each model, render one label per statistic.
        for (let i = 0; i < all_stats.length; i++) {
          const entry = all_stats[i];
          const subnet_name = entry.name || `subnet_${i}`;
          const subnet_stats = entry.stats;
          for (let j = 0; j < subnet_stats.length; j++) {
            const stat = subnet_stats[j];
            if (stat && stat.loss) {
              label(
                `${subnet_name} - ${stat.name}: ${stat.loss ? stat.loss.data[0] : "N/A"}`,
                stats_label_config
              );
            }
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
