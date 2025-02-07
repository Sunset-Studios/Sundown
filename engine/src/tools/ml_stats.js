import { MasterMind } from "../ml/mastermind.js";
import { InputProvider } from "../input/input_provider.js";
import { InputKey, InputRange } from "../input/input_types.js";
import { DevConsoleTool } from "./dev_console_tool.js";
import { Element } from "../ui/2d/element.js";
import { Panel } from "../ui/2d/panel.js";
import { Label } from "../ui/2d/label.js";

const ml_stats_name = "ml_stats";
const stats_panel_config = {
  dont_consume_cursor_events: true,
  style: {
    position: "absolute",
    display: "flex",
    flexDirection: "column",
    top: "50%",
    left: "50%",
    transform: "translate(-50%, -50%)",
    width: "600px",
    backgroundColor: "rgba(0, 0, 0, 0.7)",
    display: "flex",
    padding: "10px",
    border: "1px solid rgb(68, 68, 68)",
    borderRadius: "5px",
    fontFamily: "monospace",
  },
};
const stats_label_config = {
  color: "#fff",
  fontFamily: "monospace",
  fontSize: "16px",
  marginBottom: "5px",
};

export class MLStats extends DevConsoleTool {
  stats_panel = null;

  scene = null;
  model_label_map = new Map();

  init() {
    super.init();
    this._init_ui();
  }

  update() {
    super.update();

    if (!this.is_open) {
      return;
    }

    const input_provider = InputProvider.get();

    if (input_provider.get_action(InputKey.B_mouse_left)) {
      const mouse_x = input_provider.get_range(InputRange.M_xabs);
      const mouse_y = input_provider.get_range(InputRange.M_yabs);

      if (!this.stats_panel.is_inside(mouse_x, mouse_y)) {
        this.hide();
      }
    }

    if (!this.is_open) {
      return;
    }

    this._update_model_stats();
  }

  execute() {
    super.execute();
    this.toggle();
  }

  set_scene(scene) {
    this.scene = scene;
  }

  _init_ui() {
    this.stats_panel = Panel.create(ml_stats_name, stats_panel_config);

    const view_root = Element.get_view_root();
    view_root.add_child(this.stats_panel);

    this.stats_panel.is_visible = false;
  }

  /**
   * Updates the model stat labels.
   *
   * This version leverages the mastermind's aggregated stats:
   *   MasterMind.get_model_stats() is assumed to return an array of objects,
   *   each with a `name` property and a `stats` array (each stat having `name` and `loss`).
   *
   * For each model found in the stats:
   *   - If a persistent label group doesn't yet exist in the map, create it.
   *   - If it exists, update the corresponding label texts.
   * If a model is no longer present in the aggregated stats but exists in the map,
   * its labels are removed.
   */
  _update_model_stats() {
    const all_masterminds = MasterMind.all_masterminds;

    const all_stats = [];
    for (let i = 0; i < all_masterminds.length; i++) {
      const mastermind = all_masterminds[i];
      const stats = mastermind.get_model_stats();
      all_stats.push(...stats);
    }

    const current_models = new Set();

    for (let i = 0; i < all_stats.length; i++) {
      const entry = all_stats[i];
      const model_name = entry.name || `model_${i}`;
      const model_stats = entry.stats;
      let stat_labels = this.model_label_map.get(model_name);

      current_models.add(model_name);

      if (!stat_labels) {
        // Create new stat labels
        stat_labels = [];
        for (let j = 0; j < model_stats.length; j++) {
          const stat = model_stats[j];
          const label = Label.create(`model_stat_${model_name}_${j}`, {
            text: `${model_name} - ${stat.name}: ${stat.loss.data[0]}`,
            style: stats_label_config,
          });

          this.stats_panel.add_child(label);
          stat_labels.push(label);
        }
        this.model_label_map.set(model_name, stat_labels);
      } else {
        // Update existing labels (add new ones if needed, or remove extras)
        for (let j = 0; j < model_stats.length; j++) {
          const stat = model_stats[j];
          if (j < stat_labels.length) {
            stat_labels[j].set_text(`${model_name} - ${stat.name}: ${stat.loss.data[0]}`);
          } else {
            const label = Label.create(`model_stat_${model_name}_${j}`, {
              text: `${model_name} - ${stat.name}: ${stat.loss.data[0]}`,
              style: stats_label_config,
            });
            this.stats_panel.add_child(label);
            stat_labels.push(label);
          }
        }
        // If there are more labels than stats, remove the extra labels.
        if (stat_labels.length > model_stats.length) {
          for (let j = stat_labels.length - 1; j >= model_stats.length; j--) {
            const label_to_remove = stat_labels[j];
            this.stats_panel.remove_child(label_to_remove);
            stat_labels.pop();
          }
        }
      }
    }
    // Remove label groups for models that are no longer present.
    for (const model_name of this.model_label_map.keys()) {
      if (!current_models.has(model_name)) {
        const labels = this.model_label_map.get(model_name);
        for (const label of labels) {
          this.stats_panel.remove_child(label);
        }
        this.model_label_map.delete(model_name);
      }
    }
  }

  show() {
    super.show();

    this.stats_panel.is_visible = true;

    if (this.scene) {
      this.scene.show_dev_cursor();
    }
  }

  hide() {
    super.hide();

    this.stats_panel.is_visible = false;

    if (this.scene) {
      this.scene.hide_dev_cursor();
    }
  }
}
