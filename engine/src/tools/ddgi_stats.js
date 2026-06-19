import { DevConsoleTool } from "./dev_console_tool.js";
import { Renderer } from "../renderer/renderer.js";
import { InputProvider } from "../input/input_provider.js";
import { InputKey } from "../input/input_types.js";
import { panel, label } from "../ui/2d/immediate.js";

const stats_panel_config = {
  layout: "column",
  gap: 4,
  y: 25,
  x: 25,
  anchor_x: "right",
  dont_consume_cursor_events: true,
  background_color: "rgba(0, 0, 0, 0.7)",
  width: 520,
  padding: 10,
  border: "1px solid rgb(68, 68, 68)",
  corner_radius: 5,
};

const stats_label_config = {
  text_color: "#fff",
  wrap: true,
  font: "16px monospace",
  width: "100%",
  height: "fit-content",
  text_valign: "middle",
  text_align: "left",
  text_padding: 5,
};

const stats_label_config_small = {
  ...stats_label_config,
  width: "fit-content",
  x: 0,
};

const value_label_config = {
  ...stats_label_config,
  text_color: "#4eaaff",
  width: "fit-content",
  x: 0,
};

function format_number(value) {
  return Number(value).toLocaleString();
}

function stat_row(label_text, value_text, value_config = value_label_config) {
  panel({ layout: "row", gap: 4, width: "100%", height: 25, anchor_x: "left", x: 0 }, () => {
    label(`${label_text}:`, stats_label_config_small);
    label(value_text, value_config);
  });
}

export class DDGIStats extends DevConsoleTool {
  is_open = false;
  scene = null;

  update(delta_time) {
    if (!this.is_open) return;
    this.render();
  }

  render() {
    const render_strategy = Renderer.get().get_render_strategy();
    const gi_instance = render_strategy ? render_strategy.gi : null;
    const stats = gi_instance && typeof gi_instance.get_stats === "function"
      ? gi_instance.get_stats()
      : null;

    const panel_state = panel(stats_panel_config, () => {
      label("DDGI Stats", { ...stats_label_config, font: "18px monospace", text_color: "#0ff" });
      label("--------------------------------", stats_label_config);

      if (!stats) {
        label("DDGI stats unavailable.", stats_label_config);
        return;
      }

      stat_row("Active probes (total)", format_number(stats.active_probe_count));
      stat_row("Active probes (nonculled)", format_number(stats.active_probe_count_nonculled));
      stat_row("Active probes (culled)", format_number(stats.active_probe_count_culled));
      stat_row("Probe update count", format_number(stats.probe_update_count));
      stat_row("Total probes", format_number(stats.total_probe_count));
      stat_row("Probes per frame", format_number(stats.probes_per_frame));
      stat_row("Rays per probe", format_number(stats.max_rays_per_probe));
      stat_row("Total rays fired", format_number(stats.total_rays_fired));
      stat_row(
        "Probe grid dims",
        `${format_number(stats.probe_grid_dims[0])} x ${format_number(stats.probe_grid_dims[1])} x ${format_number(stats.probe_grid_dims[2])}`
      );
      stat_row("Cascades", format_number(stats.cascade_count));
      stat_row("Probe spacing", stats.probe_spacing.toFixed(2));
      stat_row("Probe radius", stats.probe_radius.toFixed(2));
      stat_row("Light count", format_number(stats.light_count));

      label("--------------------------------", stats_label_config);
    });

    if (this.is_open && InputProvider.get_action(InputKey.B_mouse_left)) {
      if (!panel_state.hovered) {
        this.hide();
      }
      InputProvider.consume_action(InputKey.B_mouse_left);
    }
  }

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
