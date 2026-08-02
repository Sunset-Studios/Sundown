import { DevConsoleTool } from "./dev_console_tool.js";
import { Renderer } from "../renderer/renderer.js";
import { InputProvider } from "../input/input_provider.js";
import { InputKey } from "../input/input_types.js";
import { panel, label } from "../ui/2d/immediate.js";

const accent = "#69d5ff";
const accent_soft = "rgba(105, 213, 255, 0.14)";
const secondary_accent = "#a99cff";
const panel_surface = "rgba(8, 13, 20, 0.94)";
const row_surface = "rgba(255, 255, 255, 0.035)";
const track_surface = "rgba(2, 7, 13, 0.82)";
const body_text = "#dce9f0";
const subdued_text = "#8297a3";

const stats_panel_config = {
  layout: "column",
  gap: 6,
  y: 24,
  x: 24,
  anchor_x: "left",
  dont_consume_cursor_events: true,
  background_color: panel_surface,
  width: 640,
  padding: 14,
  border: "1px solid rgba(105, 213, 255, 0.24)",
  corner_radius: 9,
  box_shadow: "0 14px 36px rgba(0, 0, 0, 0.48)",
};

function format_number(value) {
  return Number(value).toLocaleString();
}

function format_bytes(value) {
  const mib = Number(value) / (1024 * 1024);
  return `${mib.toFixed(mib < 10 ? 2 : 1)} MiB`;
}

function metric_card(label_text, value_text, value_color = accent) {
  panel(
    {
      layout: "row",
      gap: 6,
      width: 302,
      height: 31,
      x: 0,
      padding_left: 8,
      padding_right: 8,
      background_color: row_surface,
      corner_radius: 4,
    },
    () => {
      label(label_text, {
        width: 132,
        height: "100%",
        x: 0,
        font: "11px monospace",
        text_color: subdued_text,
        text_align: "left",
        text_valign: "middle",
      });
      label(value_text, {
        width: 148,
        height: "100%",
        x: 0,
        font: "11px monospace",
        text_color: value_color,
        text_align: "right",
        text_valign: "middle",
      });
    }
  );
}

function metric_pair(left_label, left_value, right_label, right_value, right_color = accent) {
  panel(
    {
      layout: "row",
      gap: 8,
      width: "100%",
      height: 31,
      x: 0,
    },
    () => {
      metric_card(left_label, left_value);
      metric_card(right_label, right_value, right_color);
    }
  );
}

function section_header(title) {
  label(title.toUpperCase(), {
    width: "100%",
    height: 21,
    x: 0,
    font: "11px monospace",
    text_color: accent,
    text_align: "left",
    text_valign: "middle",
    text_padding: 3,
  });
}

function activity_row(label_text, current, maximum, detail_text, color = accent) {
  const safe_maximum = Math.max(0, Number(maximum) || 0);
  const progress = safe_maximum > 0 ? Math.max(0, Math.min(1, current / safe_maximum)) : 0;

  panel(
    {
      layout: "row",
      gap: 8,
      width: "100%",
      height: 31,
      x: 0,
      padding_left: 8,
      padding_right: 8,
      background_color: row_surface,
      corner_radius: 5,
    },
    () => {
      label(label_text, {
        width: 108,
        height: "100%",
        x: 0,
        font: "12px monospace",
        text_color: body_text,
        text_align: "left",
        text_valign: "middle",
      });
      panel(
        {
          width: 260,
          height: 11,
          x: 0,
          background_color: track_surface,
          border: "1px solid rgba(255, 255, 255, 0.08)",
          corner_radius: 5,
          clip: true,
        },
        () => {
          panel({
            width: `${progress * 100}%`,
            height: "100%",
            x: 0,
            y: 0,
            background_color: color,
            corner_radius: 5,
            box_shadow: `0 0 10px ${color}`,
          });
        }
      );
      label(detail_text, {
        width: 204,
        height: "100%",
        x: 0,
        font: "11px monospace",
        text_color: color,
        text_align: "right",
        text_valign: "middle",
      });
    }
  );
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
    const stats =
      gi_instance && typeof gi_instance.get_stats === "function" ? gi_instance.get_stats() : null;

    const panel_state = panel(stats_panel_config, () => {
      panel(
        {
          layout: "row",
          gap: 8,
          width: "100%",
          height: 40,
          x: 0,
        },
        () => {
          label("DDGI  //  RADIANCE CACHE", {
            width: 420,
            height: "100%",
            x: 0,
            font: "18px monospace",
            text_color: "#edfaff",
            text_align: "left",
            text_valign: "middle",
            text_padding: 4,
          });
          label(stats ? "LIVE" : "UNAVAILABLE", {
            width: 156,
            height: 28,
            x: 0,
            font: "11px monospace",
            text_color: stats ? accent : subdued_text,
            text_align: "center",
            text_valign: "middle",
            background_color: stats ? accent_soft : "rgba(130, 151, 163, 0.1)",
            border: `1px solid ${stats ? accent : subdued_text}`,
            corner_radius: 14,
          });
        }
      );
      label("Dynamic diffuse global illumination and sparse probe depth storage", {
        width: "100%",
        height: 20,
        x: 0,
        font: "11px monospace",
        text_color: subdued_text,
        text_align: "left",
        text_valign: "middle",
        text_padding: 4,
      });

      if (!stats) {
        panel(
          {
            width: "100%",
            height: 52,
            x: 0,
            background_color: row_surface,
            border: "1px solid rgba(255, 255, 255, 0.06)",
            corner_radius: 5,
          },
          () => {
            label("DDGI runtime statistics are not available for the active renderer.", {
              width: "100%",
              height: "100%",
              x: 0,
              y: 0,
              font: "12px monospace",
              text_color: subdued_text,
              text_align: "center",
              text_valign: "middle",
            });
          }
        );
        return;
      }

      section_header("Runtime activity");
      activity_row(
        "Active probes",
        stats.active_probe_count,
        stats.total_probe_count,
        `${format_number(stats.active_probe_count)} / ${format_number(stats.total_probe_count)}`
      );
      activity_row(
        "Frame updates",
        stats.probe_update_count,
        stats.probes_per_frame,
        `${format_number(stats.probe_update_count)} / ${format_number(stats.probes_per_frame)}`,
        secondary_accent
      );
      metric_pair(
        "Rays / probe",
        format_number(stats.max_rays_per_probe),
        "Rays fired",
        format_number(stats.total_rays_fired),
        secondary_accent
      );

      section_header("Probe volume");
      metric_pair(
        "Grid dimensions",
        `${format_number(stats.probe_grid_dims[0])} x ${format_number(stats.probe_grid_dims[1])} x ${format_number(stats.probe_grid_dims[2])}`,
        "Cascades",
        format_number(stats.cascade_count)
      );
      metric_pair(
        "Total probes",
        format_number(stats.total_probe_count),
        "Probes / cascade",
        format_number(stats.probes_per_cascade)
      );
      metric_pair(
        "Probe spacing",
        `${stats.probe_spacing.toFixed(2)} m`,
        "Probe radius",
        `${stats.probe_radius.toFixed(2)} m`
      );
      metric_pair(
        "Frame budget",
        `${format_number(stats.probes_per_frame)} probes`,
        "Maximum ray length",
        `${stats.max_ray_length.toFixed(2)} m`
      );

      section_header("Scene inputs");
      metric_pair(
        "Lights",
        format_number(stats.light_count),
        "Depth slots",
        format_number(stats.depth_slot_count)
      );
      metric_pair(
        "Slot retention",
        `${format_number(stats.depth_slot_retention_frames)} frames`,
        "Sparse slot coverage",
        `${((stats.depth_slot_count / Math.max(1, stats.total_probe_count)) * 100).toFixed(1)}%`,
        secondary_accent
      );

      section_header("Depth storage");
      metric_pair(
        "Sparse moments",
        format_bytes(stats.depth_sparse_bytes),
        "Slot metadata",
        format_bytes(stats.depth_sparse_metadata_bytes)
      );
      metric_pair(
        "Dense packed",
        format_bytes(stats.depth_dense_packed_bytes),
        "Previous layout",
        format_bytes(stats.depth_dense_previous_bytes),
        secondary_accent
      );
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
