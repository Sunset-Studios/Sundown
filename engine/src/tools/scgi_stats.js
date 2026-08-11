import { DevConsoleTool } from "./dev_console_tool.js";
import { Renderer } from "../renderer/renderer.js";
import { InputProvider } from "../input/input_provider.js";
import { InputKey } from "../input/input_types.js";
import { panel, label } from "../ui/2d/immediate.js";

const accent = "#64e6b5";
const accent_soft = "rgba(100, 230, 181, 0.14)";
const secondary_accent = "#ffbd69";
const panel_surface = "rgba(8, 16, 17, 0.94)";
const row_surface = "rgba(255, 255, 255, 0.035)";
const track_surface = "rgba(2, 9, 10, 0.82)";
const body_text = "#def2eb";
const subdued_text = "#829d94";

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
  border: "1px solid rgba(100, 230, 181, 0.24)",
  corner_radius: 9,
  box_shadow: "0 14px 36px rgba(0, 0, 0, 0.48)",
};

function format_number(value) {
  return Number(value).toLocaleString();
}

function format_bytes(value) {
  const bytes = Number(value);
  const mib = bytes / (1024 * 1024);
  return mib >= 1 ? `${mib.toFixed(mib < 10 ? 2 : 1)} MiB` : `${(bytes / 1024).toFixed(1)} KiB`;
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

export class SCGIStats extends DevConsoleTool {
  is_open = false;
  scene = null;
  stats_source = null;

  update(delta_time) {
    if (!this.is_open) return;
    this.render();
  }

  render() {
    const render_strategy = Renderer.get().get_render_strategy();
    const gi_instance = render_strategy ? render_strategy.gi : null;
    this._set_stats_enabled(true, gi_instance);
    const raw_stats =
      gi_instance && typeof gi_instance.get_stats === "function" ? gi_instance.get_stats() : null;
    const stats = raw_stats?.strategy === "scgi" ? raw_stats : null;

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
          label("SCGI  //  SURFACE CACHE", {
            width: 420,
            height: "100%",
            x: 0,
            font: "18px monospace",
            text_color: "#effff9",
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
            background_color: stats ? accent_soft : "rgba(130, 157, 148, 0.1)",
            border: `1px solid ${stats ? accent : subdued_text}`,
            corner_radius: 14,
          });
        }
      );
      label("Hashed surface patches, ray tracing, and filtered radiance storage", {
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
            label("SCGI runtime statistics are not available for the active renderer.", {
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
        "Active patches",
        stats.active_patch_count,
        stats.total_patch_count,
        `${format_number(stats.active_patch_count)} / ${format_number(stats.total_patch_count)}`
      );
      activity_row(
        "Ray workload",
        stats.total_rays_fired,
        stats.active_set_ray_budget,
        `${format_number(stats.total_rays_fired)} / ${format_number(stats.active_set_ray_budget)}`,
        secondary_accent
      );
      metric_pair(
        "Updates / deferred",
        `${format_number(stats.update_patch_count)} / ${format_number(stats.deferred_patch_count)}`,
        "Bootstrap patches",
        `${format_number(stats.bootstrap_patch_count)} / ${format_number(stats.bootstrap_patch_capacity)}`,
        secondary_accent
      );

      section_header("Cache configuration");
      metric_pair(
        "Patch capacity",
        format_number(stats.total_patch_count),
        "Occupancy",
        `${((stats.active_patch_count / Math.max(1, stats.total_patch_count)) * 100).toFixed(1)}%`
      );
      metric_pair(
        "Entry lifetime",
        `${format_number(stats.cache_entry_lifetime)} frames`,
        "Hash probes",
        format_number(stats.hash_search_count)
      );
      metric_pair(
        "Pixel footprint",
        `${Number(stats.cache_pixel_footprint).toFixed(1)}-${(
          Number(stats.cache_pixel_footprint) * Number(stats.history_footprint_max_scale)
        ).toFixed(1)} px`,
        "Normal bias",
        Number(stats.cache_normal_bias).toFixed(4)
      );
      metric_pair(
        "History footprint",
        `${format_number(stats.history_footprint_start_samples)}-${format_number(stats.history_footprint_end_samples)} samples`,
        "Low-history scale",
        `${Number(stats.history_footprint_max_scale).toFixed(1)}x`
      );
      metric_pair(
        "Maximum ray length",
        `${Number(stats.max_ray_length).toFixed(1)} m`,
        "Resolution",
        `${format_number(stats.width)} x ${format_number(stats.height)}`
      );
      metric_pair(
        "Rays / patch",
        format_number(stats.rays_per_patch),
        "Bootstrap rays",
        format_number(stats.bootstrap_rays_per_patch)
      );
      metric_pair(
        "Mature cadence",
        stats.mature_patch_update_period > 1
          ? `1 / ${format_number(stats.mature_patch_update_period)} frames`
          : "Full rate",
        "Scene wake",
        stats.force_full_update ? "Full rate" : "Scheduled",
        stats.force_full_update ? secondary_accent : accent
      );
      metric_pair(
        "History hysteresis",
        Number(stats.history_hysteresis).toFixed(3),
        "Bootstrap ceiling",
        `${(Number(stats.bootstrap_ray_budget_fraction) * 100).toFixed(0)}% of rays`
      );

      section_header("GPU memory");
      metric_pair(
        "Surface patches",
        format_bytes(stats.surface_cache_bytes),
        "Hash table",
        format_bytes(stats.hashmap_bytes)
      );
      metric_pair(
        "SH history",
        format_bytes(stats.sh_bytes),
        "Ray working set",
        format_bytes(stats.ray_working_set_bytes),
        secondary_accent
      );
      metric_pair(
        "Scheduling",
        format_bytes(stats.scheduling_bytes),
        "GI outputs",
        format_bytes(stats.output_bytes)
      );
      metric_pair(
        "Total footprint",
        format_bytes(stats.total_memory_bytes),
        "Maximum ray count",
        format_number(stats.maximum_ray_count),
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

  _set_stats_enabled(enabled, gi_instance = null) {
    if (!gi_instance) {
      const render_strategy = Renderer.get().get_render_strategy();
      gi_instance = render_strategy ? render_strategy.gi : null;
    }

    if (this.stats_source && (!enabled || this.stats_source !== gi_instance)) {
      this.stats_source.set_stats_enabled(false);
      this.stats_source = null;
    }

    if (enabled && typeof gi_instance?.set_stats_enabled === "function") {
      gi_instance.set_stats_enabled(true);
      this.stats_source = gi_instance;
    }
  }

  toggle() {
    this.is_open = !this.is_open;
    this._set_stats_enabled(this.is_open);
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
    this._set_stats_enabled(true);
    if (this.scene && typeof this.scene.show_dev_cursor === "function") {
      this.scene.show_dev_cursor();
    }
  }

  hide() {
    this.is_open = false;
    this._set_stats_enabled(false);
    if (this.scene && typeof this.scene.hide_dev_cursor === "function") {
      this.scene.hide_dev_cursor();
    }
  }

  set_scene(scene) {
    this.scene = scene;
  }
}
