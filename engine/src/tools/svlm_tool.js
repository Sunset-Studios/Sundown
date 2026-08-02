import { CVarSystem } from "../core/cvar_system.js";
import { Renderer } from "../renderer/renderer.js";
import { DebugDrawType, GIStrategyType } from "../renderer/renderer_types.js";
import { EngineCVars } from "../../config/cvars.js";
import { InputProvider } from "../input/input_provider.js";
import { InputKey } from "../input/input_types.js";
import { panel, label } from "../ui/2d/immediate.js";
import { bytes_to_mb } from "../utility/math.js";
import { error, log, format_number } from "../utility/logging.js";
import { DevConsoleTool } from "./dev_console_tool.js";

const accent = "#72e3c2";
const accent_soft = "rgba(114, 227, 194, 0.16)";
const secondary_accent = "#8ea7ff";
const panel_surface = "rgba(8, 13, 18, 0.94)";
const row_surface = "rgba(255, 255, 255, 0.035)";
const track_surface = "rgba(2, 7, 11, 0.82)";
const body_text = "#d8e5e8";
const subdued_text = "#81959b";
const warning = "#ffca72";
const danger = "#ff7d8c";

const stats_panel_config = {
  layout: "column",
  gap: 6,
  y: 24,
  x: 24,
  anchor_x: "right",
  dont_consume_cursor_events: true,
  background_color: panel_surface,
  width: 640,
  padding: 14,
  border: "1px solid rgba(114, 227, 194, 0.24)",
  corner_radius: 9,
  box_shadow: "0 14px 36px rgba(0, 0, 0, 0.48)",
};

const stats_label_config = {
  text_color: body_text,
  wrap: true,
  font: "13px monospace",
  width: "100%",
  height: "fit-content",
  text_valign: "middle",
  text_align: "left",
  text_padding: 4,
};

function metric_card(label_text, value_text) {
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
        width: 120,
        height: "100%",
        x: 0,
        font: "11px monospace",
        text_color: subdued_text,
        text_align: "left",
        text_valign: "middle",
      });
      label(value_text, {
        width: 160,
        height: "100%",
        x: 0,
        font: "11px monospace",
        text_color: accent,
        text_align: "right",
        text_valign: "middle",
      });
    }
  );
}

function metric_pair(left_label, left_value, right_label, right_value) {
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
      metric_card(right_label, right_value);
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

function progress_row(label_text, progress, detail_text, color = accent) {
  const determinate = Number.isFinite(progress);
  const clamped_progress = determinate ? Math.max(0, Math.min(1, progress)) : 0;
  const pulse = ((Date.now() % 1800) / 1800) * 0.72;
  const fill_width = determinate ? `${clamped_progress * 100}%` : "28%";
  const fill_x = determinate ? 0 : `${pulse * 100}%`;

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
            width: fill_width,
            height: "100%",
            x: fill_x,
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

function format_eta(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "Estimating ETA";

  const remaining_seconds = Math.max(0, Math.ceil(seconds));
  if (remaining_seconds < 60) return `ETA ${remaining_seconds}s`;

  const remaining_minutes = Math.floor(remaining_seconds / 60);
  const seconds_part = remaining_seconds % 60;
  if (remaining_minutes < 60) {
    return `ETA ${remaining_minutes}m ${seconds_part}s`;
  }

  const remaining_hours = Math.floor(remaining_minutes / 60);
  const minutes_part = remaining_minutes % 60;
  return `ETA ${remaining_hours}h ${minutes_part}m`;
}

function parse_bake_options(args) {
  const options = {};

  for (let i = 0; i < args.length; i += 1) {
    const raw = args[i];
    const eq = raw.indexOf("=");
    if (eq === -1) continue;

    const key = raw.slice(0, eq).trim().toLowerCase();
    const value = raw.slice(eq + 1).trim();
    if (key === "root" || key === "root_brick_size") {
      options.root_brick_size = Number(value);
    } else if (key === "max" || key === "max_level") {
      options.max_level = Number(value);
    } else if (key === "min" || key === "min_level") {
      options.min_level = Number(value);
    } else if (key === "max_nodes") {
      options.max_nodes = Number(value);
    } else if (key === "rays" || key === "irradiance_rays_per_probe") {
      options.irradiance_rays_per_probe = Number(value);
    } else if (key === "batch" || key === "irradiance_probes_per_batch") {
      options.irradiance_probes_per_batch = Number(value);
    } else if (key === "samples" || key === "irradiance_sample_count") {
      options.irradiance_sample_count = Number(value);
    } else if (key === "tile" || key === "tile_size" || key === "world_tile_size") {
      options.world_tile_size = Number(value);
    } else if (key === "radius" || key === "streaming_radius") {
      options.streaming_radius = Number(value);
    } else if (key === "hysteresis" || key === "streaming_hysteresis") {
      options.streaming_hysteresis = Number(value);
    } else if (key === "prefetch" || key === "streaming_prefetch_tiles") {
      options.streaming_prefetch_tiles = Number(value);
    } else if (key === "requests" || key === "streaming_max_requests") {
      options.streaming_max_requests = Number(value);
    } else if (key === "budget" || key === "streaming_memory_budget_mb") {
      options.streaming_memory_budget_mb = Number(value);
    } else if (key === "transition" || key === "streaming_transition_tiles") {
      options.streaming_transition_tiles = Number(value);
    } else if (key === "coarse_min" || key === "coarse_min_lod") {
      options.coarse_min_lod = Number(value);
    } else if (key === "coarse_max" || key === "coarse_max_lod") {
      options.coarse_max_lod = Number(value);
    } else if (key === "coarse_budget" || key === "coarse_memory_budget_mb") {
      options.coarse_memory_budget_mb = Number(value);
    }
  }

  return options;
}

function parse_debug_options(args) {
  const options = {};

  let mode = "on";
  let debug_view = DebugDrawType.SVLM_Bricks;

  for (const raw_arg of args) {
    const raw = String(raw_arg || "").trim();
    if (!raw) continue;

    const lower = raw.toLowerCase();
    if (lower === "on" || lower === "off") {
      mode = lower;
    } else if (lower === "brick" || lower === "bricks") {
      debug_view = DebugDrawType.SVLM_Bricks;
    } else if (lower === "probe" || lower === "probes") {
      debug_view = DebugDrawType.SVLM_Probes;
    } else if (lower === "all") {
      options.debug_level = -1;
    } else {
      const positional_level = Number(lower);
      if (Number.isFinite(positional_level)) {
        options.debug_level = positional_level;
      }
    }
  }

  return { mode, options, debug_view };
}

// Dev-console control plane for the GPU SVLM hierarchy and irradiance bake.
export class SVLMTool extends DevConsoleTool {
  is_open = false;
  scene = null;
  bake_save_serial = 0;
  bake_status = "idle";
  bake_eta_started_at = 0;
  bake_eta_completed_at_start = 0;
  bake_eta_required_probe_samples = 0;
  bake_eta_last_completed_probe_samples = 0;
  bake_eta_seconds = Number.NaN;
  bake_eta_updated_at = 0;

  update(delta_time) {
    if (!this.is_open) return;
    this.render();
  }

  execute(args = []) {
    const command = (args[0] || "stats").toLowerCase();
    const strategy = Renderer.get().get_render_strategy();
    const svlm = strategy?.svlm || null;
    if (!svlm) return;

    switch (command) {
      case "bake":
        const bake_options = parse_bake_options(args.slice(1));
        const bake_save_serial = ++this.bake_save_serial;
        if (!this.scene) {
          error("SVLM cannot bake and save without an active scene.");
          break;
        }
        this.bake_status = "loading";
        this._reset_bake_eta();
        this.show();
        void this._bake_and_save(svlm, bake_options, this.scene, bake_save_serial);
        break;
      case "debug":
        const debug_options = parse_debug_options(args.slice(1));
        const current_debug_view = CVarSystem.get(
          EngineCVars.Renderer.DebugDraw,
          DebugDrawType.None
        );
        const enable =
          debug_options.mode === "on" || current_debug_view !== debug_options.debug_view;
        CVarSystem.set(
          EngineCVars.Renderer.DebugDraw,
          enable ? debug_options.debug_view : DebugDrawType.None,
          { source: "svlm" }
        );
        break;
      case "preview":
        CVarSystem.set(EngineCVars.Renderer.GIStrategy, GIStrategyType.SVLM, {
          source: "svlm",
        });
        break;
      case "clear":
        this.bake_save_serial++;
        this.bake_status = "idle";
        this._reset_bake_eta();
        svlm.clear();
        break;
      case "stats":
        this.show();
        break;
      case "hide":
        this.hide();
        break;
      default:
        log(
          "svlm [stats | bake [root=<size>] [max=<level>] [min=<level>] [rays=<count>] [batch=<count>] [samples=<count>] [tile=<meters>] [radius=<tiles>] [hysteresis=<tiles>] [prefetch=<tiles>] [requests=<count>] [budget=<mb>] [transition=<tiles>] [coarse_min=<lod>] [coarse_max=<lod>] [coarse_budget=<mb>] | preview | debug [bricks|probes] [on|off] [level=<n>|all] | clear | hide]"
        );
        break;
    }
  }

  async _bake_and_save(svlm, bake_options, scene, bake_save_serial) {
    try {
      await scene.when_scene_data_ready();
      if (bake_save_serial !== this.bake_save_serial || scene !== this.scene) {
        return;
      }

      svlm.bake(bake_options);
      this.bake_status = "baking";
      this._reset_bake_eta();
      await svlm.serialize_bake_tiles();
      if (bake_save_serial !== this.bake_save_serial) return;

      this.bake_status = "saving";
      const result = await scene.save_scene_data();
      if (bake_save_serial !== this.bake_save_serial) return;

      this.bake_status = "ready";
      log(
        `SVLM bake completed and saved to '${result.asset_path}' (${format_number(
          result.byte_length
        )} bytes).`
      );
    } catch (save_error) {
      if (bake_save_serial !== this.bake_save_serial) return;
      this.bake_status = "error";
      error("SVLM bake or scene-data save failed:", save_error);
    }
  }

  _reset_bake_eta() {
    this.bake_eta_started_at = 0;
    this.bake_eta_completed_at_start = 0;
    this.bake_eta_required_probe_samples = 0;
    this.bake_eta_last_completed_probe_samples = 0;
    this.bake_eta_seconds = Number.NaN;
    this.bake_eta_updated_at = 0;
  }

  _estimate_bake_eta(stats) {
    const required_probe_samples = Math.max(
      0,
      Number(stats.irradiance_required_probe_samples) || 0
    );
    const completed_probe_samples = Math.max(
      0,
      Math.min(
        required_probe_samples,
        Number(stats.irradiance_completed_probe_samples) || 0
      )
    );

    if (required_probe_samples <= 0 || completed_probe_samples >= required_probe_samples) {
      return completed_probe_samples >= required_probe_samples && required_probe_samples > 0
        ? 0
        : Number.NaN;
    }

    const now = performance.now();
    const workload_changed =
      this.bake_eta_required_probe_samples !== required_probe_samples ||
      completed_probe_samples < this.bake_eta_completed_at_start;
    if (this.bake_eta_started_at <= 0 || workload_changed) {
      this.bake_eta_started_at = now;
      this.bake_eta_completed_at_start = completed_probe_samples;
      this.bake_eta_required_probe_samples = required_probe_samples;
      this.bake_eta_last_completed_probe_samples = completed_probe_samples;
      this.bake_eta_seconds = Number.NaN;
      this.bake_eta_updated_at = 0;
      return Number.NaN;
    }

    const elapsed_seconds = (now - this.bake_eta_started_at) / 1000;
    const completed_since_start =
      completed_probe_samples - this.bake_eta_completed_at_start;
    if (elapsed_seconds < 1 || completed_since_start <= 0) {
      return Number.NaN;
    }

    if (completed_probe_samples > this.bake_eta_last_completed_probe_samples) {
      const probe_samples_per_second = completed_since_start / elapsed_seconds;
      const measured_eta_seconds =
        (required_probe_samples - completed_probe_samples) / probe_samples_per_second;
      const previous_eta_seconds = Number.isFinite(this.bake_eta_seconds)
        ? Math.max(0, this.bake_eta_seconds - (now - this.bake_eta_updated_at) / 1000)
        : Number.NaN;
      this.bake_eta_seconds = Number.isFinite(previous_eta_seconds)
        ? previous_eta_seconds * 0.75 + measured_eta_seconds * 0.25
        : measured_eta_seconds;
      this.bake_eta_last_completed_probe_samples = completed_probe_samples;
      this.bake_eta_updated_at = now;
    }

    if (!Number.isFinite(this.bake_eta_seconds)) return Number.NaN;
    return Math.max(0, this.bake_eta_seconds - (now - this.bake_eta_updated_at) / 1000);
  }

  render() {
    const strategy = Renderer.get().get_render_strategy();
    const svlm = strategy?.svlm || null;
    if (!svlm) return;

    const stats = svlm.get_stats();
    const has_bake_data = stats.baked || stats.tile_streaming_enabled;
    const bake_active =
      this.bake_status === "loading" ||
      this.bake_status === "baking" ||
      this.bake_status === "saving";
    const has_error = this.bake_status === "error" || !!stats.tile_serialization_error;
    const status_text = has_error
      ? "FAILED"
      : bake_active
        ? this.bake_status.toUpperCase()
        : has_bake_data || this.bake_status === "ready"
          ? "READY"
          : "IDLE";
    const status_color = has_error
      ? danger
      : bake_active
        ? warning
        : has_bake_data || this.bake_status === "ready"
          ? accent
          : subdued_text;
    const status_background = has_error
      ? "rgba(255, 125, 140, 0.12)"
      : bake_active
        ? "rgba(255, 202, 114, 0.12)"
        : has_bake_data || this.bake_status === "ready"
          ? accent_soft
          : "rgba(129, 149, 155, 0.1)";

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
          label("SVLM  //  BAKED GI", {
            width: 420,
            height: "100%",
            x: 0,
            font: "18px monospace",
            text_color: "#ecfbf7",
            text_align: "left",
            text_valign: "middle",
            text_padding: 4,
          });
          label(status_text, {
            width: 156,
            height: 28,
            x: 0,
            font: "11px monospace",
            text_color: status_color,
            text_align: "center",
            text_valign: "middle",
            background_color: status_background,
            border: `1px solid ${status_color}`,
            corner_radius: 14,
          });
        }
      );
      label("Sparse volumetric lightmap bake and runtime tile residency", {
        width: "100%",
        height: 20,
        x: 0,
        font: "11px monospace",
        text_color: subdued_text,
        text_align: "left",
        text_valign: "middle",
        text_padding: 4,
      });

      if (bake_active) {
        section_header("Bake progress");
        if (this.bake_status === "loading") {
          progress_row("Bake", null, "Preparing scene data", secondary_accent);
        } else if (this.bake_status === "saving") {
          progress_row("Bake", 1, "Complete · Saving package", accent);
        } else if ((stats.irradiance_required_probe_samples || 0) <= 0) {
          progress_row("Bake", null, "Preparing bake workload", secondary_accent);
        } else {
          const bake_progress = Math.max(0, Math.min(1, stats.irradiance_progress || 0));
          const eta_text = format_eta(this._estimate_bake_eta(stats));
          progress_row(
            "Bake",
            bake_progress,
            `${(bake_progress * 100).toFixed(1)}% · ${eta_text}`,
            accent
          );
        }
      }

      if (!has_bake_data) {
        if (!bake_active) {
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
              label("No SVLM bake is loaded. Run `svlm bake` to create one.", {
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
          if (stats.tile_serialization_error) {
            label(`Bake error: ${stats.tile_serialization_error}`, {
              ...stats_label_config,
              text_color: danger,
            });
          }
        }
        return;
      }

      section_header("Streaming");
      metric_pair(
        "World tile size",
        `${Number(stats.config.world_tile_size).toFixed(1)} m`,
        "Streaming radius",
        `${format_number(stats.config.streaming_radius)} tiles / ${(
          stats.config.streaming_radius * stats.config.world_tile_size
        ).toFixed(1)} m`
      );
      metric_pair(
        "Tile residency",
        `${format_number(
          stats.resident_tile_count
        )} / ${format_number(stats.serialized_tile_count)}`,
        "Pending requests",
        format_number(stats.requested_tile_count)
      );
      metric_pair(
        "Fine GPU pool",
        `${bytes_to_mb(stats.streamed_gpu_bytes)} / ${bytes_to_mb(
          stats.streaming_memory_budget_bytes
        )} MB`,
        "Coarse hierarchy",
        `${format_number(stats.coarse_record_count)} records / ${bytes_to_mb(
          stats.coarse_bytes
        )} MB`
      );
      metric_pair(
        "Retention ring",
        `${format_number(stats.config.streaming_hysteresis)} tiles`,
        "Motion prefetch",
        `${format_number(stats.config.streaming_prefetch_tiles)} tiles`
      );

      section_header("Bake output");
      metric_pair(
        "Bake serial",
        format_number(stats.bake_serial),
        "Nodes",
        format_number(stats.node_count)
      );
      metric_pair(
        "Leaf bricks",
        format_number(stats.leaf_count),
        "Allocated probes",
        format_number(stats.probe_count)
      );
      metric_pair(
        "Allocation budget",
        `${format_number(stats.config.max_nodes)} records`,
        "Rays / probe",
        format_number(stats.irradiance_rays_per_probe)
      );

      section_header("Volume");
      metric_pair(
        "Root dimensions",
        `${stats.root_dims[0]} x ${stats.root_dims[1]} x ${stats.root_dims[2]}`,
        "Root brick size",
        Number(stats.root_brick_size).toFixed(2)
      );
      metric_pair(
        "Forced min level",
        `L${stats.min_level}`,
        "Reached / maximum",
        `L${stats.max_level_reached} / L${stats.max_level}`
      );

      section_header("Memory");
      metric_pair(
        "Peak bake GPU",
        `${bytes_to_mb(stats.total_bytes)} MB`,
        "Tiled irradiance",
        `${bytes_to_mb(stats.irradiance_bytes)} MB serialized`
      );
      metric_pair(
        "Peak tile alloc.",
        `${bytes_to_mb(stats.irradiance_allocated_bytes)} MB`,
        "Irradiance sets",
        format_number(stats.irradiance_sample_count)
      );

      if (stats.truncated_by_node_limit || stats.truncated_by_leaf_limit) {
        label("Bake hit an SVLM allocation limit.", {
          ...stats_label_config,
          text_color: warning,
        });
      }
      if (stats.irradiance_capacity_exceeded) {
        label("Baked irradiance exceeds the storage-buffer limit.", {
          ...stats_label_config,
          text_color: danger,
        });
      }
      if (stats.tile_serialization_error) {
        label(`Bake error: ${stats.tile_serialization_error}`, {
          ...stats_label_config,
          text_color: danger,
        });
      }
    });

    if (this.is_open && InputProvider.get_action(InputKey.B_mouse_left)) {
      if (!panel_state.hovered) {
        this.hide();
      }
    }
  }

  toggle() {
    this.is_open = !this.is_open;
  }

  show() {
    this.is_open = true;
  }

  hide() {
    this.is_open = false;
  }

  set_scene(scene) {
    if (scene !== this.scene) {
      this.bake_save_serial++;
      this.bake_status = "idle";
      this._reset_bake_eta();
    }
    this.scene = scene;
  }
}
