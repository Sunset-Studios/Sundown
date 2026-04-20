import { CVarSystem } from "../core/cvar_system.js";
import { Renderer } from "../renderer/renderer.js";
import { DebugDrawType } from "../renderer/renderer_types.js";
import { EngineCVars } from "../../config/cvars.js";
import { InputProvider } from "../input/input_provider.js";
import { InputKey } from "../input/input_types.js";
import { panel, label } from "../ui/2d/immediate.js";
import { bytes_to_mb } from "../utility/math.js";
import { log, warn } from "../utility/logging.js";
import { DevConsoleTool } from "./dev_console_tool.js";

const stats_panel_config = {
  layout: "column",
  gap: 4,
  y: 25,
  x: 25,
  anchor_x: "right",
  dont_consume_cursor_events: true,
  background_color: "rgba(0, 0, 0, 0.72)",
  width: 560,
  padding: 10,
  border: "1px solid rgb(68, 68, 68)",
  corner_radius: 5,
};

const stats_label_config = {
  text_color: "#fff",
  wrap: true,
  font: "15px monospace",
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
  text_color: "#75e0b8",
  width: "fit-content",
  x: 0,
};

function format_number(value) {
  return Number(value || 0).toLocaleString();
}

function format_vec3(v) {
  if (!v) {
    return "0.00, 0.00, 0.00";
  }
  return `${Number(v[0]).toFixed(2)}, ${Number(v[1]).toFixed(2)}, ${Number(v[2]).toFixed(2)}`;
}

function stat_row(label_text, value_text, value_config = value_label_config) {
  panel({ layout: "row", gap: 4, width: "100%", height: 25, anchor_x: "left", x: 0 }, () => {
    label(`${label_text}:`, stats_label_config_small);
    label(value_text, value_config);
  });
}

function get_svlm() {
  const strategy = Renderer.get().get_render_strategy();
  return strategy?.svlm || null;
}

function parse_bake_options(args) {
  const options = {};
  const positional = [];

  for (let i = 0; i < args.length; i += 1) {
    const raw = args[i];
    const eq = raw.indexOf("=");
    if (eq === -1) {
      positional.push(raw);
      continue;
    }

    const key = raw.slice(0, eq).trim().toLowerCase();
    const value = raw.slice(eq + 1).trim();
    if (key === "root" || key === "root_size" || key === "root_brick_size") {
      options.root_brick_size = Number(value);
    } else if (key === "max" || key === "max_level") {
      options.max_level = Number(value);
    } else if (key === "min" || key === "min_level") {
      options.min_level = Number(value);
    } else if (key === "leaves" || key === "max_leaf_bricks") {
      options.max_leaf_bricks = Number(value);
    } else if (key === "nodes" || key === "max_nodes") {
      options.max_nodes = Number(value);
    } else if (key === "debug_leaves" || key === "max_debug_leaf_bricks") {
      options.max_debug_leaf_bricks = Number(value);
    }
  }

  if (positional.length > 0) {
    options.root_brick_size = Number(positional[0]);
  }
  if (positional.length > 1) {
    options.max_level = Number(positional[1]);
  }
  if (positional.length > 2) {
    options.min_level = Number(positional[2]);
  }

  return options;
}

function parse_debug_options(args) {
  const options = {};
  let mode = "toggle";
  let debug_view = DebugDrawType.SVLM_Bricks;
  let saw_option = false;

  for (const raw_arg of args) {
    const raw = String(raw_arg || "").trim();
    if (!raw) {
      continue;
    }

    const lower = raw.toLowerCase();
    if (lower === "on" || lower === "off" || lower === "toggle") {
      mode = lower;
      continue;
    }

    if (lower === "brick" || lower === "bricks" || lower === "box" || lower === "boxes") {
      debug_view = DebugDrawType.SVLM_Bricks;
      continue;
    }

    if (lower === "probe" || lower === "probes" || lower === "sphere" || lower === "spheres") {
      debug_view = DebugDrawType.SVLM_Probes;
      continue;
    }

    if (lower === "all" || lower === "any" || lower === "levels") {
      options.debug_level = -1;
      saw_option = true;
      continue;
    }

    if (lower === "next" || lower === "+") {
      options.debug_level_delta = 1;
      saw_option = true;
      continue;
    }

    if (lower === "prev" || lower === "previous" || lower === "-") {
      options.debug_level_delta = -1;
      saw_option = true;
      continue;
    }

    const eq = lower.indexOf("=");
    if (eq !== -1) {
      const key = lower.slice(0, eq).trim();
      const value = lower.slice(eq + 1).trim();
      if (key === "level" || key === "debug_level" || key === "l") {
        options.debug_level = value === "all" || value === "any" ? -1 : Number(value);
        saw_option = true;
      } else if (key === "radius" || key === "probe_radius" || key === "debug_probe_radius") {
        options.debug_probe_radius = Number(value);
        saw_option = true;
      }
      continue;
    }

    const positional_level = Number(lower);
    if (Number.isFinite(positional_level)) {
      options.debug_level = positional_level;
      saw_option = true;
    }
  }

  if (saw_option && mode === "toggle") {
    mode = "on";
  }

  return { mode, options, debug_view };
}

// Dev-console control plane for the GPU SVLM builder. This intentionally keeps
// bake/debug/stats in one tool so iteration on brick allocation is quick while
// irradiance accumulation is still future work.
export class SVLMTool extends DevConsoleTool {
  is_open = false;
  scene = null;

  update(delta_time) {
    if (!this.is_open) return;
    this.render();
  }

  execute(args = []) {
    const command = (args[0] || "stats").toLowerCase();
    const svlm = get_svlm();

    if (!svlm) {
      warn("SVLM is unavailable on the active render strategy.");
      return;
    }

    switch (command) {
      case "bake":
        const stats = svlm.bake(parse_bake_options(args.slice(1)));
        this.show();
        log("SVLM GPU bake queued. Stats will update after the next rendered frame.");
        if (stats.message) {
          warn(stats.message);
        }
        break;
      case "debug":
        const parsed = parse_debug_options(args.slice(1));
        if (parsed.options.debug_level_delta !== undefined) {
          const stats = svlm.get_stats();
          const current_level = stats.debug_level ?? -1;
          const base_level = current_level < 0 ? 0 : current_level;
          parsed.options.debug_level = base_level + parsed.options.debug_level_delta;
          delete parsed.options.debug_level_delta;
        }
        const debug_stats = svlm.set_debug_options(parsed.options);
        const current = CVarSystem.get(EngineCVars.Renderer.DebugDraw, DebugDrawType.None);
        const enable = parsed.mode === "on" || (parsed.mode === "toggle" && current !== parsed.debug_view);
        CVarSystem.set(
          EngineCVars.Renderer.DebugDraw,
          enable ? parsed.debug_view : DebugDrawType.None,
          { source: "svlm" }
        );
        if (enable) {
          // The level filter is shared by brick boxes and probe spheres, making
          // it possible to inspect one refinement level without visual clutter.
          const level_text = (debug_stats?.debug_level ?? -1) < 0 ? "all levels" : `level ${debug_stats.debug_level}`;
          const view_text = parsed.debug_view === DebugDrawType.SVLM_Probes ? "probe" : "brick";
          log(`SVLM ${view_text} debug enabled: ${level_text}.`);
        } else {
          log("SVLM debug disabled.");
        }
        break;
      case "clear":
        svlm.clear();
        log("SVLM bake data cleared.");
        break;
      case "stats":
        this.show();
        break;
      case "hide":
        this.hide();
        break;
      default:
        log("svlm [stats | bake [root=<size>] [max=<level>] [min=<level>] | debug [bricks|probes] [on|off|toggle] [level=<n>|all|next|prev] [radius=<world>] | clear | hide]");
        break;
    }
  }

  render() {
    const svlm = get_svlm();
    const stats = svlm?.get_stats?.() || null;

    const panel_state = panel(stats_panel_config, () => {
      label("SVLM", { ...stats_label_config, font: "18px monospace", text_color: "#75e0b8" });
      label("--------------------------------", stats_label_config);

      if (!stats || !stats.baked) {
        label(stats?.message || "No SVLM bake yet. Run: svlm bake", stats_label_config);
        return;
      }

      stat_row("Bake serial", format_number(stats.bake_serial));
      stat_row("TLAS tests", format_number(stats.geometry_count));
      stat_row("BLAS tests", format_number(stats.blas_geometry_tests));
      stat_row("Nodes", format_number(stats.node_count));
      stat_row("Leaf bricks", format_number(stats.leaf_count));
      stat_row(
        "Budget",
        `${format_number(stats.config.max_nodes)} nodes, ${format_number(stats.config.max_leaf_bricks)} leaves`
      );
      stat_row("Allocated probes", format_number(stats.probe_count));
      stat_row("Root dims", `${stats.root_dims[0]} x ${stats.root_dims[1]} x ${stats.root_dims[2]}`);
      stat_row("Root brick size", Number(stats.root_brick_size).toFixed(2));
      stat_row("Levels", `${stats.min_level} forced, ${stats.max_level_reached}/${stats.max_level} reached`);
      stat_row("World min", format_vec3(stats.world_min));
      stat_row("World max", format_vec3(stats.world_max));
      stat_row("GPU data", `${bytes_to_mb(stats.total_bytes)} MB`);
      stat_row("Debug level", stats.debug_level < 0 ? "all" : `L${stats.debug_level}`);
      stat_row("Debug bricks", `${format_number(stats.debug_leaf_count)} / ${format_number(stats.leaf_count)}`);

      const level_parts = [];
      for (let i = 0; i < stats.per_level_counts.length; i += 1) {
        if (stats.per_level_counts[i] > 0) {
          level_parts.push(`L${i}:${format_number(stats.per_level_counts[i])}`);
        }
      }
      stat_row("Level leaves", level_parts.join("  ") || "none");

      if (stats.truncated_by_node_limit || stats.truncated_by_leaf_limit) {
        label("Bake hit an SVLM allocation limit.", { ...stats_label_config, text_color: "#ffb15c" });
      }

      label("--------------------------------", stats_label_config);
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
    this.scene = scene;
  }
}
