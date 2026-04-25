import { CVarSystem } from "../core/cvar_system.js";
import { Renderer } from "../renderer/renderer.js";
import { DebugDrawType } from "../renderer/renderer_types.js";
import { EngineCVars } from "../../config/cvars.js";
import { InputProvider } from "../input/input_provider.js";
import { InputKey } from "../input/input_types.js";
import { panel, label } from "../ui/2d/immediate.js";
import { bytes_to_mb } from "../utility/math.js";
import { log, format_number, format_vec3 } from "../utility/logging.js";
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

function stat_row(label_text, value_text, value_config = value_label_config) {
  panel({ layout: "row", gap: 4, width: "100%", height: 25, anchor_x: "left", x: 0 }, () => {
    label(`${label_text}:`, stats_label_config_small);
    label(value_text, value_config);
  });
}

function parse_bake_options(args) {
  const options = {};

  for (let i = 0; i < args.length; i += 1) {
    const raw = args[i];
    const eq = raw.indexOf("=");
    if (eq === -1) continue;

    const key = raw.slice(0, eq).trim().toLowerCase();
    const value = raw.slice(eq + 1).trim();
    if (key === "root_brick_size") {
      options.root_brick_size = Number(value);
    } else if (key === "max_level") {
      options.max_level = Number(value);
    } else if (key === "min_level") {
      options.min_level = Number(value);
    } else if (key === "max_leaf_bricks") {
      options.max_leaf_bricks = Number(value);
    } else if (key === "max_nodes") {
      options.max_nodes = Number(value);
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
    } else if (lower === "brick") {
      debug_view = DebugDrawType.SVLM_Bricks;
    } else if (lower === "probe") {
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
    const strategy = Renderer.get().get_render_strategy();
    const svlm = strategy?.svlm || null;
    if (!svlm) return;

    switch (command) {
      case "bake":
        const bake_options = parse_bake_options(args.slice(1));
        svlm.bake(bake_options);
        this.show();
        break;
      case "debug":
        const debug_options = parse_debug_options(args.slice(1));
        const current_debug_view = CVarSystem.get(EngineCVars.Renderer.DebugDraw, DebugDrawType.None);
        const enable = debug_options.mode === "on" || current_debug_view !== debug_options.debug_view;
        CVarSystem.set(
          EngineCVars.Renderer.DebugDraw,
          enable ? debug_options.debug_view : DebugDrawType.None,
          { source: "svlm" }
        );
        break;
      case "clear":
        svlm.clear();
        break;
      case "stats":
        this.show();
        break;
      case "hide":
        this.hide();
        break;
      default:
        log("svlm [stats | bake [root=<size>] [max=<level>] [min=<level>] | debug [bricks|probes] [on|off] [level=<n>|all|next|prev] [radius=<world>] | clear | hide]");
        break;
    }
  }

  render() {
    const strategy = Renderer.get().get_render_strategy();
    const svlm = strategy?.svlm || null;
    if (!svlm) return;

    const stats = svlm.get_stats();

    const panel_state = panel(stats_panel_config, () => {
      label("SVLM", { ...stats_label_config, font: "18px monospace", text_color: "#75e0b8" });
      label("--------------------------------", stats_label_config);

      if (!stats.baked) return;

      stat_row("Bake serial", format_number(stats.bake_serial));
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
