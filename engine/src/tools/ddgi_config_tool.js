import { InputProvider } from "../input/input_provider.js";
import { InputKey } from "../input/input_types.js";
import { summarize_ddgi_config } from "../renderer/global_illumination/ddgi_config.js";
import { Renderer } from "../renderer/renderer.js";
import { button, label, panel, slider } from "../ui/2d/immediate.js";
import { log } from "../utility/logging.js";
import { DevConsoleTool } from "./dev_console_tool.js";

export { summarize_ddgi_config } from "../renderer/global_illumination/ddgi_config.js";

const accent = "#66e3ff";
const accent_soft = "rgba(102, 227, 255, 0.16)";
const panel_surface = "rgba(10, 15, 22, 0.96)";
const section_surface = "rgba(255, 255, 255, 0.025)";
const field_surface = "rgba(255, 255, 255, 0.055)";
const field_hover = "rgba(102, 227, 255, 0.12)";
const field_active = "rgba(102, 227, 255, 0.2)";
const subdued_text = "#7f91a4";
const body_text = "#dce7ef";
const warning = "#ffbf69";

const grid_sizes = Object.freeze([8, 16, 32, 64, 128]);
const depth_resolutions = Object.freeze([4, 8, 16, 32]);

export const DDGI_CONFIG_RANGES = Object.freeze({
  probe_spacing: Object.freeze({ min: 0.25, max: 8, step: 0.05, precision: 2 }),
  probe_radius: Object.freeze({ min: 0.01, max: 1, step: 0.01, precision: 2 }),
  max_rays_per_probe: Object.freeze({ min: 16, max: 256, step: 16, precision: 0 }),
  probes_per_frame: Object.freeze({ min: 0, max: 8192, step: 128, precision: 0 }),
  indirect_boost: Object.freeze({ min: 0, max: 4, step: 0.05, precision: 2 }),
  cascade_count: Object.freeze({ min: 1, max: 6, step: 1, precision: 0 }),
  cascade_spacing_multiplier: Object.freeze({
    min: 1,
    max: 4,
    step: 0.05,
    precision: 2,
  }),
  probe_depth_slot_count: Object.freeze({
    min: 1024,
    max: 262144,
    step: 1024,
    precision: 0,
  }),
  probe_depth_slot_retention_frames: Object.freeze({
    min: 1,
    max: 600,
    step: 1,
    precision: 0,
  }),
  max_emissive_lights: Object.freeze({
    min: 1024,
    max: 65536,
    step: 1024,
    precision: 0,
  }),
  diffuse_sample_upscale_factor: Object.freeze({
    min: 1,
    max: 4,
    step: 1,
    precision: 0,
  }),
  diffuse_atrous_pass_count: Object.freeze({ min: 0, max: 6, step: 1, precision: 0 }),
  diffuse_atrous_phi_depth: Object.freeze({
    min: 0.005,
    max: 0.2,
    step: 0.005,
    precision: 3,
  }),
  diffuse_atrous_phi_normal: Object.freeze({
    min: 1,
    max: 128,
    step: 1,
    precision: 0,
  }),
  diffuse_atrous_luma_sigma: Object.freeze({
    min: 0.05,
    max: 4,
    step: 0.05,
    precision: 2,
  }),
});

const tool_panel_config = {
  layout: "column",
  gap: 10,
  x: 24,
  y: 24,
  anchor_x: "right",
  width: 852,
  height: 748,
  padding: 12,
  background_color: panel_surface,
  border: "1px solid rgba(102,227,255,0.26)",
  corner_radius: 8,
  box_shadow: "0 12px 32px #00000073",
};

const header_label_config = {
  width: 520,
  height: 44,
  x: 0,
  y: 0,
  font: "18px monospace",
  text_color: "#effaff",
  text_align: "left",
  text_valign: "middle",
  text_padding: 10,
};

const section_panel_config = {
  layout: "column",
  gap: 2,
  x: 0,
  y: 0,
  width: 403,
  height: 620,
  padding: 10,
  background_color: section_surface,
  border: "1px solid rgba(255,255,255,0.06)",
  corner_radius: 6,
};

const section_header_config = {
  width: "100%",
  height: 32,
  x: 0,
  y: 0,
  font: "12px monospace",
  text_color: accent,
  text_align: "left",
  text_valign: "middle",
  text_padding: 4,
};

const field_label_config = {
  width: 153,
  height: 28,
  x: 0,
  y: 0,
  font: "12px monospace",
  text_color: body_text,
  text_align: "left",
  text_valign: "middle",
  text_padding: 4,
};

const slider_base_config = {
  width: 218,
  height: 26,
  x: 0,
  y: 0,
  background_color: field_surface,
  hover_color: field_hover,
  active_color: field_active,
  fill_color: "rgba(102, 227, 255, 0.42)",
  handle_color: "#b9f4ff",
  text_color: "#effaff",
  font: "12px monospace",
  border: "1px solid rgba(255,255,255,0.08)",
  corner_radius: 4,
  track_padding: 2,
  show_value: true,
};

const small_button_config = {
  width: 84,
  height: 30,
  x: 0,
  y: 0,
  background_color: "rgba(255, 255, 255, 0.06)",
  border: "1px solid rgba(255,255,255,0.1)",
  corner_radius: 4,
  text_color: "#c7d5df",
  font: "12px monospace",
};

function current_ddgi() {
  const strategy = Renderer.get()?.get_render_strategy?.();
  const gi = strategy?.gi ?? null;
  return gi?.config?.probe_grid_dimensions && typeof gi.set_config === "function" ? gi : null;
}

function compact_integer(value) {
  return Math.round(Number(value)).toLocaleString("en-US");
}

function nearest_choice_index(value, choices) {
  let nearest_index = 0;
  let nearest_distance = Number.POSITIVE_INFINITY;
  for (let index = 0; index < choices.length; index++) {
    const distance = Math.abs(Number(value) - choices[index]);
    if (distance < nearest_distance) {
      nearest_distance = distance;
      nearest_index = index;
    }
  }
  return nearest_index;
}

function section_header(title, subtitle) {
  label(`${title.toUpperCase()}  //  ${subtitle}`, section_header_config);
}

function field_row(label_text, callback) {
  panel(
    {
      layout: "row",
      gap: 8,
      x: 0,
      y: 0,
      width: "100%",
      height: 30,
    },
    () => {
      label(label_text, field_label_config);
      callback();
    }
  );
}

function numeric_field(ddgi, key, label_text, options = {}) {
  const range = DDGI_CONFIG_RANGES[key];
  field_row(label_text, () => {
    const result = slider(ddgi.config[key], {
      ...slider_base_config,
      ...range,
      id: `ddgi.${key}`,
      mode: options.mode ?? "bar",
      scrub_speed: options.scrub_speed ?? range.step,
      suffix: options.suffix,
      format_value: options.format_value,
    });
    if (result.changed) {
      const next_value = options.integer ? Math.round(result.value) : result.value;
      ddgi.set_config({ [key]: next_value });
    }
  });
}

function choice_field(ddgi, key, label_text, choices, array_index = null) {
  const current_value =
    array_index === null ? ddgi.config[key] : (ddgi.config[key]?.[array_index] ?? choices[0]);
  const current_index = nearest_choice_index(current_value, choices);
  const id_suffix = array_index === null ? "" : `.${array_index}`;

  field_row(label_text, () => {
    const result = slider(current_index, {
      ...slider_base_config,
      id: `ddgi.${key}${id_suffix}`,
      min: 0,
      max: choices.length - 1,
      step: 1,
      precision: 0,
      format_value: (index) => compact_integer(choices[Math.round(index)]),
    });
    if (!result.changed) return;

    const next_value = choices[Math.round(result.value)];
    if (array_index === null) {
      ddgi.set_config({ [key]: next_value });
    } else {
      const next_array = [...(ddgi.config[key] || [])];
      while (next_array.length < 6) next_array.push(choices[0]);
      next_array[array_index] = next_value;
      ddgi.set_config({ [key]: next_array });
    }
  });
}

function grid_axis_field(ddgi, axis, label_text) {
  const current = ddgi.config.probe_grid_dimensions?.[axis] ?? 64;
  const current_index = nearest_choice_index(current, grid_sizes);
  field_row(label_text, () => {
    const result = slider(current_index, {
      ...slider_base_config,
      id: `ddgi.probe_grid_dimensions.${axis}`,
      min: 0,
      max: grid_sizes.length - 1,
      step: 1,
      precision: 0,
      format_value: (index) => String(grid_sizes[Math.round(index)]),
    });
    if (!result.changed) return;

    const dimensions = [...(ddgi.config.probe_grid_dimensions || [64, 64, 64])];
    dimensions[axis] = grid_sizes[Math.round(result.value)];
    ddgi.set_config({ probe_grid_dimensions: dimensions });
  });
}

function boolean_field(ddgi, key, label_text) {
  field_row(label_text, () => {
    const enabled = ddgi.config[key] === true;
    const result = button(enabled ? "ENABLED" : "DISABLED", {
      ...slider_base_config,
      width: 218,
      background_color: enabled ? accent_soft : field_surface,
      border: enabled
        ? "1px solid rgba(102,227,255,0.42)"
        : "1px solid rgba(255,255,255,0.08)",
      text_color: enabled ? "#b9f4ff" : subdued_text,
    });
    if (result.clicked) {
      ddgi.set_config({ [key]: !enabled });
    }
  });
}

function metric(label_text, value, value_color = accent) {
  panel(
    {
      layout: "row",
      gap: 6,
      x: 0,
      y: 0,
      width: 190,
      height: 24,
      padding_left: 7,
      padding_right: 7,
      background_color: "rgba(255, 255, 255, 0.035)",
      corner_radius: 4,
    },
    () => {
      label(label_text, {
        width: "fit-content",
        height: "100%",
        x: 0,
        font: "11px monospace",
        text_color: subdued_text,
        text_valign: "middle",
      });
      label(value, {
        width: "fit-content",
        height: "100%",
        x: 0,
        font: "11px monospace",
        text_color: value_color,
        text_valign: "middle",
      });
    }
  );
}

export class DDGIConfigTool extends DevConsoleTool {
  is_open = false;
  scene = null;

  update() {
    if (this.is_open) {
      this.render();
    }
  }

  execute(args = []) {
    const command = String(args[0] ?? "config").toLowerCase();
    if (command === "config" || command === "show") {
      this.show();
    } else if (command === "hide" || command === "close") {
      this.hide();
    } else if (command === "reset") {
      current_ddgi()?.reset_config?.();
      this.show();
    } else {
      log("ddgi [config | reset | hide]");
    }
  }

  render() {
    const ddgi = current_ddgi();
    const panel_state = panel(tool_panel_config, () => {
      panel(
        {
          layout: "row",
          gap: 8,
          x: 0,
          y: 0,
          width: "100%",
          height: 44,
        },
        () => {
          label("DDGI  //  LIVE CONFIG", header_label_config);
          label(ddgi ? "● ACTIVE" : "● UNAVAILABLE", {
            width: 112,
            height: 30,
            x: 0,
            y: 0,
            font: "11px monospace",
            text_color: ddgi ? "#7df5b5" : warning,
            text_align: "center",
            text_valign: "middle",
            background_color: ddgi ? "rgba(65, 214, 139, 0.1)" : "rgba(255, 191, 105, 0.1)",
            border: ddgi
              ? "1px solid rgba(65,214,139,0.25)"
              : "1px solid rgba(255,191,105,0.25)",
            corner_radius: 12,
          });
          if (button("RESET", small_button_config).clicked) {
            ddgi?.reset_config?.();
          }
          if (
            button("CLOSE", {
              ...small_button_config,
              width: 72,
              text_color: "#ff9eaa",
            }).clicked
          ) {
            this.hide();
          }
        }
      );

      if (!ddgi) {
        panel(
          {
            x: 0,
            y: 0,
            width: "100%",
            height: 620,
            background_color: "rgba(255, 191, 105, 0.04)",
            border: "1px solid rgba(255,191,105,0.16)",
            corner_radius: 6,
          },
          () => {
            label("DDGI is not the active GI strategy.", {
              x: 0,
              y: 0,
              width: "100%",
              height: 54,
              font: "14px monospace",
              text_color: warning,
              text_align: "center",
              text_valign: "middle",
            });
          }
        );
        return;
      }

      panel(
        {
          layout: "row",
          gap: 10,
          x: 0,
          y: 0,
          width: "100%",
          height: 620,
        },
        () => {
          panel(section_panel_config, () => {
            section_header("Probe volume", "spatial layout");
            grid_axis_field(ddgi, 0, "Grid X");
            grid_axis_field(ddgi, 1, "Grid Y");
            grid_axis_field(ddgi, 2, "Grid Z");
            numeric_field(ddgi, "probe_spacing", "Probe spacing", { suffix: " m" });
            numeric_field(ddgi, "probe_radius", "Probe radius", { suffix: " m" });

            section_header("Update budget", "trace workload");
            numeric_field(ddgi, "max_rays_per_probe", "Rays / probe", {
              mode: "numeric",
              integer: true,
              format_value: compact_integer,
            });
            numeric_field(ddgi, "probes_per_frame", "Probes / frame", {
              mode: "numeric",
              integer: true,
              format_value: (value) => (value === 0 ? "ALL" : compact_integer(value)),
            });
            numeric_field(ddgi, "indirect_boost", "Indirect boost", { suffix: "×" });
            numeric_field(ddgi, "cascade_count", "Cascade count", { integer: true });
            numeric_field(ddgi, "cascade_spacing_multiplier", "Cascade scale", { suffix: "×" });
            numeric_field(ddgi, "max_emissive_lights", "Emissive lights", {
              mode: "numeric",
              integer: true,
              format_value: compact_integer,
            });
          });

          panel(section_panel_config, () => {
            section_header("Depth cache", "sparse moments");
            const cascade_count = Math.max(1, Math.min(6, Math.floor(ddgi.config.cascade_count)));
            for (let cascade = 0; cascade < cascade_count; cascade++) {
              choice_field(
                ddgi,
                "probe_depth_resolutions",
                `Cascade ${cascade + 1} depth`,
                depth_resolutions,
                cascade
              );
            }
            numeric_field(ddgi, "probe_depth_slot_count", "Depth slots", {
              mode: "numeric",
              integer: true,
              format_value: compact_integer,
            });
            numeric_field(ddgi, "probe_depth_slot_retention_frames", "Slot retention", {
              mode: "numeric",
              integer: true,
              suffix: " f",
            });

            section_header("Diffuse resolve", "à-trous filter");
            numeric_field(ddgi, "diffuse_sample_upscale_factor", "Upscale factor", {
              integer: true,
              suffix: "×",
            });
            boolean_field(ddgi, "diffuse_atrous_enabled", "À-trous filter");
            numeric_field(ddgi, "diffuse_atrous_pass_count", "Filter passes", {
              integer: true,
            });
            numeric_field(ddgi, "diffuse_atrous_phi_depth", "Depth phi");
            numeric_field(ddgi, "diffuse_atrous_phi_normal", "Normal phi");
            numeric_field(ddgi, "diffuse_atrous_luma_sigma", "Luma sigma");
          });
        }
      );

      const summary = summarize_ddgi_config(ddgi.config);
      panel(
        {
          layout: "row",
          gap: 8,
          x: 0,
          y: 0,
          width: "100%",
          height: 24,
        },
        () => {
          metric("PROBES", compact_integer(summary.total_probes));
          metric("RAYS / FRAME", compact_integer(summary.rays_per_frame));
          metric("DEPTH SLOTS", compact_integer(summary.depth_slots));
          label("drag bars • scrub numbers", {
            width: 220,
            height: "100%",
            x: 0,
            font: "11px monospace",
            text_color: subdued_text,
            text_align: "right",
            text_valign: "middle",
          });
        }
      );
    });

    if (this.is_open && InputProvider.get_action(InputKey.B_mouse_left) && !panel_state.hovered) {
      this.hide();
      InputProvider.consume_action(InputKey.B_mouse_left);
    }
  }

  show() {
    this.is_open = true;
    this.scene?.show_dev_cursor?.();
  }

  hide() {
    this.is_open = false;
    this.scene?.hide_dev_cursor?.();
  }

  toggle() {
    if (this.is_open) this.hide();
    else this.show();
  }

  set_scene(scene) {
    this.scene = scene;
  }
}
