import { DevConsoleTool } from "./dev_console_tool.js";
import { panel, label, UIContext } from "../ui/2d/immediate.js";
import { MLTrace } from "../ml/tick_trace.js";

const MAX_RENDERED_LINES = 160;

function parse_frame_count(args) {
  const raw = args[1] ?? args[0]?.split("=")[1];
  const frame_count = Number(raw);
  return Number.isFinite(frame_count) && frame_count > 0 ? Math.floor(frame_count) : Infinity;
}

export class MLTraceTool extends DevConsoleTool {
  container_config = {
    x: 0,
    y: 0,
    anchor_x: "left",
    anchor_y: "bottom",
    width: "100%",
    height: 300,
    padding: 8,
    background_color: "rgba(3, 5, 10, 0.88)",
    border: "1px solid rgba(255, 180, 90, 0.34)",
    corner_radius: 0,
    clip: true,
  };

  panel_config = {
    x: 0,
    y: 0,
    width: "100%",
    height: "100%",
    layout: "column",
    gap: 4,
    padding: 8,
  };

  log_panel_config = {
    x: 0,
    y: 0,
    width: "100%",
    height: 168,
    layout: "column",
    gap: 4,
    padding_top: 15,
    scrollable: true,
    scroll_speed: 35,
    clip: true,
    widget_id: "ml_trace_output",
  };

  header_panel_config = {
    x: 0,
    y: 0,
    width: "100%",
    height: 92,
    layout: "column",
    gap: 5,
    padding: 8,
    background_color: "rgba(8, 10, 18, 0.97)",
    border: "1px solid rgba(47, 236, 210, 0.44)",
  };

  accent_line_config = {
    x: 0,
    y: 0,
    width: "100%",
    height: 3,
    background_color: "rgba(255, 63, 172, 0.95)",
  };

  header_row_config = {
    x: 0,
    y: 0,
    width: "100%",
    height: 34,
    layout: "row",
    gap: 8,
  };

  title_config = {
    x: 0,
    y: 0,
    width: "45%",
    height: "100%",
    font: "600 17px monospace",
    text_color: "#eafff7",
    text_valign: "middle",
    text_align: "left",
    text_padding: 8,
    background_color: "rgba(11, 30, 28, 0.86)",
    border: "1px solid rgba(47, 236, 210, 0.42)",
  };

  status_config = {
    x: 0,
    y: 0,
    width: "55%",
    height: "100%",
    font: "600 12px monospace",
    text_color: "#ffd28a",
    text_valign: "middle",
    text_align: "right",
    text_padding: 8,
    background_color: "rgba(30, 19, 14, 0.84)",
    border: "1px solid rgba(255, 180, 90, 0.34)",
  };

  button_row_config = {
    x: 0,
    y: 0,
    width: "100%",
    height: 28,
    layout: "row",
    gap: 6,
  };

  button_config = {
    x: 0,
    y: 0,
    width: 78,
    height: "100%",
    font: "600 11px monospace",
    text_color: "#bdfaf0",
    text_valign: "middle",
    text_align: "center",
    text_padding: 4,
    background_color: "rgba(12, 20, 28, 0.92)",
    border: "1px solid rgba(47, 236, 210, 0.28)",
    underline_on_hover: true,
    underline_color: "#ff3fac",
  };

  active_button_config = {
    x: 0,
    y: 0,
    width: 78,
    height: "100%",
    font: "700 11px monospace",
    text_color: "#14070f",
    text_valign: "middle",
    text_align: "center",
    text_padding: 4,
    background_color: "rgba(255, 211, 110, 0.96)",
    border: "1px solid rgba(255, 244, 190, 0.96)",
  };

  row_config = {
    x: 0,
    y: 0,
    width: "100%",
    height: 21,
    font: "12px monospace",
    text_color: "#cdeee9",
    text_valign: "middle",
    text_align: "left",
    text_padding: 4,
    border_bottom: "1px solid rgba(47,236,210,0.08)",
  };

  update_layout_metrics() {
    const canvas_height = UIContext.canvas_size.height || 900;
    const canvas_width = UIContext.canvas_size.width || 1280;
    const compact = canvas_height < 650;
    const narrow = canvas_width < 900;

    const panel_height = Math.min(Math.max(Math.round(canvas_height * 0.36), 330), 440);
    const header_height = compact ? 92 : 108;
    const row_height = compact ? 24 : 27;
    const log_height = Math.max(120, panel_height - header_height - 36);
    const button_width = narrow ? 68 : 92;

    this.container_config.height = panel_height;
    this.header_panel_config.height = header_height;
    this.header_row_config.height = compact ? 34 : 40;
    this.button_row_config.height = compact ? 28 : 32;
    this.log_panel_config.height = log_height;

    this.title_config.font = compact ? "600 18px monospace" : "600 21px monospace";
    this.status_config.font = compact ? "600 13px monospace" : "600 15px monospace";
    this.button_config.font = compact ? "600 12px monospace" : "600 14px monospace";
    this.active_button_config.font = compact ? "700 12px monospace" : "700 14px monospace";
    this.row_config.font = compact ? "13px monospace" : "15px monospace";
    this.row_config.height = row_height;
    this.button_config.width = button_width;
    this.active_button_config.width = button_width;
  }

  render_button(text, selected, callback) {
    const result = label(text, selected ? this.active_button_config : this.button_config);
    if (result.clicked) {
      callback();
    }
  }

  update() {
    super.update();
    if (!this.is_open) return;
    this.render();
  }

  execute(args = []) {
    super.execute(args);

    const command = (args[0] ?? "on").toLowerCase();
    if (command === "off") {
      MLTrace.disable();
      this.hide();
      return;
    }

    if (command === "hide") {
      this.hide();
      return;
    }

    if (command === "clear") {
      MLTrace.clear();
      this.show();
      return;
    }

    if (command === "frames" || command.startsWith("frames=")) {
      MLTrace.enable(parse_frame_count(args));
      this.show();
      return;
    }

    MLTrace.enable();
    this.show();
  }

  render() {
    this.update_layout_metrics();

    const state = MLTrace.get_state();
    const lines = MLTrace.get_lines();
    const frame_text = Number.isFinite(state.frames_remaining)
      ? `frames=${state.frames_remaining}`
      : "continuous";
    const finite_frame_limit = Number.isFinite(state.frame_limit) ? state.frame_limit : null;

    panel(this.container_config, () => {
      panel(this.panel_config, () => {
        panel(this.header_panel_config, () => {
          panel(this.accent_line_config, () => { });
          panel(this.header_row_config, () => {
            label("ML TRACE // TELEMETRY", this.title_config);
            label(
              `STATE:${state.enabled ? "ARMED" : "IDLE"}  TICK:${state.tick_index}  LINES:${state.line_count}  MODE:${frame_text}`,
              this.status_config
            );
          });

          panel(this.button_row_config, () => {
            this.render_button("LIVE", state.enabled && finite_frame_limit === null, () => {
              MLTrace.enable();
            });
            this.render_button("1T", state.enabled && finite_frame_limit === 1, () => {
              MLTrace.enable(1);
            });
            this.render_button("8T", state.enabled && finite_frame_limit === 8, () => {
              MLTrace.enable(8);
            });
            this.render_button("60T", state.enabled && finite_frame_limit === 60, () => {
              MLTrace.enable(60);
            });
            this.render_button("STOP", !state.enabled, () => {
              MLTrace.disable();
            });
            this.render_button("CLEAR", false, () => {
              MLTrace.clear();
            });
            this.render_button("HIDE", false, () => {
              this.hide();
            });
          });
        });

        panel(this.log_panel_config, () => {
          const first_line = Math.max(0, lines.length - MAX_RENDERED_LINES);
          for (let i = first_line; i < lines.length; i++) {
            label(lines[i], this.row_config);
          }
        });
      });
    });
  }
}
