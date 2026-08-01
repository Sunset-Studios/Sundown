import { Renderer } from "../renderer/renderer.js";
import { DevConsoleTool } from "./dev_console_tool.js";
import { button, input, label, panel, UIContext } from "../ui/2d/immediate.js";
import { GPUTimeQuery } from "../renderer/query.js";
import { ResourceCache } from "../renderer/resource_cache.js";
import { CacheTypes } from "../renderer/renderer_types.js";
import { RollingAverage } from "../utility/rolling_average.js";
import { InputProvider } from "../input/input_provider.js";
import { InputKey } from "../input/input_types.js";

const filter_input_name = "gpu_timer_filter";

function create_filter_tokens(value) {
  return value
    .trim()
    .toLowerCase()
    .split(/\s+/)
    .filter((token) => token.length > 0);
}

function fuzzy_token_matches(candidate, token) {
  if (candidate.includes(token)) return true;

  let token_index = 0;
  for (let i = 0; i < candidate.length && token_index < token.length; i++) {
    if (candidate[i] === token[token_index]) {
      token_index++;
    }
  }

  return token_index === token.length;
}

function fuzzy_matches(value, filter_tokens) {
  if (filter_tokens.length === 0) return true;

  const candidate = value.toLowerCase();
  for (let i = 0; i < filter_tokens.length; i++) {
    if (!fuzzy_token_matches(candidate, filter_tokens[i])) return false;
  }

  return true;
}

export class GPUTimerView extends DevConsoleTool {
  is_open = false;
  pass_averages = new Map();
  sort_by_time = false;
  filter_value = "";
  filter_tokens = [];
  focus_filter_on_open = false;

  container_config = {
    x: 18,
    y: 24,
    width: 570,
    height: "94%",
    padding: 10,
    background_color: "rgba(5, 10, 15, 0.58)",
    border: "1px solid rgba(105, 225, 255, 0.28)",
    corner_radius: 10,
    box_shadow: "0 14px 36px #00000080",
    clip: true,
  };

  panel_config = {
    x: 0,
    y: 0,
    width: "100%",
    height: "100%",
    layout: "column",
    gap: 7,
    padding: 10,
    scrollable: true,
    scroll_speed: 35,
    clip: true,
  };

  header_panel_config = {
    x: 0,
    y: 0,
    width: "100%",
    height: 66,
    layout: "column",
    gap: 0,
    padding_left: 12,
    padding_right: 12,
    background_color: "rgba(13, 26, 35, 0.72)",
    border: "1px solid rgba(105, 225, 255, 0.18)",
    corner_radius: 7,
    clip: true,
  };

  accent_line_config = {
    x: 0,
    y: 0,
    width: "100%",
    height: 3,
    background_color: "#69e1ff",
    box_shadow: "0 0 12px #69e1ff80",
  };

  title_label_config = {
    x: 0,
    y: 0,
    width: "100%",
    height: 34,
    font: "700 18px monospace",
    text_color: "#e9f8ff",
    text_valign: "middle",
    text_align: "left",
    text_padding: 4,
  };

  status_label_config = {
    x: 0,
    y: 0,
    width: "100%",
    height: 23,
    font: "11px monospace",
    text_color: "#7898a5",
    text_valign: "middle",
    text_align: "left",
    text_padding: 4,
  };

  toolbar_config = {
    x: 0,
    y: 0,
    width: "100%",
    height: 38,
    layout: "row",
    gap: 6,
  };

  filter_input_config = {
    x: 0,
    y: 0,
    width: "62%",
    height: "100%",
    padding_left: 10,
    padding_right: 10,
    background_color: "rgba(2, 8, 12, 0.62)",
    border: "1px solid rgba(105, 225, 255, 0.3)",
    corner_radius: 6,
    text_color: "#dff7ff",
    placeholder: "Filter passes...",
    placeholder_color: "rgba(151, 181, 192, 0.58)",
    cursor_color: "#69e1ff",
    font: "13px monospace",
    blur_on_click_away: true,
  };

  toolbar_button_config = {
    x: 0,
    y: 0,
    width: "17%",
    height: "100%",
    background_color: "rgba(19, 38, 49, 0.68)",
    hover_color: "rgba(42, 77, 92, 0.95)",
    border: "1px solid rgba(133, 178, 194, 0.2)",
    corner_radius: 6,
    font: "600 11px monospace",
    text_color: "#93adba",
    text_align: "center",
    text_valign: "middle",
  };

  active_sort_button_config = {
    ...this.toolbar_button_config,
    background_color: "rgba(30, 91, 108, 0.76)",
    hover_color: "rgba(38, 111, 130, 0.92)",
    border: "1px solid rgba(105, 225, 255, 0.48)",
    text_color: "#bff5ff",
  };

  summary_panel_config = {
    x: 0,
    y: 0,
    width: "100%",
    height: 52,
    layout: "row",
    gap: 8,
    padding_left: 12,
    padding_right: 12,
    background_color: "rgba(105, 225, 255, 0.08)",
    border: "1px solid rgba(105, 225, 255, 0.16)",
    corner_radius: 7,
  };

  total_label_config = {
    x: 0,
    y: 0,
    width: "70%",
    height: "100%",
    font: "600 12px monospace",
    text_color: "#8ba9b5",
    text_valign: "middle",
    text_align: "left",
  };

  total_time_label_config = {
    x: 0,
    y: 0,
    width: "26%",
    height: "100%",
    font: "700 20px monospace",
    text_color: "#69e1ff",
    text_align: "right",
    text_valign: "middle",
  };

  column_header_config = {
    x: 0,
    y: 0,
    width: "100%",
    height: 24,
    layout: "row",
    gap: 6,
    padding_left: 10,
    padding_right: 10,
    border_bottom: "1px solid rgba(105, 225, 255, 0.2)",
  };

  column_name_config = {
    x: 0,
    y: 0,
    width: "76%",
    height: "100%",
    font: "600 10px monospace",
    text_color: "#67828d",
    text_valign: "middle",
    text_align: "left",
  };

  column_time_config = {
    ...this.column_name_config,
    width: "20%",
    text_align: "right",
  };

  row_panel_config = {
    x: 0,
    y: 0,
    width: "100%",
    height: 31,
    layout: "row",
    gap: 6,
    padding_left: 10,
    padding_right: 10,
    background_color: "rgba(255, 255, 255, 0.025)",
    border_bottom: "1px solid rgba(255, 255, 255, 0.055)",
    corner_radius: 4,
  };

  alternate_row_panel_config = {
    ...this.row_panel_config,
    background_color: "rgba(105, 225, 255, 0.035)",
  };

  name_label_config = {
    x: 0,
    y: 0,
    width: "76%",
    height: "100%",
    font: "12px monospace",
    text_color: "#c4d5dc",
    text_valign: "middle",
    text_align: "left",
  };

  time_label_config = {
    x: 0,
    y: 0,
    width: "20%",
    height: "100%",
    font: "600 12px monospace",
    text_color: "#8fdcf0",
    text_align: "right",
    text_valign: "middle",
  };

  empty_label_config = {
    x: 0,
    y: 0,
    width: "100%",
    height: 64,
    font: "12px monospace",
    text_color: "#708b96",
    text_align: "center",
    text_valign: "middle",
  };

  init() {
    super.init();
  }

  update() {
    super.update();
    if (!this.is_open) return;

    if (InputProvider.get_action(InputKey.K_Escape)) {
      const filter_state = UIContext.input_field_state[filter_input_name];
      if (filter_state) {
        filter_state.is_focused = false;
      }
      InputProvider.consume_action(InputKey.K_Escape);
      InputProvider.consume_state(InputKey.K_Escape);
      UIContext.consume_key(InputKey.K_Escape);
      this.hide();
      return;
    }

    this.render();
  }

  execute() {
    super.execute();

    const was_open = this.is_open;
    this.toggle();
    this.focus_filter_on_open = !was_open;
  }

  sync_filter(value) {
    if (value === this.filter_value) return;

    this.filter_value = value;
    this.filter_tokens = create_filter_tokens(value);
  }

  collect_pass_entries(results) {
    const pass_entries = [];
    const resolved_passes = Renderer.get().render_graph.get_resolved_non_culled_passes();

    for (let i = 0; i < resolved_passes.length; i++) {
      const pass_id = resolved_passes[i];
      const pass = ResourceCache.get().fetch(CacheTypes.PASS, pass_id);
      if (!pass) continue;

      const pass_timer_idx1 = pass.timer_query_indices[0];
      const pass_timer_idx2 = pass.timer_query_indices[1];
      const have_indices =
        pass_timer_idx1 !== undefined &&
        pass_timer_idx2 !== undefined &&
        pass_timer_idx1 < results.length &&
        pass_timer_idx2 < results.length &&
        results[pass_timer_idx1] !== undefined &&
        results[pass_timer_idx2] !== undefined;

      const result_ns = have_indices ? results[pass_timer_idx2] - results[pass_timer_idx1] : 0n;
      const result_ms = Number(result_ns) / 1e6;
      const pass_name = pass.config.name;

      let average = this.pass_averages.get(pass_name);
      if (!average) {
        average = new RollingAverage(30);
        this.pass_averages.set(pass_name, average);
      }
      average.add_sample(result_ms);

      const average_ms = average.get_average();
      if (fuzzy_matches(pass_name, this.filter_tokens)) {
        pass_entries.push({ name: pass_name, average_ms });
      }
    }

    return { pass_entries, pass_count: resolved_passes.length };
  }

  render() {
    const filter_state = UIContext.input_field_state[filter_input_name];
    const filter_is_focused = filter_state?.is_focused ?? false;
    if (!filter_is_focused && InputProvider.get_action(InputKey.K_r)) {
      this.sort_by_time = !this.sort_by_time;
    }

    this.sync_filter(filter_state?.value ?? "");

    const results = GPUTimeQuery.get_results() || [];
    const { pass_entries, pass_count } = this.collect_pass_entries(results);

    if (this.sort_by_time) {
      pass_entries.sort((a, b) => b.average_ms - a.average_ms);
    }

    let total_ms = 0;
    for (let i = 0; i < pass_entries.length; i++) {
      total_ms += pass_entries[i].average_ms;
    }

    panel(this.container_config, () => {
      panel(this.panel_config, () => {
        panel(this.header_panel_config, () => {
          panel(this.accent_line_config, () => {});
          label("GPU TIMINGS // PROFILER", this.title_label_config);
          label(
            `VISIBLE ${pass_entries.length}/${pass_count}  //  30 FRAME ROLLING AVERAGE`,
            this.status_label_config
          );
        });

        panel(this.toolbar_config, () => {
          const current_filter_state = input(filter_input_name, this.filter_input_config);
          if (this.focus_filter_on_open) {
            current_filter_state.is_focused = true;
            this.focus_filter_on_open = false;
          }

          if (button("CLEAR", this.toolbar_button_config).clicked) {
            current_filter_state.value = "";
            this.sync_filter("");
          }

          const sort_config = this.sort_by_time
            ? this.active_sort_button_config
            : this.toolbar_button_config;
          if (button(this.sort_by_time ? "TIME DESC" : "SORT TIME", sort_config).clicked) {
            this.sort_by_time = !this.sort_by_time;
            current_filter_state.is_focused = false;
          }
        });

        panel(this.summary_panel_config, () => {
          label(
            this.filter_tokens.length > 0 ? "FILTERED GPU TOTAL" : "TOTAL GPU TIME",
            this.total_label_config
          );
          label(`${total_ms.toFixed(3)} ms`, this.total_time_label_config);
        });

        panel(this.column_header_config, () => {
          label("RENDER PASS", this.column_name_config);
          label("AVG MS", this.column_time_config);
        });

        if (pass_entries.length === 0) {
          label(
            this.filter_tokens.length > 0
              ? "No passes match this filter"
              : "No GPU timings available",
            this.empty_label_config
          );
        }

        for (let i = 0; i < pass_entries.length; i++) {
          const entry = pass_entries[i];
          const row_config = i % 2 === 0 ? this.row_panel_config : this.alternate_row_panel_config;
          panel(row_config, () => {
            label(entry.name, this.name_label_config);
            label(entry.average_ms.toFixed(4), this.time_label_config);
          });
        }
      });
    });
  }
}
