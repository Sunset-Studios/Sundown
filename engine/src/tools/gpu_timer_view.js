import { Renderer } from "../renderer/renderer.js";
import { DevConsoleTool } from "./dev_console_tool.js";
import { panel, label } from "../ui/2d/immediate.js";
import { GPUTimeQuery } from "../renderer/query.js";
import { RenderPass } from "../renderer/render_pass.js";
import { ResourceCache } from "../renderer/resource_cache.js";
import { CacheTypes } from "../renderer/renderer_types.js";
import { RollingAverage } from "../utility/rolling_average.js";
import { InputProvider } from "../input/input_provider.js";
import { InputKey } from "../input/input_types.js";

export class GPUTimerView extends DevConsoleTool {
  is_open = false;
  pass_averages = new Map();
  sort_by_time = false;

  container_config = {
    x: 0,
    y: 25,
    width: 450,
    height: "95%",
    padding: 10,
    background_color: "rgba(0, 0, 0, 0.7)",
    border: "1px solid rgb(68, 68, 68)",
    corner_radius: 5,
    clip: true,
  };

  panel_config = {
    x: 0,
    y: 0,
    width: "100%",
    height: "100%",
    layout: "column",
    gap: 6,
    padding: 10,
    scrollable: true,
    scroll_speed: 35,
    clip: true,
  };

  header_label_config = {
    x: 0,
    y: 0,
    width: "100%",
    height: 34,
    font: "600 18px monospace",
    text_color: "#e0e0e0",
    text_valign: "middle",
    text_align: "left",
    text_padding: 6,
  };

  header_panel_config = {
    x: 0,
    y: 0,
    width: "100%",
    height: 45,
    padding: 6,
    background_color: "rgba(15, 32, 16, 0.85)",
  };

  divider_config = {
    x: 0,
    y: 0,
    width: "100%",
    height: 5,
  };

  row_panel_config = {
    x: 0,
    y: 0,
    width: "100%",
    height: 30,
    layout: "row",
    gap: 6,
    border_bottom: "1px solid rgba(255,255,255,0.08)",
  };

  name_label_cfg = {
    x: 0,
    y: 0,
    wrap: true,
    height: "fit-content",
    width: "80%",
    font: "14px monospace",
    text_color: "#ccc",
    text_valign: "middle",
    text_align: "left",
  };

  time_label_cfg = {
    x: 0,
    y: 0,
    width: "20%",
    height: "100%",
    font: "14px monospace",
    text_color: "#ccc",
    text_align: "right",
    text_valign: "middle",
  };

  init() {
    super.init();
  }

  update() {
    super.update();
    if (!this.is_open) return;
    this.render();
  }

  execute() {
    super.execute();
    this.toggle();
  }

  render() {
    const results = GPUTimeQuery.get_results() || [];
    let total_ms = 0;

    if (InputProvider.get_action(InputKey.K_r)) {
      this.sort_by_time = !this.sort_by_time;
    }

    panel(this.container_config, () => {
      panel(this.panel_config, () => {
        panel(this.header_panel_config, () => {
          const header_text = `GPU Pass Timings (ms)  [R: sort ${this.sort_by_time ? "ON" : "OFF"}]`;
          label(header_text, this.header_label_config);
        });
        panel(this.divider_config, () => {});

        // collect and sort pass timings by averaged duration (descending)
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

          const res_ns = have_indices ? results[pass_timer_idx2] - results[pass_timer_idx1] : 0n;
          const res_ms = Number(res_ns) / 1e6;

          const key = pass.config.name;
          let avg = this.pass_averages.get(key);
          if (!avg) {
            avg = new RollingAverage(30);
            this.pass_averages.set(key, avg);
          }
          avg.add_sample(res_ms);
          const avg_ms = avg.get_average();

          total_ms += avg_ms;

          pass_entries.push({ name: key, avg_ms });
        }

        if (this.sort_by_time) {
          pass_entries.sort((a, b) => (a.avg_ms > b.avg_ms ? -1 : a.avg_ms < b.avg_ms ? 1 : 0));
        }

        for (let i = 0; i < pass_entries.length; i++) {
          const entry = pass_entries[i];
          panel(this.row_panel_config, () => {
            label(entry.name, this.name_label_cfg);
            label(entry.avg_ms.toFixed(4), this.time_label_cfg);
          });
        }

        panel(this.row_panel_config, () => {
          label("Total", this.name_label_cfg);
          label(total_ms.toFixed(4), this.time_label_cfg);
        });
      });
    });
  }
}
