import { DevConsoleTool } from "./dev_console_tool.js";
import { button, input, label, panel, UIContext } from "../ui/2d/immediate.js";
import { ResourceCache } from "../renderer/resource_cache.js";
import { CacheTypes } from "../renderer/renderer_types.js";
import { InputProvider } from "../input/input_provider.js";
import { InputKey } from "../input/input_types.js";
import { Texture } from "../renderer/texture.js";

const MB_THRESHOLD = 0.01;
const filter_input_name = "gpu_memory_filter";

function format_size(bytes) {
  const mb = bytes / (1024.0 * 1024.0);
  if (mb < MB_THRESHOLD) {
    const kb = bytes / 1024.0;
    return `${kb.toFixed(2)} KB`;
  }
  return `${mb.toFixed(2)} MB`;
}

function compute_texture_size(config) {
  if (!config) return 0;

  const bytes_per_pixel = Texture.stride_from_format(config.format);
  const depth = config.depth || 1;
  const mip_levels = config.mip_levels || 1;

  let total_bytes = 0;
  for (let mip = 0; mip < mip_levels; mip++) {
    const width = Math.max(1, (config.width || 1) >> mip);
    const height = Math.max(1, (config.height || 1) >> mip);
    total_bytes += width * height * depth * bytes_per_pixel;
  }

  return total_bytes;
}

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

export class GPUMemoryView extends DevConsoleTool {
  is_open = false;
  sort_by_size = false;
  filter_value = "";
  filter_tokens = [];
  focus_filter_on_open = false;

  container_config = {
    x: 18,
    y: 24,
    width: 620,
    height: "94%",
    padding: 10,
    background_color: "rgba(8, 6, 15, 0.58)",
    border: "1px solid rgba(196,139,255,0.28)",
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
    background_color: "rgba(29, 19, 42, 0.72)",
    border: "1px solid rgba(196,139,255,0.18)",
    corner_radius: 7,
    clip: true,
  };

  accent_line_config = {
    x: 0,
    y: 0,
    width: "100%",
    height: 3,
    background_color: "#c48bff",
    box_shadow: "0 0 12px #c48bff80",
  };

  title_label_config = {
    x: 0,
    y: 0,
    width: "100%",
    height: 34,
    font: "700 18px monospace",
    text_color: "#f7efff",
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
    text_color: "#9987aa",
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
    background_color: "rgba(10, 5, 16, 0.62)",
    border: "1px solid rgba(196,139,255,0.3)",
    corner_radius: 6,
    text_color: "#f4e9ff",
    placeholder: "Filter resources...",
    placeholder_color: "rgba(177, 155, 194, 0.58)",
    cursor_color: "#c48bff",
    font: "13px monospace",
    blur_on_click_away: true,
  };

  toolbar_button_config = {
    x: 0,
    y: 0,
    width: "17%",
    height: "100%",
    background_color: "rgba(45, 30, 59, 0.68)",
    hover_color: "rgba(76, 50, 96, 0.9)",
    border: "1px solid rgba(188,157,211,0.2)",
    corner_radius: 6,
    font: "600 11px monospace",
    text_color: "#b4a1c2",
    text_align: "center",
    text_valign: "middle",
  };

  active_sort_button_config = {
    ...this.toolbar_button_config,
    background_color: "rgba(96, 56, 128, 0.76)",
    hover_color: "rgba(118, 68, 156, 0.92)",
    border: "1px solid rgba(196,139,255,0.48)",
    text_color: "#edd8ff",
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
    background_color: "rgba(196, 139, 255, 0.08)",
    border: "1px solid rgba(196,139,255,0.16)",
    corner_radius: 7,
  };

  total_label_config = {
    x: 0,
    y: 0,
    width: "70%",
    height: "100%",
    font: "600 12px monospace",
    text_color: "#a692b6",
    text_valign: "middle",
    text_align: "left",
  };

  total_size_label_config = {
    x: 0,
    y: 0,
    width: "26%",
    height: "100%",
    font: "700 20px monospace",
    text_color: "#c48bff",
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
    border_bottom: "1px solid rgba(196,139,255,0.2)",
  };

  column_name_config = {
    x: 0,
    y: 0,
    width: "76%",
    height: "100%",
    font: "600 10px monospace",
    text_color: "#776684",
    text_valign: "middle",
    text_align: "left",
  };

  column_size_config = {
    ...this.column_name_config,
    width: "20%",
    text_align: "right",
  };

  section_header_config = {
    x: 0,
    y: 0,
    width: "100%",
    height: 32,
    layout: "row",
    gap: 6,
    padding_left: 10,
    padding_right: 10,
    background_color: "rgba(196, 139, 255, 0.065)",
    border_left: "2px solid rgba(196,139,255,0.7)",
    corner_radius: 4,
  };

  section_title_config = {
    x: 0,
    y: 0,
    width: "70%",
    height: "100%",
    font: "700 11px monospace",
    text_color: "#c9a6e8",
    text_valign: "middle",
    text_align: "left",
  };

  section_total_config = {
    x: 0,
    y: 0,
    width: "26%",
    height: "100%",
    font: "600 11px monospace",
    text_color: "#ac91bf",
    text_align: "right",
    text_valign: "middle",
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
    border_bottom: "1px solid rgba(255,255,255,0.055)",
    corner_radius: 4,
  };

  alternate_row_panel_config = {
    ...this.row_panel_config,
    background_color: "rgba(196, 139, 255, 0.035)",
  };

  name_label_config = {
    x: 0,
    y: 0,
    width: "76%",
    height: "100%",
    font: "12px monospace",
    text_color: "#d7cbdc",
    text_valign: "middle",
    text_align: "left",
  };

  size_label_config = {
    x: 0,
    y: 0,
    width: "20%",
    height: "100%",
    font: "600 12px monospace",
    text_color: "#c0a2d4",
    text_align: "right",
    text_valign: "middle",
  };

  empty_label_config = {
    x: 0,
    y: 0,
    width: "100%",
    height: 48,
    font: "12px monospace",
    text_color: "#806f8c",
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

  collect_entries() {
    const buffer_map = ResourceCache.get().fetch_all(CacheTypes.BUFFER);
    const image_map = ResourceCache.get().fetch_all(CacheTypes.IMAGE);
    const buffer_entries = [];
    const texture_entries = [];
    let buffer_count = 0;
    let texture_count = 0;

    for (const [key, buffer] of buffer_map) {
      buffer_count++;
      const name = buffer?.config?.name ?? `<buffer ${key}>`;
      if (!fuzzy_matches(name, this.filter_tokens)) continue;

      const bytes = buffer?.config?.size ?? 0;
      buffer_entries.push({ name, bytes });
    }

    for (const [key, texture] of image_map) {
      if (texture?.config?.pool_key) continue;
      texture_count++;

      const name = texture?.config?.name ?? `<texture ${key}>`;
      if (!fuzzy_matches(name, this.filter_tokens)) continue;

      const bytes = compute_texture_size(texture?.config);
      texture_entries.push({ name, bytes });
    }

    return { buffer_entries, texture_entries, buffer_count, texture_count };
  }

  render_entries(entries) {
    if (entries.length === 0) {
      label(
        this.filter_tokens.length > 0 ? "No resources match this filter" : "No resources allocated",
        this.empty_label_config
      );
      return;
    }

    for (let i = 0; i < entries.length; i++) {
      const entry = entries[i];
      const row_config = i % 2 === 0 ? this.row_panel_config : this.alternate_row_panel_config;
      panel(row_config, () => {
        label(entry.name, this.name_label_config);
        label(format_size(entry.bytes), this.size_label_config);
      });
    }
  }

  render() {
    const filter_state = UIContext.input_field_state[filter_input_name];
    const filter_is_focused = filter_state?.is_focused ?? false;
    if (!filter_is_focused && InputProvider.get_action(InputKey.K_r)) {
      this.sort_by_size = !this.sort_by_size;
    }

    this.sync_filter(filter_state?.value ?? "");

    const { buffer_entries, texture_entries, buffer_count, texture_count } = this.collect_entries();

    if (this.sort_by_size) {
      buffer_entries.sort((a, b) => b.bytes - a.bytes);
      texture_entries.sort((a, b) => b.bytes - a.bytes);
    }

    let total_buffer_bytes = 0;
    for (let i = 0; i < buffer_entries.length; i++) {
      total_buffer_bytes += buffer_entries[i].bytes;
    }

    let total_texture_bytes = 0;
    for (let i = 0; i < texture_entries.length; i++) {
      total_texture_bytes += texture_entries[i].bytes;
    }

    const total_bytes = total_buffer_bytes + total_texture_bytes;
    const visible_count = buffer_entries.length + texture_entries.length;
    const resource_count = buffer_count + texture_count;

    panel(this.container_config, () => {
      panel(this.panel_config, () => {
        panel(this.header_panel_config, () => {
          panel(this.accent_line_config, () => {});
          label("GPU MEMORY // ALLOCATIONS", this.title_label_config);
          label(
            `VISIBLE ${visible_count}/${resource_count}  //  BUFFERS + TEXTURES`,
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

          const sort_config = this.sort_by_size
            ? this.active_sort_button_config
            : this.toolbar_button_config;
          if (button(this.sort_by_size ? "SIZE DESC" : "SORT SIZE", sort_config).clicked) {
            this.sort_by_size = !this.sort_by_size;
            current_filter_state.is_focused = false;
          }
        });

        panel(this.summary_panel_config, () => {
          label(
            this.filter_tokens.length > 0 ? "FILTERED GPU MEMORY" : "TOTAL GPU MEMORY",
            this.total_label_config
          );
          label(format_size(total_bytes), this.total_size_label_config);
        });

        panel(this.column_header_config, () => {
          label("GPU RESOURCE", this.column_name_config);
          label("SIZE", this.column_size_config);
        });

        panel(this.section_header_config, () => {
          label(`BUFFERS  ${buffer_entries.length}/${buffer_count}`, this.section_title_config);
          label(format_size(total_buffer_bytes), this.section_total_config);
        });
        this.render_entries(buffer_entries);

        panel(this.section_header_config, () => {
          label(`TEXTURES  ${texture_entries.length}/${texture_count}`, this.section_title_config);
          label(format_size(total_texture_bytes), this.section_total_config);
        });
        this.render_entries(texture_entries);
      });
    });
  }
}
