import { DevConsoleTool } from "./dev_console_tool.js";
import { panel, label } from "../ui/2d/immediate.js";
import { ResourceCache } from "../renderer/resource_cache.js";
import { CacheTypes } from "../renderer/renderer_types.js";
import { InputProvider } from "../input/input_provider.js";
import { InputKey } from "../input/input_types.js";
import { Texture } from "../renderer/texture.js";

const MB_THRESHOLD = 0.01; // Below this MB value, show KB instead

/**
 * Converts bytes to a human-readable size string.
 * Shows KB for sizes below 0.01 MB, otherwise shows MB.
 * @param {number} bytes - The number of bytes.
 * @returns {string} The formatted size string with unit.
 */
function format_size(bytes) {
  const mb = bytes / (1024.0 * 1024.0);
  if (mb < MB_THRESHOLD) {
    const kb = bytes / 1024.0;
    return `${kb.toFixed(2)} KB`;
  }
  return `${mb.toFixed(2)} MB`;
}

/**
 * Computes the total size in bytes for a texture, including all mip levels.
 * @param {Object} config - The texture configuration.
 * @returns {number} The total size in bytes.
 */
function compute_texture_size(config) {
  if (!config) return 0;

  const bytes_per_pixel = Texture.stride_from_format(config.format);
  const depth = config.depth || 1;
  const mip_levels = config.mip_levels || 1;

  let total_bytes = 0;
  for (let mip = 0; mip < mip_levels; mip++) {
    const w = Math.max(1, (config.width || 1) >> mip);
    const h = Math.max(1, (config.height || 1) >> mip);
    total_bytes += w * h * depth * bytes_per_pixel;
  }

  return total_bytes;
}

export class GPUMemoryView extends DevConsoleTool {
  is_open = false;
  sort_by_size = false;

  container_config = {
    x: 0,
    y: 25,
    width: 550,
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
    background_color: "rgba(32, 15, 32, 0.85)",
  };

  section_header_config = {
    x: 0,
    y: 0,
    width: "100%",
    height: 30,
    font: "600 14px monospace",
    text_color: "#88ccff",
    text_valign: "middle",
    text_align: "left",
    text_padding: 6,
    background_color: "rgba(40, 60, 80, 0.5)",
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
    width: "75%",
    font: "14px monospace",
    text_color: "#ccc",
    text_valign: "middle",
    text_align: "left",
  };

  size_label_cfg = {
    x: 0,
    y: 0,
    width: "25%",
    height: "100%",
    font: "14px monospace",
    text_color: "#ccc",
    text_align: "right",
    text_valign: "middle",
  };

  total_label_cfg = {
    x: 0,
    y: 0,
    width: "75%",
    height: "100%",
    font: "600 14px monospace",
    text_color: "#ffcc88",
    text_valign: "middle",
    text_align: "left",
  };

  total_size_label_cfg = {
    x: 0,
    y: 0,
    width: "25%",
    height: "100%",
    font: "600 14px monospace",
    text_color: "#ffcc88",
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
    if (InputProvider.get_action(InputKey.K_r)) {
      this.sort_by_size = !this.sort_by_size;
    }

    const buffer_map = ResourceCache.get().fetch_all(CacheTypes.BUFFER);
    const image_map = ResourceCache.get().fetch_all(CacheTypes.IMAGE);

    // Collect buffer entries
    const buffer_entries = [];
    for (const [key, buffer] of buffer_map) {
      const name = buffer?.config?.name ?? `<buffer ${key}>`;
      const bytes = buffer?.config?.size ?? 0;
      buffer_entries.push({ name, bytes });
    }

    // Collect texture entries
    const texture_entries = [];
    for (const [key, texture] of image_map) {
      if (texture?.config?.pool_key) continue;
      const name = texture?.config?.name ?? `<texture ${key}>`;
      const bytes = compute_texture_size(texture?.config);
      texture_entries.push({ name, bytes });
    }

    // Sort if enabled
    if (this.sort_by_size) {
      buffer_entries.sort((a, b) => b.bytes - a.bytes);
      texture_entries.sort((a, b) => b.bytes - a.bytes);
    }

    // Calculate totals
    let total_buffer_bytes = 0;
    for (let i = 0; i < buffer_entries.length; i++) {
      total_buffer_bytes += buffer_entries[i].bytes;
    }

    let total_texture_bytes = 0;
    for (let i = 0; i < texture_entries.length; i++) {
      total_texture_bytes += texture_entries[i].bytes;
    }

    const total_bytes = total_buffer_bytes + total_texture_bytes;

    panel(this.container_config, () => {
      panel(this.panel_config, () => {
        // Header
        panel(this.header_panel_config, () => {
          const header_text = `GPU Memory (MB)  [R: sort ${this.sort_by_size ? "ON" : "OFF"}]`;
          label(header_text, this.header_label_config);
        });
        panel(this.divider_config, () => {});

        // Buffers section
        label(`Buffers (${buffer_entries.length})`, this.section_header_config);
        for (let i = 0; i < buffer_entries.length; i++) {
          const entry = buffer_entries[i];
          panel(this.row_panel_config, () => {
            label(entry.name, this.name_label_cfg);
            label(format_size(entry.bytes), this.size_label_cfg);
          });
        }

        // Buffer total
        panel(this.row_panel_config, () => {
          label("Buffer Total", this.total_label_cfg);
          label(format_size(total_buffer_bytes), this.total_size_label_cfg);
        });

        panel(this.divider_config, () => {});

        // Textures section
        label(`Textures (${texture_entries.length})`, this.section_header_config);
        for (let i = 0; i < texture_entries.length; i++) {
          const entry = texture_entries[i];
          panel(this.row_panel_config, () => {
            label(entry.name, this.name_label_cfg);
            label(format_size(entry.bytes), this.size_label_cfg);
          });
        }

        // Texture total
        panel(this.row_panel_config, () => {
          label("Texture Total", this.total_label_cfg);
          label(format_size(total_texture_bytes), this.total_size_label_cfg);
        });

        panel(this.divider_config, () => {});

        // Grand total
        panel(this.row_panel_config, () => {
          label("Grand Total", { ...this.total_label_cfg, text_color: "#88ff88" });
          label(format_size(total_bytes), { ...this.total_size_label_cfg, text_color: "#88ff88" });
        });
      });
    });
  }
}
