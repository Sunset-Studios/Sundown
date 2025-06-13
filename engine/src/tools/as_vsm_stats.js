import { DevConsoleTool } from "./dev_console_tool.js";
import { AdaptiveSparseVirtualShadowMaps } from "../renderer/shadows/as_vsm.js";
import { InputProvider } from "../input/input_provider.js";
import { InputKey } from "../input/input_types.js";
import { panel, label } from "../ui/2d/immediate.js";

// Panel configuration for the stats overlay
const stats_panel_config = {
  layout: "column",
  gap: 4,
  y: 25,
  x: 25,
  anchor_x: "right",
  dont_consume_cursor_events: true,
  background_color: "rgba(0, 0, 0, 0.7)",
  width: 500,
  padding: 10,
  border: "1px solid rgb(68, 68, 68)",
  corner_radius: 5,
};

// Label configuration for each stat line
const stats_label_config = {
  text_color: "#fff",
  wrap: true,
  font: "16px monospace",
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

// Value (pointer) label config – link-like but not deep-blue
const pointer_value_label_config = {
  ...stats_label_config,
  text_color: "#4eaaff",
  width: "fit-content",
  x: 0,
  underline_on_hover: true,
  underline_color: "#ee00ff",
};

/**
 * Render a single stat line in "label: value" form, where
 * the value can be styled separately (defaults to link colour).
 */
function stat_row(label_text, value_text, value_config = pointer_value_label_config) {
  panel({ layout: "row", gap: 4, width: "100%", height: 25, anchor_x: "left", x: 0 }, () => {
    label(`${label_text}:`, stats_label_config_small);
    label(value_text, value_config);
 });
}

/**
 * ASVSMStats displays the properties of every active
 * AdaptiveSparseVirtualShadowMaps instance using the
 * immediate-mode UI system.
 */
export class ASVSMStats extends DevConsoleTool {
  is_open = false;
  scene = null;

  update(delta_time) {
    if (!this.is_open) return;
    this.render();
  }

  render() {
    const panel_state = panel(stats_panel_config, () => {
      const instances = AdaptiveSparseVirtualShadowMaps.all_instances || [];
      if (instances.length === 0) {
        label("No AS-VSM instances found.", stats_label_config);
        return;
      }

      for (let i = 0; i < instances.length; i++) {
        const vsm = instances[i];
        label(`AS-VSM ${i}`, { ...stats_label_config, font: "18px monospace", text_color: "#0ff" });
        label("--------------------------------", stats_label_config);

        label(`Tile Size: ${vsm.tile_size} px`, stats_label_config);
        label(`Virtual Dim: ${vsm.virtual_dim} px`, stats_label_config);
        label(`Atlas Size: ${vsm.atlas_size} px`, stats_label_config);
        label(`Max LODs: ${vsm.max_lods}`, stats_label_config);
        label(`Virtual Tiles Per Row: ${vsm.virtual_tiles_per_row} tiles`, stats_label_config);
        label(`Total Virtual Tiles: ${vsm.total_virtual_tiles} tiles`, stats_label_config);
        label(`Physical Tiles Per Row: ${vsm.physical_tiles_per_row} tiles`, stats_label_config);
        label(`Total Physical Tiles: ${vsm.total_physical_tiles} tiles`, stats_label_config);
        label(`Cached Light Count: ${vsm.cached_light_count} lights`, stats_label_config);
        label(`Bitmask U32 Count: ${vsm.bitmask_u32_count}`, stats_label_config);
        label(`Max Tile Requests: ${vsm.max_tile_requests} tiles`, stats_label_config);
        stat_row("Requested Tiles Buffer", `${vsm.requested_tiles_buf} *`);
        stat_row("Histogram Buffer", `${vsm.histogram_buf} *`);
        stat_row("Settings Buffer", `${vsm.settings_buf} *`);
        stat_row("Page Table Buffer", `${vsm.page_table} *`);
        stat_row("Bitmask Buffer", `${vsm.bitmask_buf} *`);
        stat_row("LRU Buffer", `${vsm.lru_buf} *`);
        stat_row("Physical to Virtual Map Buffer", `${vsm.physical_to_virtual_map_buf} *`);
        stat_row("Shadow Atlas", `${vsm.shadow_atlas} *`);

        label("--------------------------------", stats_label_config);
      }
    });

    // Close when clicking outside the panel
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

  toggle() {
    this.is_open = !this.is_open;
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
    if (this.scene && typeof this.scene.show_dev_cursor === "function") {
      this.scene.show_dev_cursor();
    }
  }

  hide() {
    this.is_open = false;
    if (this.scene && typeof this.scene.hide_dev_cursor === "function") {
      this.scene.hide_dev_cursor();
    }
  }

  set_scene(scene) {
    this.scene = scene;
  }
} 