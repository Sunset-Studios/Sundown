import { DevConsoleTool } from "./dev_console_tool.js";
import { panel, label } from "../ui/2d/immediate.js";
import { InputProvider } from "../input/input_provider.js";
import { InputKey } from "../input/input_types.js";
import { FragmentGpuBuffer } from "../core/ecs/solar/memory.js";
import { BVH } from "../acceleration/bvh.js";
import { bytes_to_mb } from "../utility/math.js";

// Panel configuration
const stats_panel_config = {
  layout: "column",
  gap: 4,
  y: 25,
  x: 25,
  anchor_x: "right",
  dont_consume_cursor_events: true,
  background_color: "rgba(0, 0, 0, 0.7)",
  width: 520,
  padding: 10,
  border: "1px solid rgb(68, 68, 68)",
  corner_radius: 5,
};

const header_label_config = {
  text_color: "#0ff",
  font: "18px monospace",
  width: "100%",
  height: "fit-content",
  text_valign: "middle",
  text_align: "left",
  text_padding: 5,
};

const row_config = {
  layout: "row",
  gap: 4,
  width: "100%",
  height: 24,
  anchor_x: "left",
  x: 0,
};

const row_left_label_config = {
  x: 0,
  text_color: "#fff",
  font: "14px monospace",
  width: "70%",
  height: 24,
  text_valign: "middle",
  text_align: "left",
  text_padding: 5,
};

const row_right_label_config = {
  x: 0,
  text_color: "#fff",
  font: "14px monospace",
  width: "30%",
  height: 24,
  text_valign: "middle",
  text_align: "right",
  text_padding: 5,
};

const divider_config = {
  width: "100%",
  height: 3,
  x: 0,
  y: 0,
  background_color: "rgba(0.2, 0.2, 0.2, 0.5)",
};

// Mapping for future memory debug view types
const MemoryDebugViewType = {
  SolarFragments: "solar",
  BVH: "bvh",
};

export class DebugMemory extends DevConsoleTool {
  is_open = false;
  scene = null;
  current_view_type = null;

  update(delta_time) {
    if (!this.is_open) return;
    this.render();
  }

  execute(args) {
    if (args && args.length > 0) {
      const requested = String(args[0]).toLowerCase();
      // Only one view for now; keep extensible
      if (requested === MemoryDebugViewType.SolarFragments) {
        this.current_view_type = MemoryDebugViewType.SolarFragments;
      } else if (requested === MemoryDebugViewType.BVH) {
        this.current_view_type = MemoryDebugViewType.BVH;
      }

      if (this.current_view_type !== null) {
        this.show();
      }
    }
  }

  render() {
    const panel_state = panel(stats_panel_config, () => {
      if (this.current_view_type === MemoryDebugViewType.SolarFragments) {
        this._render_solar_memory_stats();
      } else if (this.current_view_type === MemoryDebugViewType.BVH) {
        this._render_bvh_memory_stats();
      }
    });

    // Close on outside click
    if (this.is_open && InputProvider.get_action(InputKey.B_mouse_left)) {
      if (!panel_state.hovered) {
        this.hide();
      }
      InputProvider.consume_action(InputKey.B_mouse_left);
    }
  }

  _render_solar_memory_stats() {
    label("Solar Fragment GPU Buffers", header_label_config);
    panel(divider_config);

    const buffers = FragmentGpuBuffer.all_buffers || [];
    let total_bytes = 0;
    for (let i = 0; i < buffers.length; i++) {
      const buf = buffers[i];
      const name = buf?.name ?? "<unnamed>";
      const bytes = buf?.buffer?.config?.size ?? 0;
      total_bytes += bytes;

      panel(row_config, () => {
        label(String(name), row_left_label_config);
        label(`${bytes_to_mb(bytes)} MB`, row_right_label_config);
      });
    }

    panel(divider_config);
    panel(row_config, () => {
      label("Total", { ...row_left_label_config, text_color: "#0ff" });
      label(`${bytes_to_mb(total_bytes)} MB`, { ...row_right_label_config, text_color: "#0ff" });
    });
  }

  _render_bvh_memory_stats() {
    label("BVH Memory Stats", header_label_config);
    panel(divider_config);

    // Ensure buffers exist if BVH subsystem was never touched
    if (!BVH.is_initialized) {
      // Do not auto-initialize; just show empty if not used yet
    }

    const entries = [];
    const push_buf = (display_name, buf) => {
      if (!buf) return;
      const size_bytes = buf?.config?.size ?? 0;
      entries.push({ name: display_name, bytes: size_bytes });
    };

    push_buf("Scene Bounds", BVH.scene_bounds_buffer);
    push_buf("Node Bounds", BVH.bounds_buffer);
    push_buf("Morton Codes", BVH.morton_codes_buffer);
    push_buf("Temp Morton Codes", BVH.temp_morton_codes_buffer);
    push_buf("Sorted Indices", BVH.sorted_indices_buffer);
    push_buf("Temp Sorted Indices", BVH.temp_sorted_indices_buffer);
    push_buf("BVH2 Nodes", BVH.bvh2_nodes_buffer);
    push_buf("BVH4 Nodes", BVH.bvh4_nodes_buffer);
    push_buf("Onesweep Data", BVH.onesweep_data_buffer);
    push_buf("Node Counters", BVH.node_counters_buffer);
    push_buf("Clusters", BVH.clusters_buffer);
    push_buf("Parent Indices", BVH.parent_idx_buffer);

    let total_bytes = 0;
    for (let i = 0; i < entries.length; i++) {
      const e = entries[i];
      total_bytes += e.bytes;
      panel(row_config, () => {
        label(e.name, row_left_label_config);
        label(`${bytes_to_mb(e.bytes)} MB`, row_right_label_config);
      });
    }

    panel(divider_config);
    panel(row_config, () => {
      label("Total", { ...row_left_label_config, text_color: "#0ff" });
      label(`${bytes_to_mb(total_bytes)} MB`, { ...row_right_label_config, text_color: "#0ff" });
    });
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
