import { EntityManager } from "../core/ecs/entity.js";
import { DevConsoleTool } from "./dev_console_tool.js";
import { AABBEntityAdapter } from "../core/subsystems/aabb_entity_adapter.js";
import { AABBTreeDebugRenderer } from "../core/subsystems/aabb_debug_renderer.js";
import { InputProvider } from "../input/input_provider.js";
import { InputKey } from "../input/input_types.js";
import { panel, label, button, begin_container, end_container } from "../ui/2d/immediate.js";

const stats_panel_config = {
  layout: "column",
  gap: 4,
  y: 25,
  x: 25,
  anchor_x: "right",
  dont_consume_cursor_events: true,
  background_color: "rgba(0, 0, 0, 0.7)",
  width: 600,
  padding: 10,
  border: "1px solid rgb(68, 68, 68)",
  corner_radius: 5,
};

const stats_label_config = {
  text_color: "#fff",
  x: 0,
  y: 0,
  wrap: true,
  font: "16px monospace",
  width: "100%",
  height: "fit-content",
  text_valign: "middle",
  text_align: "left",
  text_padding: 5,
};

const button_config = {
  y: 0,
  x: 0,
  width: "fit-content",
  font: "bold 16px monospace",
  height: 30,
  background_color: "#FFA500",
  text_color: "#111111",
  corner_radius: 5,
  text_padding: 10,
};

/**
 * CameraInfo displays camera information using the
 * immediate mode UI framework.
 */
export class AABBDebug extends DevConsoleTool {
  is_open = false;
  scene = null;
  aabb_entity_adapter = null;
  aabb_tree_debug_renderer = null;
  show_aabb_tree_debug = false;
  debug_max_depth = 20;

  /**
   * Called each frame to update the MLStats state. If the user clicks
   * outside the stats panel, the panel is hidden.
   */
  update(delta_time) {
    if (!this.is_open) return;
    this.render();
  }

  /**
   * Renders the camera info panel.
   * This method should be called every frame as part of the render loop.
   */
  render() {
    if (!this.aabb_entity_adapter || !this.aabb_tree_debug_renderer) return;

    const stats = this.aabb_entity_adapter.get_stats();

    // Use immediate mode UI panel instead of window
    let panel_state = panel(stats_panel_config, () => {
      // AABB tree stats
      label(`Allocated Nodes: ${stats.allocated_nodes}`, stats_label_config);
      label(`Tree Leaf Nodes: ${stats.leaf_nodes}`, stats_label_config);
      label(`Tree Internal Nodes: ${stats.internal_nodes}`, stats_label_config);
      label(`Tree Depth: ${stats.max_depth}`, stats_label_config);
      label(`Entity Count: ${EntityManager.get_entity_count()}`, stats_label_config);
      label(`Dirty Nodes: ${stats.dirty_nodes}`, stats_label_config);

      // Add buttons for common actions
      begin_container({
        layout: "row",
        x: 0,
        gap: 10,
        height: 40,
        padding_top: 10,
      });

      // Add a button to toggle AABB tree debug visualization
      const debug_vis_text = `AABB Debug: ${this.show_aabb_tree_debug ? "ON" : "OFF"}`;
      const debug_vis_button = button(debug_vis_text, button_config);
      if (debug_vis_button.clicked) {
        this.show_aabb_tree_debug = this.aabb_tree_debug_renderer.toggle_visualization();
      }

      end_container();

      // AABB tree Debug visualization options (only show when debug is enabled)
      if (this.show_aabb_tree_debug) {
        // Node type toggles
        begin_container({
          layout: "row",
          x: 0,
          gap: 10,
          height: 40,
          padding_top: 10,
          padding_bottom: 10,
        });

        const bounds_text = `Bounds: ${this.aabb_tree_debug_renderer.show_bounds ? "ON" : "OFF"}`;
        const bounds_button = button(bounds_text, button_config);
        if (bounds_button.clicked) {
          this.aabb_tree_debug_renderer.toggle_bounds();
        }

        const leaf_nodes_text = `Leaf Nodes: ${this.aabb_tree_debug_renderer.show_leaf_nodes ? "ON" : "OFF"}`;
        const leaf_nodes_button = button(leaf_nodes_text, button_config);
        if (leaf_nodes_button.clicked) {
          this.aabb_tree_debug_renderer.toggle_leaf_nodes();
        }

        const internal_nodes_text = `Internal Nodes: ${this.aabb_tree_debug_renderer.show_internal_nodes ? "ON" : "OFF"}`;
        const internal_nodes_button = button(internal_nodes_text, button_config);
        if (internal_nodes_button.clicked) {
          this.aabb_tree_debug_renderer.toggle_internal_nodes();
        }

        end_container();
      }
    });

    if (this.is_open && InputProvider.get_action(InputKey.B_mouse_left)) {
      if (!panel_state.hovered) {
        this.hide();
      }
      InputProvider.consume_action(InputKey.B_mouse_left);
    }
  }

  /**
   * Toggles the display of the ML stats panel.
   */
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
    this.aabb_entity_adapter = this.scene.get_layer(AABBEntityAdapter);
    this.aabb_tree_debug_renderer = this.scene.get_layer(AABBTreeDebugRenderer);
  }
}
