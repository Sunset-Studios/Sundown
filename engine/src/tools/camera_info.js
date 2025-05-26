import { DevConsoleTool } from "./dev_console_tool.js";
import { SharedViewBuffer } from "../core/shared_data.js";
import { InputProvider } from "../input/input_provider.js";
import { InputKey } from "../input/input_types.js";
import { panel, label } from "../ui/2d/immediate.js";

/*
  Panel and label configurations adapted for immediate mode UI.
  Adjust these values as needed.
*/
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
  wrap: true,
  font: "16px monospace",
  width: "100%",
  height: "fit-content",
  text_valign: "middle",
  text_align: "left",
  text_padding: 5,
};

/**
 * CameraInfo displays camera information using the
 * immediate mode UI framework.
 */
export class CameraInfo extends DevConsoleTool {
  is_open = false;
  scene = null;

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
    let panel_state = panel(stats_panel_config, () => {
      const count = SharedViewBuffer.get_view_data_count();
      for (let i = 0; i < count; i++) {
        const view_data = SharedViewBuffer.get_view_data(i);
        label(
          `Camera ${i} Position: ${view_data.view_position[0]}, ${view_data.view_position[1]}, ${view_data.view_position[2]}`,
          stats_label_config
        );
        label(
          `Camera ${i} Rotation: ${view_data.view_rotation[0]}, ${view_data.view_rotation[1]}, ${view_data.view_rotation[2]}, ${view_data.view_rotation[3]}`,
          stats_label_config
        );
        label(`Camera ${i} FOV: ${view_data.fov}`, stats_label_config);
        label(`Camera ${i} Near: ${view_data.near}`, stats_label_config);
        label(`Camera ${i} Far: ${view_data.far}`, stats_label_config);
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
  }
}
