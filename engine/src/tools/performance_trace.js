import { DevConsoleTool } from "./dev_console_tool.js";
import { panel, label } from "../ui/2d/immediate.js";
import { is_trace_activated, set_trace_activated } from "../utility/performance.js";

/*
  Panel and label configurations adapted for immediate mode UI.
  Adjust these values as needed.
*/
const trace_panel_config = {
  layout: "column",
  gap: 4,
  y: 25,
  x: 25,
  anchor_x: "right",
  dont_consume_cursor_events: true,
  background_color: "rgba(98, 16, 16, 0.7)",
  padding: 10,
  width: 300,
  border: "1px solid rgb(68, 68, 68)",
  corner_radius: 5,
};

const trace_label_config = {
  text_color: "#fff",
  font: "16px monospace",
  height: 20,
  width: "fit-content",
  text_valign: "middle",
  text_align: "left",
  text_padding: 5,
};

/**
 * PerformanceTrace displays performance trace information using the
 * immediate mode UI framework.
 */
export class PerformanceTrace extends DevConsoleTool {
  scene = null;

  /**
   * Called each frame to update the PerformanceTrace state. If the user clicks
   * outside the trace panel, the panel is hidden.
   */
  update(delta_time) {
    if (!is_trace_activated()) return;
    this.render();
  }

  /**
   * Renders the performance trace panel.
   * This method should be called every frame as part of the render loop.
   */
  render() {
    if (!is_trace_activated()) return;

    panel(trace_panel_config,
      () => {
        label("Performance Tracing...", trace_label_config);
      }
    );
  }

  /**
   * Toggles the display of the performance trace panel.
   */
  execute() {
    this.toggle();
  }

  /**
   * Toggles the display of the performance trace panel.
   */
  toggle() {
    const is_tracing = is_trace_activated();
    set_trace_activated(!is_tracing);
  }

  /**
   * Sets the scene for the performance trace tool.
   * @param {Object} scene - The scene to set.
   */
  set_scene(scene) {
    this.scene = scene;
  }
}
