import { DevConsoleTool } from "./dev_console_tool.js";
import { RenderPassFlags, render_pass_flags_to_string } from "../renderer/renderer_types.js";
import { Renderer } from "../renderer/renderer.js";
import { Name } from "../utility/names.js";

// Import the new immediate–mode UI API functions.
import { panel, begin_container, end_container, button, label, UIContext } from "../ui/2d/immediate.js";

const yes_name = "Yes";
const no_name = "No";

const pass_display_label_no_info_text =
  "This pass is not currently loaded and cannot display any details.";

export class RenderPassOrganizer extends DevConsoleTool {
  passes_data = [];
  scene = null;
  is_open = false;
  pos_x = 20;
  pos_y = 50;
  width = 600;
  height = 400;
  is_panel_dragging = false;
  drag_offset_x = 0;
  drag_offset_y = 0;
  dragging_pass_index = null;
  current_pass_details = null;

  init() {
    super.init();
    // If UIContext.canvas_size is set already then adjust our panel position.
    if (UIContext.canvas_size.width && UIContext.canvas_size.height) {
      // For a top–right anchored panel (translate right style into a left offset)
      this.pos_x = UIContext.canvas_size.width - this.width - 20;
      this.height = UIContext.canvas_size.height * 0.8;
    }
  }

  update() {
    super.update();

    const render_graph = Renderer.get().render_graph;

    if (render_graph.is_default_pass_order_ready()) {
      const registered_pass_names = render_graph.registry.render_passes.map(
        (pass) => pass.pass_config.name
      );
      const default_pass_order = render_graph.get_default_pass_order();
      const default_scene_pass_order_need_refresh = registered_pass_names.some(
        (name) => !default_pass_order.includes(name)
      );

      if (default_scene_pass_order_need_refresh) {
        this._save_default_pass_order().then(() => {
          this._save_custom_pass_order();
        });
      }
    }

    if (!this.is_open) {
      return;
    }

    // Update passes data for the current frame.
    this.passes_data = Renderer.get().render_graph.registry.render_passes.map((pass, index) => ({
      index: index,
      name: pass.pass_config.name,
      encoded_name: Name.from(pass.pass_config.name),
      num_inputs: pass.parameters.inputs.length,
      num_outputs: pass.parameters.outputs.length,
      num_attachments: pass.pass_config.attachments?.length || 0,
      has_depth: pass.pass_config.depth_stencil_attachment !== null,
      type: render_pass_flags_to_string(pass.pass_config.flags),
      raw_flags: pass.pass_config.flags,
    }));

    // Rebuild the UI every frame.
    this.render();
  }

  execute() {
    super.execute();
    this.toggle();
  }

  set_scene(scene) {
    this.scene = scene;
  }

  render() {
    console.log(this.height)
    // Build the organizer panel using the new immediate–mode API.
    panel({
      x: this.pos_x,
      y: this.pos_y,
      anchor_x: "right",
      height: this.height,
      width: this.width,
      background_color: "rgba(0, 0, 0, 0.7)",
      border: "1px solid rgb(68, 68, 68)",
      corner_radius: "5px",
      padding: 10,
    }, () => {
      // Draw a header that acts as the draggable area for the panel.
      const header = button("Render Pass Organizer", {
        width: this.width - 20, // subtract horizontal padding
        height: 30,
        background_color: "rgba(0, 0, 0, 0.7)",
        text_color: "#8f8f8f",
        font: "16px monospace",
        draggable: true,
        on_drag_start: (widget_id) => {
          this.is_panel_dragging = true;
          // Store offset from mouse to panel origin.
          this.drag_offset_x = UIContext.input_state.x - this.pos_x;
          this.drag_offset_y = UIContext.input_state.y - this.pos_y;
        },
        on_drag: (widget_id, new_x, new_y) => {
          // Update the panel position.
          this.pos_x = new_x;
          this.pos_y = new_y;
        },
        on_drop: (widget_id, final_x, final_y) => {
          this.is_panel_dragging = false;
        },
      });
      // (You may add additional header styling or a close–button here if desired.)

      // Compute inner container dimensions.
      const content_height = this.height - 30 - 10;
      const content_width = this.width - 20;

      // Create a horizontal container to hold the passes list (left) and details view (right).
      begin_container({
        x: 0,
        y: 40,
        height: content_height,
        width: content_width,
        layout: "row",
        gap: 10,
        padding: 0,
      });

      // ---- Passes Panel (Left) ----
      const passes_panel_width = Math.floor(content_width * 2 / 3);
      begin_container({
        x: 0,
        y: 0,
        width: passes_panel_width,
        height: content_height,
        layout: "column",
        gap: 8,
        padding: 0,
        background_color: "transparent",
      });

      // Title row for passes panel: contains a title label and a reset button.
      begin_container({
        x: 0,
        y: 0,
        width: passes_panel_width,
        height: 30,
        layout: "row",
        gap: 10,
        padding: 0,
      });
      label("Render Passes", {
        font: "16px monospace",
        text_color: "#8f8f8f",
      });
      const reset = button("Reset Order", {
        width: 80,
        height: 30,
        background_color: "#4a1515",
        text_color: "white",
        font: "12px monospace",
      });
      if (reset.clicked) {
        this._reset_pass_order();
      }
      end_container(); // end title row

      // List the passes as draggable buttons.
      const render_graph = Renderer.get().render_graph;
      const current_pass_order = render_graph.get_scene_pass_order();
      for (let i = 0; i < current_pass_order.length; i++) {
        const pass_name = current_pass_order[i];
        let pass = this.passes_data.find((p) => p.name === pass_name);
        let pass_panel_color = "#151515";
        if (pass) {
          if (pass.raw_flags & RenderPassFlags.Present) {
            pass_panel_color = "#4a1515";
          } else if (pass.raw_flags & RenderPassFlags.Compute) {
            pass_panel_color = "#153d4a";
          } else if (pass.raw_flags & RenderPassFlags.Graphics) {
            pass_panel_color = "#154a1d";
          } else if (pass.raw_flags & RenderPassFlags.GraphLocal) {
            pass_panel_color = "#4a3d15";
          }
        }
        const pass_button = button(pass_name, {
          width: passes_panel_width,
          height: 30,
          background_color: pass_panel_color,
          text_color: "white",
          font: "14px monospace",
          draggable: true,
          on_drag_start: (widget_id) => {
            this.dragging_pass_index = i;
          },
          on_drop: (widget_id, dropX, dropY) => {
            // In this simple implementation we assume each pass button takes ~38px (30px height + 8px gap).
            const drop_index = Math.floor(drop_y / 38);
            if (drop_index !== this.dragging_pass_index) {
              // Determine drop direction by comparing dropY to the vertical midpoint of the target item.
              const target_mid = drop_index * 38 + 15;
              const drop_before = drop_y < target_mid;
              this._reorder_passes(this.dragging_pass_index, drop_index, !drop_before);
              this._save_custom_pass_order();
            }
            this.dragging_pass_index = null;
          },
        });
        // If a pass button is clicked (and not dragged) then show its details.
        if (pass_button.clicked) {
          this._show_pass_details(pass_name);
        }
      }
      end_container(); // end passes panel container

      // ---- Details Panel (Right) ----
      const details_panel_width = content_width - passes_panel_width - 10;
      begin_container({
        x: 0,
        y: 0,
        width: details_panel_width,
        height: content_height,
        layout: "column",
        gap: 8,
        padding: 0,
        background_color: "rgb(29,29,29)",
      });
      label("Pass Details", {
        font: "16px monospace",
        text_color: "#8f8f8f",
      });
      if (this.current_pass_details) {
        let pass = this.passes_data.find((p) => p.name === this.current_pass_details);
        if (pass) {
          label("Name: " + pass.name, { font: "14px monospace", text_color: "white" });
          label("Inputs: " + pass.num_inputs, { font: "14px monospace", text_color: "white" });
          label("Outputs: " + pass.num_outputs, { font: "14px monospace", text_color: "white" });
          label("Color Attachments: " + pass.num_attachments, {
            font: "14px monospace",
            text_color: "white",
          });
          label("Has Depth: " + (pass.has_depth ? yes_name : no_name), {
            font: "14px monospace",
            text_color: "white",
          });
          label("Type: " + pass.type, { font: "14px monospace", text_color: "white" });
        } else {
          label(pass_display_label_no_info_text, { font: "14px monospace", text_color: "white" });
        }
      } else {
        label("Select a pass to view details", { font: "14px monospace", text_color: "#888" });
      }
      end_container(); // end details panel

      end_container(); // end horizontal container for panels
    });
  }

  _reorder_passes(from_index, to_index, drop_before) {
    const render_graph = Renderer.get().render_graph;
    const current_pass_order = render_graph.get_scene_pass_order();

    const true_from_index = from_index;
    const true_to_index = to_index;

    const moving_element = current_pass_order[true_from_index];
    current_pass_order.splice(true_from_index, 1);

    let insert_index = true_to_index;
    if (true_from_index < true_to_index && !drop_before) {
      insert_index--;
    } else if (true_from_index > true_to_index && drop_before) {
      insert_index++;
    }

    current_pass_order.splice(insert_index, 0, moving_element);
    render_graph.set_scene_pass_order(current_pass_order);
  }

  async _save_default_pass_order() {
    const render_graph = Renderer.get().render_graph;
    const registered_pass_names = render_graph.registry.render_passes.map(
      (pass) => pass.pass_config.name
    );
    const default_pass_order = render_graph.get_default_pass_order();

    // Use registered_pass_names as the source of truth
    let merged_pass_order = [...registered_pass_names];

    for (let i = 0, merged_index = 0; i < default_pass_order.length; i++) {
      const default_pass = default_pass_order[i];
      if (
        merged_pass_order.length < merged_index ||
        merged_pass_order[merged_index] !== default_pass
      ) {
        const index_in_registered = merged_pass_order.indexOf(default_pass);
        if (index_in_registered === -1) {
          merged_pass_order.splice(merged_index + 1, 0, default_pass);
          merged_index++;
        } else {
          const diff = index_in_registered - i + 1;
          merged_index += diff;
        }
      }
    }

    render_graph.set_default_pass_order(merged_pass_order);

    await render_graph.record_default_pass_order();
  }

  async _save_custom_pass_order() {
    const render_graph = Renderer.get().render_graph;

    const default_scene_pass_order = render_graph.get_default_pass_order();
    const scene_pass_order = render_graph.get_scene_pass_order();

    if (scene_pass_order.length < default_scene_pass_order.length) {
      const missing_passes = default_scene_pass_order
        .map((pass, index) => ({ name: pass, index: index }))
        .filter((pass) => !render_graph.registry.pass_order_map.has(pass.name));

      if (missing_passes.length > 0) {
        for (let i = 0; i < missing_passes.length; i++) {
          scene_pass_order.splice(missing_passes[i].index, 0, missing_passes[i].name);
        }
      }
    }

    render_graph.set_scene_pass_order(scene_pass_order);

    await render_graph.record_custom_pass_order();
  }

  async _reset_pass_order() {
    const render_graph = Renderer.get().render_graph;

    const default_scene_pass_order = [...render_graph.get_default_pass_order()];

    render_graph.set_scene_pass_order(default_scene_pass_order);

    await render_graph.record_custom_pass_order();
  }

  _show_pass_details(pass_name) {
    // Save the currently selected pass so that the details panel can render its info.
    this.current_pass_details = pass_name;
  }

  show() {
    super.show();

    if (this.scene) {
      this.scene.show_dev_cursor();
    }

    const render_graph = Renderer.get().render_graph;
    this.passes_data = render_graph.registry.render_passes.map((pass, index) => ({
      index: index,
      name: pass.pass_config.name,
      encoded_name: Name.from(pass.pass_config.name),
      num_inputs: pass.parameters.inputs.length,
      num_outputs: pass.parameters.outputs.length,
      num_attachments: pass.pass_config.attachments?.length || 0,
      has_depth: pass.pass_config.depth_stencil_attachment !== null,
      type: render_pass_flags_to_string(pass.pass_config.flags),
      raw_flags: pass.pass_config.flags,
    }));

    if (!this.current_pass_details && this.passes_data.length > 0) {
      this.current_pass_details = this.passes_data[0].name;
    }
  }

  hide() {
    super.hide();

    if (this.scene) {
      this.scene.hide_dev_cursor();
    }
  }
}
