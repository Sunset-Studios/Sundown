import { DevConsoleTool } from "./dev_console_tool.js";
import { InputProvider } from "../input/input_provider.js";
import { InputKey } from "../input/input_types.js";
import { RenderPassFlags, render_pass_flags_to_string } from "../renderer/renderer_types.js";
import { Renderer } from "../renderer/renderer.js";
import { ResizableBitArray } from "../memory/container.js";
import { Name } from "../utility/names.js";

// Import the new immediate–mode UI API functions.
import { panel, button, label, UIContext } from "../ui/2d/immediate.js";

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
  panel_height = 400;
  dragging_pass_index = null;
  current_drag_indicator = null;
  current_pass_details = null;
  current_hovered_item = null;
  pass_button_hovered_bit_array = new ResizableBitArray();

  panel_config = {
    x: 0,
    y: 0,
    anchor_x: "right",
    height: 0,
    width: 0,
    background_color: "rgba(0, 0, 0, 0.7)",
    border: "1px solid rgb(68, 68, 68)",
    corner_radius: 5,
    padding: 10,
    draggable: true,
    drag_delay: 0.6,
    layout: "column",
    gap: 5,
    clip: true,
    on_drag_start: () => {},
    on_drag: (new_x, new_y) => {
      this.pos_x += new_x;
      this.pos_y += new_y;
    },
    on_drop: (final_x, final_y) => {},
  };
  organizer_header_config = {
    y: 0,
    width: 0,
    height: 30,
    background_color: "rgba(0, 0, 0, 0.7)",
    text_color: "#8f8f8f",
    font: "18px monospace",
  };
  content_panel_config = {
    x: 0,
    y: 10,
    height: 0,
    width: 0,
    layout: "row",
    gap: 10,
    padding: 0,
    clip: true,
  };
  passes_panel_config = {
    x: 0,
    y: 0,
    width: 0,
    height: 0,
    layout: "column",
    gap: 8,
    padding: 0,
    background_color: "transparent",
    scrollable: true,
    scroll_speed: 35,
    clip: true,
  };
  title_row_config = {
    x: 0,
    y: 0,
    width: 0,
    height: 30,
    layout: "row",
    gap: 10,
    padding: 0,
  };
  render_passes_label_config = {
    x: 0,
    width: 0,
    height: 30,
    padding_left: 100,
    text_valign: "middle",
    text_align: "center",
    font: "bold 16px monospace",
    text_color: "#8f8f8f",
  };
  reset_button_config = {
    x: 0,
    anchor_x: "right",
    width: 80,
    height: 30,
    background_color: "#4a1515",
    text_color: "white",
    font: "12px monospace",
  };
  details_panel_config = {
    x: 0,
    y: 0,
    width: 0,
    height: 0,
    layout: "column",
    gap: 8,
    padding: 15,
    background_color: "rgb(29,29,29)",
    scrollable: true,
    clip: true,
  };
  pass_details_label_config = {
    y: 0,
    x: 0,
    width: '100%',
    height: 'fit-content',
    font: "16px monospace",
    text_color: "#8f8f8f",
    wrap: true,
  };
  pass_details_info_label_config = {
    y: 0,
    x: 0,
    width: 'fit-content',
    height: 'fit-content',
    font: "14px monospace",
    text_color: "white",
    wrap: true,
  };

  init() {
    super.init();
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
    // Build the organizer panel using the new immediate–mode API.
    this.panel_height = UIContext.canvas_size.height * 0.9;

    if (this.pos_x === 0) {
      // For a top–right anchored panel (translate right style into a left offset)
      this.pos_x = this.width - 20;
    }

    if (this.pos_y === 0) {
      this.pos_y = "15%";
    }

    this.panel_config.height = this.panel_height;
    this.panel_config.width = this.width;
    this.panel_config.x = this.pos_x;
    this.panel_config.y = this.pos_y;

    let panel_state = panel(this.panel_config, () => {
      // Draw a header that acts as the draggable area for the panel.
      this.organizer_header_config.width = this.width - 20;
      button("Render Pass Organizer", this.organizer_header_config);

      // Compute inner container dimensions.
      const content_height = this.panel_height - 50;
      const content_width = this.width - 20;

      this.content_panel_config.height = content_height;
      this.content_panel_config.width = content_width;

      // Create a horizontal container to hold the passes list (left) and details view (right).
      panel(this.content_panel_config, () => {
        // ---- Passes Panel (Left) ----
        const passes_panel_width = Math.floor((content_width * 2) / 3);

        this.passes_panel_config.width = passes_panel_width;
        this.passes_panel_config.height = content_height;

        panel(this.passes_panel_config, () => {
          // Title row for passes panel: contains a title label and a reset button.
          this.title_row_config.width = passes_panel_width;

          panel(this.title_row_config, () => {
            this.render_passes_label_config.width = passes_panel_width - 80;
            label("Render Passes", this.render_passes_label_config);

            const reset = button("Reset Order", this.reset_button_config);
            if (reset.clicked) {
              this._reset_pass_order();
            }
          });

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
            
            if (this.pass_button_hovered_bit_array.get(i)) {
              pass_panel_color = "#2a2a2a";
            }

            // Build the button configuration.
            const button_config = {
              y: 0,
              width: passes_panel_width,
              height: 30,
              background_color: pass_panel_color,
              text_color: "white",
              font: "14px monospace",
              draggable: true,
              drag_delay: 0.4,
              on_drag_start: () => {
                this.dragging_pass_index = i;
                this.current_drag_indicator = null;
              },
              on_drag_over: (x, y, width, height) => {
                const candidate_mid = y + height / 2;
                const drag_position = UIContext.input_state.y < candidate_mid ? "top" : "bottom";
                this.current_drag_indicator = {
                  index: i,
                  position: drag_position
                };
              },
              on_drop: (drop_x, drop_y) => {
                if (this.current_drag_indicator.index !== this.dragging_pass_index) {
                  const drop_before = this.current_drag_indicator.position === "top";
                  this._reorder_passes(this.dragging_pass_index, this.current_drag_indicator.index, !drop_before);
                  this._save_custom_pass_order();
                }
                this.dragging_pass_index = null;
                this.current_drag_indicator = null;
              },
            };
            
            // Optionally override the border style to show a drag indicator.
            // If the current drag indicator points at this index (top insertion) or if the candidate
            // drop index is after this button (bottom insertion), modify the button border.
            if (this.current_drag_indicator && i === this.current_drag_indicator.index) {
              if (this.current_drag_indicator.position === "top") {
                // Show an indicator at the top of this button.
                button_config.border_top = "2px dashed red";
              } else if (this.current_drag_indicator.position === "bottom") {
                // Show an indicator at the bottom of this button.
                button_config.border_bottom = "2px dashed red";
              }
            }
            
            const pass_button = button(pass_name, button_config);
            
            // If a pass button is clicked (and not dragged) then show its details.
            if (pass_button.clicked) {
              this._show_pass_details(pass_name);
            }

            this.pass_button_hovered_bit_array.set(i, pass_button.hovered);
          }
        });

        // ---- Details Panel (Right) ----
        this.details_panel_config.width = content_width - passes_panel_width - 10;
        this.details_panel_config.height = content_height;
        panel(this.details_panel_config, () => {
          label("Pass Details", this.pass_details_label_config);
          if (this.current_pass_details) {
            let pass = this.passes_data.find((p) => p.name === this.current_pass_details);
            if (pass) {
              label(pass.name, this.pass_details_info_label_config);
              label("Inputs: " + pass.num_inputs, this.pass_details_info_label_config);
              label("Outputs: " + pass.num_outputs, this.pass_details_info_label_config);
              label(
                "Color Attachments: " + pass.num_attachments,
                this.pass_details_info_label_config
              );
              label(
                "Has Depth: " + (pass.has_depth ? yes_name : no_name),
                this.pass_details_info_label_config
              );
              label("Type: " + pass.type, this.pass_details_info_label_config);
            } else {
              label(pass_display_label_no_info_text, this.pass_details_info_label_config);
            }
          } else {
            label("Select a pass to view details", this.pass_details_info_label_config);
          }
        });
      });
    });

    if (InputProvider.get_action(InputKey.B_mouse_left) && !panel_state.hovered) {
      this.hide();
    }
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
