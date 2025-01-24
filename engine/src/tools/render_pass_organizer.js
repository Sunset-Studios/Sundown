import { InputProvider } from "../input/input_provider.js";
import { InputKey, InputRange } from "../input/input_types.js";
import { DevConsoleTool } from "./dev_console_tool.js";
import { RenderPassFlags, render_pass_flags_to_string } from "../renderer/renderer_types.js";
import { Renderer } from "../renderer/renderer.js";
import { Element } from "../ui/2d/element.js";
import { Panel } from "../ui/2d/panel.js";
import { Label } from "../ui/2d/label.js";
import { Name } from "../utility/names.js";

const none_name = "none";
const yes_name = "Yes";
const no_name = "No";

const render_pass_organizer_name = "render_pass_organizer";
const passes_panel_name = "passes_panel";
const details_panel_name = "details_panel";
const title_name = "title";
const details_name = "details";
const select_prompt_name = "select_prompt";
const pass_display_label_name = "Name";
const pass_display_label_inputs_name = "Inputs";
const pass_display_label_outputs_name = "Outputs";
const pass_display_label_color_attachments_name = "Color Attachments";
const pass_display_label_has_depth_name = "Has Depth";
const pass_display_label_type_name = "Type";
const pass_display_label_no_info_name = "No Info";
const pass_display_label_no_info_text =
  "This pass is not currently loaded and cannot display any details.";
const reset_button_name = "reset_order_button";

const organizer_panel_config = {
  is_draggable: true,
  drag_hold_delay: 500,
  allows_cursor_events: true,
  dont_consume_cursor_events: true,
  style: {
    position: "absolute",
    top: "50px",
    right: "20px",
    width: "600px",
    height: "80%",
    backgroundColor: "rgba(0, 0, 0, 0.7)",
    display: "flex",
    padding: "10px",
    border: "1px solid rgb(68, 68, 68)",
    borderRadius: "5px",
    fontFamily: "monospace",
    cursor: "move",
  },
};
const passes_panel_config = {
  is_scrollable: true,
  scroll_amount: 1,
  allows_cursor_events: true,
  dont_consume_cursor_events: true,
  style: {
    flex: "2",
    backgroundColor: "transparent",
    padding: "10px",
    overflowY: "auto",
    scrollbarWidth: "none",
  },
};
const reset_button_config = {
  allows_cursor_events: true,
  style: {
    backgroundColor: "#4a1515", // Dark red to match theme
    color: "white",
    padding: "4px 8px",
    border: "none",
    borderRadius: "3px",
    marginLeft: "10px",
    cursor: "pointer",
    fontSize: "12px",
    fontFamily: "monospace",
    transition: "background-color 0.2s ease",
  },
  hover_style: {
    backgroundColor: "#661c1c", // Slightly lighter red on hover
  },
};
const render_passes_label_config = {
  text: "Render Passes",
  style: {
    display: "flex",
    alignItems: "center",
    marginBottom: "10px",
    fontSize: "16px",
    fontWeight: "bold",
    color: "#8f8f8f",
  },
};
const details_panel_config = {
  style: {
    flex: "1",
    backgroundColor: "rgb(29, 29, 29)",
    padding: "10px",
  },
};
const details_label_config = {
  text: "Pass Details",
  style: {
    display: "block",
    marginBottom: "10px",
    fontSize: "16px",
    fontWeight: "bold",
    color: "#8f8f8f",
  },
};
const select_prompt_config = {
  text: "Select a pass to view details",
  style: {
    display: "block",
    color: "#888",
  },
};
const detail_row_config = {
  style: {
    marginBottom: "8px",
  },
};

export class RenderPassOrganizer extends DevConsoleTool {
  passes_data = [];
  organizer_panel = null;
  details_panel = null;
  passes_panel = null;
  scene = null;
  pass_list_needs_update = false;

  init() {
    super.init();

    this._init_ui();
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
          this._mark_passes_list_dirty();
        });
      }
    }

    if (!this.is_open) {
      return;
    }

    const input_provider = InputProvider.get();

    if (input_provider.get_action(InputKey.B_mouse_left)) {
      const mouse_x = input_provider.get_range(InputRange.M_xabs);
      const mouse_y = input_provider.get_range(InputRange.M_yabs);

      if (!this.organizer_panel.is_inside(mouse_x, mouse_y)) {
        this.hide();
      }

      const pass_order = render_graph.get_scene_pass_order();
      for (let i = 0; i < this.passes_panel.children.length; i++) {
        if (this.passes_panel.children[i].is_inside(mouse_x, mouse_y)) {
          const pass_index = this.passes_panel.children[i].pass_index;
          if (pass_index !== undefined) {
            this._show_pass_details(pass_order[pass_index]);
            input_provider.consume_action(InputKey.B_mouse_left);
          }
          break;
        }
      }
    }

    if (!this.is_open) {
      return;
    }

    if (this.pass_list_needs_update) {
      this._update_passes_list();
    }
  }

  execute() {
    super.execute();
    this.toggle();
  }

  set_scene(scene) {
    this.scene = scene;
  }

  _init_ui() {
    this.organizer_panel = Panel.create(render_pass_organizer_name, organizer_panel_config);

    this.organizer_panel.config.on_drag = (element) => {
      const input_provider = InputProvider.get();
      const mouse_x = input_provider.get_range(InputRange.M_xabs);
      const mouse_y = input_provider.get_range(InputRange.M_yabs);

      element.dom.style.right = "auto"; // Remove right positioning to allow free movement
      element.dom.style.left = `${mouse_x - element.drag_offset_x}px`;
      element.dom.style.top = `${mouse_y - element.drag_offset_y}px`;
    };

    this.passes_panel = Panel.create(passes_panel_name, passes_panel_config, [
      Label.create(title_name, render_passes_label_config),
    ]);

    this.details_panel = Panel.create(details_panel_name, details_panel_config, [
      Label.create(details_name, details_label_config),
      Label.create(select_prompt_name, select_prompt_config),
    ]);

    this.organizer_panel.add_child(this.passes_panel);
    this.organizer_panel.add_child(this.details_panel);

    const view_root = Element.get_view_root();
    view_root.add_child(this.organizer_panel);

    this.organizer_panel.is_visible = false;
  }

  _create_pass_element(pass_name, index) {
    const pass = this.passes_data.find((p) => p.name === pass_name);

    let pass_panel_color;
    if (pass && pass.raw_flags & RenderPassFlags.Present) {
      pass_panel_color = "#4a1515"; // Dark red for present
    } else if (pass && pass.raw_flags & RenderPassFlags.Compute) {
      pass_panel_color = "#153d4a"; // Dark blue for compute
    } else if (pass && pass.raw_flags & RenderPassFlags.Graphics) {
      pass_panel_color = "#154a1d"; // Dark green for graphics
    } else if (pass && pass.raw_flags & RenderPassFlags.GraphLocal) {
      pass_panel_color = "#4a3d15"; // Dark yellow/gold for graph local
    } else {
      pass_panel_color = "#151515"; // Dark gray for unknown
    }

    const pass_panel = Panel.create(
      `pass_${index}`,
      {
        is_draggable: true,
        drag_hold_delay: 500,
        allows_cursor_events: true,
        dont_consume_cursor_events: true,
        accepts_drops: true,
        style: {
          backgroundColor: pass_panel_color,
          padding: "8px",
          marginBottom: "4px",
          cursor: "move",
          userSelect: "none",
          display: "flex",
          alignItems: "center",
          gap: "10px",
          color: "white",
          boxShadow: "0px 0px 3px 0px #2d2d2d",
          transition: "all 0.2s ease",
        },
        hover_style: {
          backgroundColor: "rgb(39 39 39)",
          boxShadow: "0px 0px 3px 0px #2d2d2d",
          transition: "all 0.2s ease",
        },
      },
      [
        Label.create(`name_${index}`, {
          text: pass_name,
          style: {
            flex: "1",
          },
        }),
      ]
    );

    pass_panel.pass_index = index;

    {
      pass_panel.config.on_drag_start = (element) => {
        element.dom.style.opacity = "0.4";
        element.dom.style.zIndex = "1000";
      };

      pass_panel.config.on_drag_end = (element) => {
        element.dom.style.opacity = "1";
        element.dom.style.zIndex = "auto";
        this._remove_drag_indicators();

        element.dom.style.left = element.original_position.left;
        element.dom.style.top = element.original_position.top;
      };

      pass_panel.config.on_drag = (element, target) => {
        if (target) {
          const target_rect = target.rect;
          const mouse_y = InputProvider.get().get_range(InputRange.M_yabs);
          const drop_before = mouse_y > target_rect.top + target_rect.height / 2;

          this._remove_drag_indicators();

          target.dom.style.borderTop = !drop_before ? "2px solid #00ff00" : "none";
          target.dom.style.borderBottom = drop_before ? "2px solid #00ff00" : "none";
        }
      };

      pass_panel.config.on_drop = (element, target) => {
        const from_index = element.pass_index;
        const to_index = target.pass_index;

        if (from_index === to_index) return;

        const target_rect = target.rect;
        const mouse_y = InputProvider.get().get_range(InputRange.M_yabs);
        const drop_before = mouse_y < target_rect.top + target_rect.height / 2;

        element.dom.style.opacity = "1";
        element.dom.style.zIndex = "auto";
        element.dom.style.left = element.original_position.left;
        element.dom.style.top = element.original_position.top;

        this._remove_drag_indicators();
        this._reorder_passes(from_index, to_index, !drop_before);
        this._mark_passes_list_dirty();
        this._save_custom_pass_order();
      };
    }

    return pass_panel;
  }

  _remove_drag_indicators() {
    for (let i = 0; i < this.passes_panel.children.length; i++) {
      this.passes_panel.children[i].dom.style.borderTop = none_name;
      this.passes_panel.children[i].dom.style.borderBottom = none_name;
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

  _show_pass_details(pass_name) {
    const pass = this.passes_data.find((p) => p.name === pass_name);

    this.details_panel.clear_children();

    const details = [[pass_display_label_name, pass_name]];

    if (pass) {
      details.push([pass_display_label_inputs_name, pass.num_inputs]);
      details.push([pass_display_label_outputs_name, pass.num_outputs]);
      details.push([pass_display_label_color_attachments_name, pass.num_attachments]);
      details.push([pass_display_label_has_depth_name, pass.has_depth ? yes_name : no_name]);
      details.push([pass_display_label_type_name, pass.type]);
    } else {
      details.push([pass_display_label_no_info_name, pass_display_label_no_info_text]);
    }

    for (let i = 0; i < details.length; i++) {
      const detail_row = Panel.create(`detail_${i}`, detail_row_config, [
        Label.create(`${i}_label`, {
          text: `${details[i][0]}:`,
          style: {
            display: "inline-block",
            width: "120px",
            fontWeight: "bold",
            color: "white",
          },
        }),
        Label.create(`${i}_value`, {
          text: details[i][1].toString(),
          style: {
            display: "inline-block",
            color: "white",
          },
        }),
      ]);
      this.details_panel.add_child(detail_row);
    }
  }

  _mark_passes_list_dirty() {
    this.pass_list_needs_update = true;
  }

  _update_passes_list() {
    this.passes_panel.clear_children();

    // Create title panel with reset button
    const title_panel = Panel.create("title_panel", {
      style: {
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        marginBottom: "10px",
      },
    });

    // Add the title label
    title_panel.add_child(Label.create(title_name, render_passes_label_config));

    // Add the reset button
    const reset_button = Panel.create(reset_button_name, reset_button_config, [
      Label.create("reset_text", {
        text: "Reset Order",
        style: {
          color: "inherit",
        },
      }),
    ]);
    reset_button.on("selected", () => {
      this._reset_pass_order();
      this._mark_passes_list_dirty();
    });

    title_panel.add_child(reset_button);

    this.passes_panel.add_child(title_panel);

    const render_graph = Renderer.get().render_graph;

    // Add pass elements in order
    const current_pass_order = render_graph.get_scene_pass_order();
    for (let i = 0; i < current_pass_order.length; i++) {
      const pass_name = current_pass_order[i];
      this.passes_panel.add_child(this._create_pass_element(pass_name, i));
    }

    this.pass_list_needs_update = false;
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

  show() {
    super.show();

    this.organizer_panel.is_visible = true;

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

    this._mark_passes_list_dirty();
  }

  hide() {
    super.hide();

    this.organizer_panel.is_visible = false;

    if (this.scene) {
      this.scene.hide_dev_cursor();
    }
  }
}
