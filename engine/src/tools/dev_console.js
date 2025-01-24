import { SimulationLayer } from "../core/simulation_layer.js";
import { InputProvider } from "../input/input_provider.js";
import { InputRange, InputKey } from "../input/input_types.js";
import { Element } from "../ui/2d/element.js";
import { Input } from "../ui/2d/input.js";
import { Panel } from "../ui/2d/panel.js";
import { Label } from "../ui/2d/label.js";
import { RenderPassOrganizer } from "./render_pass_organizer.js";

const console_panel_name = "dev_console";
const input_name = "console_input";
const keydown_event_name = "keydown";
const click_event_name = "click";
const visible_name = "visible";
const hidden_name = "hidden";
const display_none_name = "none";
const display_block_name = "block";

const enter_key = "Enter";
const escape_key = "Escape";
const arrow_up_key = "ArrowUp";
const arrow_down_key = "ArrowDown";
const tab_key = "Tab";
const equal_key = "=";

const console_panel_config = {
  style: {
    position: "absolute",
    bottom: "125px",
    left: "50%",
    transform: "translateX(-50%)",
    width: "600px",
    visibility: "hidden",
    backgroundColor: "rgba(0, 0, 0, 0.8)",
    padding: "8px",
    borderRadius: "4px",
    border: "1px solid #444",
  },
};
const input_config = {
  type: "text",
  style: {
    width: "100%",
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    border: "none",
    padding: "8px 0px",
    color: "#fff",
    fontSize: "14px",
    fontFamily: "monospace",
    outline: "none",
  },
};

export class DevConsole extends SimulationLayer {
  name = "DevConsole";
  is_open = false;
  history = [];
  history_index = -1;
  console_panel = null;
  input = null;
  cursor = null;
  command_handlers = new Map();
  command_handler_list = [];
  suggestions_panel = null;
  current_suggestions = [];
  suggestion_index = -1;
  scene = null;
  use_own_cursor = true;

  init() {
    super.init();

    this._init_ui();
    this._bind_events();

    this.register_command("render_pass_organizer", new RenderPassOrganizer());
  }

  update(delta_time) {
    super.update(delta_time);

    if (this.is_open) {
      const input_provider = InputProvider.get();

      if (input_provider.get_action(InputKey.B_mouse_left)) {
        const mouse_x = input_provider.get_range(InputRange.M_xabs);
        const mouse_y = input_provider.get_range(InputRange.M_yabs);

        if (!this.console_panel.is_inside(mouse_x, mouse_y)) {
          this.hide();
        }

        input_provider.consume_action(InputKey.B_mouse_left);
      }
    }

    for (let i = 0; i < this.command_handler_list.length; i++) {
      this.command_handler_list[i].update(delta_time);
    }
  }

  set_scene(scene) {
    this.scene = scene;
    for (let i = 0; i < this.command_handler_list.length; i++) {
      this.command_handler_list[i].set_scene(scene);
    }
  }

  _init_ui() {
    this.console_panel = Panel.create(console_panel_name, console_panel_config);
    this.input = Input.create(input_name, input_config);

    // Suggestions panel
    this.suggestions_panel = Panel.create("console_suggestions", {
      style: {
        width: "100%",
        backgroundColor: "rgba(0, 0, 0, 0.8)",
        marginTop: "4px",
        maxHeight: "200px",
        overflowY: "auto",
        display: "none",
      },
    });

    this.console_panel.add_child(this.input);
    this.console_panel.add_child(this.suggestions_panel);

    const view_root = Element.get_view_root();
    view_root.add_child(this.console_panel);
  }

  _bind_events() {
    document.addEventListener(keydown_event_name, (e) => {
      if (e.key === equal_key && !e.repeat) {
        e.preventDefault();
        this.toggle();
      }

      if (!this.is_open) return;

      e.stopPropagation();

      if (e.key === escape_key) {
        this.hide();
      } else if (e.key === arrow_up_key) {
        if (this.suggestions_panel.dom.style.display === display_block_name) {
          this._navigate_suggestions(-1);
        } else {
          this._navigate_history(1);
        }
      } else if (e.key === arrow_down_key) {
        if (this.suggestions_panel.dom.style.display === display_block_name) {
          this._navigate_suggestions(1);
        } else {
          this._navigate_history(-1);
        }
      } else if (e.key === tab_key) {
        this._complete_suggestion();
      } else if (e.key === enter_key) {
        this._handle_command(this.input.dom.value);
        this.hide();
      } else {
        // Update suggestions on any other key press
        this._update_suggestions(this.input.dom.value);
      }
    });
  }

  _update_suggestions(input) {
    if (!input) {
      this.suggestions_panel.dom.style.display = display_none_name;
      return;
    }

    const commands = Array.from(this.command_handlers.keys());
    this.current_suggestions = commands.filter((cmd) =>
      cmd.toLowerCase().startsWith(input.toLowerCase())
    );

    if (this.current_suggestions.length === 0) {
      this.suggestions_panel.dom.style.display = display_none_name;
      return;
    }

    this.suggestion_index = -1;
    this._render_suggestions();
  }

  _render_suggestions() {
    this.suggestions_panel.clear_children();

    this.current_suggestions.forEach((suggestion, index) => {
      const suggestion_panel = Panel.create(`suggestion_${index}`, {
        style: {
          padding: "4px 8px",
          cursor: "pointer",
          backgroundColor:
            index === this.suggestion_index ? "rgba(255, 255, 255, 0.1)" : "transparent",
        },
      });

      const label = Label.create(`suggestion_label_${index}`, {
        text: suggestion,
        style: {
          color: "#fff",
          fontFamily: "monospace",
        },
      });

      suggestion_panel.add_child(label);
      suggestion_panel.dom.addEventListener(click_event_name, () => {
        this.input.dom.value = suggestion;
        this.input.dom.focus();
        this._update_suggestions(suggestion);
      });

      this.suggestions_panel.add_child(suggestion_panel);
    });

    this.suggestions_panel.dom.style.display = display_block_name;
  }

  _navigate_suggestions(direction) {
    if (this.current_suggestions.length === 0) return;

    this.suggestion_index = Math.max(
      -1,
      Math.min(this.current_suggestions.length - 1, this.suggestion_index + direction)
    );

    this._render_suggestions();

    if (this.suggestion_index !== -1) {
      this.input.dom.value = this.current_suggestions[this.suggestion_index];
    }
  }

  _complete_suggestion() {
    if (this.current_suggestions.length === 0) return;

    if (this.suggestion_index === -1) {
      this.input.dom.value = this.current_suggestions[0];
    } else {
      this.input.dom.value = this.current_suggestions[this.suggestion_index];
    }

    this._update_suggestions(this.input.dom.value);
  }

  _navigate_history(direction) {
    if (this.history.length === 0) return;

    this.history_index = Math.max(
      -1,
      Math.min(this.history.length - 1, this.history_index + direction)
    );

    if (this.history_index === -1) {
      this.input.dom.value = "";
    } else {
      this.input.dom.value = this.history[this.history_index];
    }
  }

  _handle_command(command) {
    if (!command) return;

    // Add to history if it's a new command
    if (this.history[0] !== command) {
      this.history.unshift(command);
      if (this.history.length > 50) {
        this.history.pop();
      }
    }
    this.history_index = -1;

    // Parse command and arguments
    const parts = command.split(" ");
    const cmd = parts[0].toLowerCase();
    const args = parts.slice(1);

    // Execute handler if registered
    const handler_index = this.command_handlers.get(cmd);
    if (handler_index !== undefined) {
      this.command_handler_list[handler_index].execute(args);
    } else {
      console.warn(`Unknown command: ${cmd}`);
    }
  }

  /**
   * Register a new command handler
   * @param {string} command - The command name
   * @param {Function} handler - The function to handle the command
   */
  register_command(command, command_handler) {
    if (command_handler) {
      command_handler.init();
      this.command_handlers.set(command.toLowerCase(), this.command_handler_list.length);
      this.command_handler_list.push(command_handler);
    }
    return command_handler;
  }

  /**
   * Get a command handler by command name
   * @param {string} command - The command name
   * @returns {Function} The command handler
   */
  get_command_handler(command) {
    return this.command_handler_list[this.command_handlers.get(command.toLowerCase())];
  }

  show() {
    this.is_open = true;
    this.console_panel.dom.style.visibility = visible_name;
    this.input.dom.value = "";
    this.input.dom.focus();
    this.suggestions_panel.dom.style.display = display_none_name;

    for (let i = 0; i < this.command_handler_list.length; i++) {
      this.command_handler_list[i].hide();
    }
  }

  hide() {
    this.is_open = false;
    this.console_panel.dom.style.visibility = hidden_name;
    this.suggestions_panel.dom.style.display = display_none_name;
    this.input.dom.blur();
  }

  toggle() {
    if (this.is_open) {
      this.hide();
    } else {
      this.show();
    }
  }
}
