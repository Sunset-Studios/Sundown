import { SimulationLayer } from "../core/simulation_layer.js";
import { InputProvider } from "../input/input_provider.js";
import { InputKey } from "../input/input_types.js";
import { panel, input, label, UIContext } from "../ui/2d/immediate.js";
import { RenderPassOrganizer } from "./render_pass_organizer.js";
import { MLStats } from "./ml_stats.js";
import { CameraInfo } from "./camera_info.js";
import { AABBDebug } from "./aabb_debug.js";
import { PerformanceTrace } from "./performance_trace.js";
import { DebugDrawPicker } from "./debug_draw_picker.js";
import { log, warn, error } from "../utility/logging.js";

// Constants for naming and key codes
const input_name = "console_input";
const bottom = "bottom";

// These configurations roughly mirror the previous style settings
const console_panel_config = {
  layout: "column",
  gap: 4,
  y: "20%",
  anchor_y: bottom,
  width: 600,
  background_color: "rgba(0, 0, 0, 0.8)",
  padding: 8,
  corner_radius: 4,
  border: "1px solid #444",
};

const input_config = {
  // Renders a textfield spanning the panel width
  width: "100%",
  height: 30,
  background_color: "rgba(0, 0, 0, 0.5)",
  border: "none",
  padding_left: 8,
  padding_right: 8,
  text_color: "#fff",
  font: "14px monospace",
  cursor_color: "#fff",
};

const console_suggestions_config = {
  layout: "column",
  gap: 4,
  width: "100%",
  background_color: "rgba(0, 0, 0, 0.5)",
  padding: 4,
};

//
// DevConsole class using the new immediate–mode UI framework.
//
export class DevConsole extends SimulationLayer {
  name = "DevConsole";
  is_open = false;
  history = [];
  history_index = -1;
  command_handlers = new Map();
  command_handler_list = [];
  current_suggestions = [];
  suggestion_index = -1;
  scene = null;

  init() {
    super.init();
    // Register stat command handlers.
    this.register_command("render_pass_organizer", new RenderPassOrganizer());
    this.register_command("ml_stats", new MLStats());
    this.register_command("camera_info", new CameraInfo());
    this.register_command("aabb_debug", new AABBDebug());
    this.register_command("performance_trace", new PerformanceTrace());

    // Register debug rendering command handlers.
    this.register_command("debug_draw", new DebugDrawPicker());
  }

  update(delta_time) {
    super.update(delta_time);

    // Process keyboard input from UIContext each frame.
    this._update_input();

    this.render();

    // Update any registered command handlers (if they implement an update method).
    for (let i = 0; i < this.command_handler_list.length; i++) {
      if (typeof this.command_handler_list[i].update === "function") {
        this.command_handler_list[i].update(delta_time);
      }
    }
  }

  /**
   * Handles keyboard events from UIContext instead of document events.
   * This method should be called once per frame (e.g., in update()).
   */
  _update_input() {
    const keys = UIContext.keyboard_events;

    // Check for the toggle key ("=") to open/close the console.
    for (let i = 0; i < keys.length; i++) {
      const key = keys.get(i);
      if (key.key === InputKey.K_Equals && !key.held) {
        this.toggle();
        key.consumed = true;
        // Break so that we don't process further events from this frame that might toggle again.
        break;
      }
    }

    // Don't process further events if the console isn't open.
    if (!this.is_open) return;

    // Retrieve the input field state from the global UIContext.
    const input_state = UIContext.input_field_state[input_name] || { value: "" };

    // Process each key event.
    for (let i = 0; i < keys.length; i++) {
      const key = keys.get(i);

      let consume = false;

      // Skip the equal key since it is already handled.
      if (key.key === InputKey.K_Equals) continue;

      if (key.key === InputKey.K_Escape) {
        this.hide();
        consume = true;
      } else if (key.key === InputKey.K_Up) {
        if (this.current_suggestions.length > 0) {
          this._navigate_suggestions(-1, input_state);
          consume = true;
        } else {
          this._navigate_history(1, input_state);
          consume = true;
        }
      } else if (key.key === InputKey.K_Down) {
        if (this.current_suggestions.length > 0) {
          this._navigate_suggestions(1, input_state);
          consume = true;
        } else {
          this._navigate_history(-1, input_state);
          consume = true;
        }
      } else if (key.key === InputKey.K_Tab) {
        // Complete the current suggestion.
        this._complete_suggestion(input_state);
        consume = true;
      } else if (key.key === InputKey.K_Return) {
        // Execute the command.
        this._handle_command(input_state.value);
        this.hide();
        consume = true;
      }

      if (consume) {
        key.consumed = true;
      }

      InputProvider.consume_action(key.key);
      InputProvider.consume_state(key.key);
    }
  }

  /**
   * Render the developer console using immediate mode UI calls.
   * This method should be called during the main render loop.
   */
  render() {
    if (!this.is_open) return;

    // Render the console panel.
    const panel_state = panel(console_panel_config, () => {
      // Render the input field.
      const input_state = input(input_name, input_config);
      // Force focus.
      if (!input_state.is_focused) {
        input_state.is_focused = true;
      }

      const current_text = input_state.value.trim();
      if (current_text === "") {
        this.current_suggestions = [];
        this.suggestion_index = -1;
      } else {
        const commands = Array.from(this.command_handlers.keys());
        this.current_suggestions = commands.filter((cmd) =>
          cmd.toLowerCase().startsWith(current_text.toLowerCase())
        );
        if (this.suggestion_index >= this.current_suggestions.length) {
          this.suggestion_index = -1;
        }
      }

      if (this.current_suggestions.length > 0) {
        panel(console_suggestions_config, () => {
          for (let index = 0; index < this.current_suggestions.length; index++) {
            const suggestion = this.current_suggestions[index];
            const btn = label(suggestion, {
              width: "100%",
              text_valign: "middle",
              text_padding: 5,
              height: 30,
              padding_left: 4,
              padding_right: 4,
              padding_top: 8,
              padding_bottom: 8,
              background_color:
                index === this.suggestion_index ? "rgba(255, 255, 255, 0.1)" : "rgba(0, 0, 0, 1.0)",
              text_color: "#ccc",
              font: "14px monospace",
            });
            if (btn.clicked) {
              input_state.value = suggestion;
              input_state.is_focused = true;
              this.suggestion_index = index;
            }
          }
        });
      }
    });

    if (this.is_open && InputProvider.get_action(InputKey.B_mouse_left)) {
      if (!panel_state.hovered) {
        this.hide();
      }
      InputProvider.consume_action(InputKey.B_mouse_left);
    }
  }

  set_scene(scene) {
    this.scene = scene;
    for (let i = 0; i < this.command_handler_list.length; i++) {
      if (typeof this.command_handler_list[i].set_scene === "function") {
        this.command_handler_list[i].set_scene(scene);
      }
    }
  }

  _navigate_suggestions(direction, input_state) {
    if (this.current_suggestions.length === 0) return;

    this.suggestion_index = Math.max(
      -1,
      Math.min(this.current_suggestions.length - 1, this.suggestion_index + direction)
    );

    if (this.suggestion_index !== -1) {
      input_state.value = this.current_suggestions[this.suggestion_index];
    }
  }

  _complete_suggestion(input_state) {
    if (this.current_suggestions.length === 0) return;

    if (this.suggestion_index === -1) {
      input_state.value = this.current_suggestions[0];
      this.suggestion_index = 0;
    } else {
      input_state.value = this.current_suggestions[this.suggestion_index];
    }
  }

  _navigate_history(direction, input_state) {
    if (this.history.length === 0) return;
    this.history_index = Math.max(
      -1,
      Math.min(this.history.length - 1, this.history_index + direction)
    );
    if (this.history_index === -1) {
      input_state.value = "";
    } else {
      input_state.value = this.history[this.history_index];
    }
  }

  _handle_command(command) {
    if (!command) return;

    // Add the command to history if it is different from the last.
    if (this.history[0] !== command) {
      this.history.unshift(command);
      if (this.history.length > 50) {
        this.history.pop();
      }
    }
    this.history_index = -1;

    // Parse command and its arguments.
    const parts = command.split(" ");
    const cmd = parts[0].toLowerCase();
    const args = parts.slice(1);

    const handler_index = this.command_handlers.get(cmd);
    if (handler_index !== undefined) {
      this.command_handler_list[handler_index].execute(args);
    } else {
      warn(`Unknown command: ${cmd}`);
    }
  }

  /**
   * Register a new command handler.
   * @param {string} command - The command name.
   * @param {Function} command_handler - The command handler.
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
   * Get a command handler by its command name.
   * @param {string} command - The command name.
   * @returns {Function} The command handler.
   */
  get_command_handler(command) {
    return this.command_handler_list[this.command_handlers.get(command.toLowerCase())];
  }

  show() {
    this.is_open = true;
    const input_state = UIContext.input_field_state[input_name];
    if (input_state) {
      input_state.value = "";
      input_state.is_focused = true;
    }
    this.current_suggestions = [];
    this.suggestion_index = -1;

    for (let i = 0; i < this.command_handler_list.length; i++) {
      if (typeof this.command_handler_list[i].hide === "function") {
        this.command_handler_list[i].hide();
      }
    }
  }

  hide() {
    this.is_open = false;
    const input_state = UIContext.input_field_state[input_name];
    if (input_state) {
      input_state.is_focused = false;
    }
    this.current_suggestions = [];
  }

  toggle() {
    if (this.is_open) {
      this.hide();
    } else {
      this.show();
    }
  }
}
