import { InputKey, InputRange, InputType } from "@/input/input_types.js";

export class InputProcessor {
  static key_mapping = {
    Enter: InputKey.K_Return,
    Escape: InputKey.K_Escape,
    Backspace: InputKey.K_Backspace,
    Tab: InputKey.K_Tab,
    Space: InputKey.K_Space,
    Pause: InputKey.K_Pause,
    Quote: InputKey.K_Quote,
    Comma: InputKey.K_Comma,
    Minus: InputKey.K_Minus,
    Period: InputKey.K_Period,
    Slash: InputKey.K_Slash,
    Digit0: InputKey.K_0,
    Digit1: InputKey.K_1,
    Digit2: InputKey.K_2,
    Digit3: InputKey.K_3,
    Digit4: InputKey.K_4,
    Digit5: InputKey.K_5,
    Digit6: InputKey.K_6,
    Digit7: InputKey.K_7,
    Digit8: InputKey.K_8,
    Digit9: InputKey.K_9,
    Semicolon: InputKey.K_Semicolon,
    Equal: InputKey.K_Equals,
    BracketLeft: InputKey.K_LeftBracket,
    Backslash: InputKey.K_Backslash,
    BracketRight: InputKey.K_RightBracket,
    Backquote: InputKey.K_Backquote,
    KeyA: InputKey.K_a,
    KeyB: InputKey.K_b,
    KeyC: InputKey.K_c,
    KeyD: InputKey.K_d,
    KeyE: InputKey.K_e,
    KeyF: InputKey.K_f,
    KeyG: InputKey.K_g,
    KeyH: InputKey.K_h,
    KeyI: InputKey.K_i,
    KeyJ: InputKey.K_j,
    KeyK: InputKey.K_k,
    KeyL: InputKey.K_l,
    KeyM: InputKey.K_m,
    KeyN: InputKey.K_n,
    KeyO: InputKey.K_o,
    KeyP: InputKey.K_p,
    KeyQ: InputKey.K_q,
    KeyR: InputKey.K_r,
    KeyS: InputKey.K_s,
    KeyT: InputKey.K_t,
    KeyU: InputKey.K_u,
    KeyV: InputKey.K_v,
    KeyW: InputKey.K_w,
    KeyX: InputKey.K_x,
    KeyY: InputKey.K_y,
    KeyZ: InputKey.K_z,
  };

  static button_mapping = {
    0: InputKey.B_mouse_left,
    1: InputKey.B_mouse_middle,
    2: InputKey.B_mouse_right,
  };

  key_map = new Map();
  ranges_array = new Array(InputRange.NumRanges).fill(0.0);
  mouse_x = 0;
  mouse_y = 0;
  last_mouse_x = 0;
  last_mouse_y = 0;

  constructor() {
    this.handle_key_down = this.handle_key_down.bind(this);
    this.handle_key_up = this.handle_key_up.bind(this);
    this.handle_mouse_down = this.handle_mouse_down.bind(this);
    this.handle_mouse_up = this.handle_mouse_up.bind(this);
    this.handle_mouse_move = this.handle_mouse_move.bind(this);
  }

  init() {
    window.addEventListener("keydown", this.handle_key_down);
    window.addEventListener("keyup", this.handle_key_up);
    window.addEventListener("mousedown", this.handle_mouse_down);
    window.addEventListener("mouseup", this.handle_mouse_up);
    window.addEventListener("mousemove", this.handle_mouse_move);
  }

  shutdown() {
    window.removeEventListener("keydown", this.handle_key_down);
    window.removeEventListener("keyup", this.handle_key_up);
    window.removeEventListener("mousedown", this.handle_mouse_down);
    window.removeEventListener("mouseup", this.handle_mouse_up);
    window.removeEventListener("mousemove", this.handle_mouse_move);
  }

  handle_key_down(event) {
    const key = this.browser_key_to_input_key(event.code);
    if (key !== undefined) {
      this.key_map.set(key, true);
    }
  }

  handle_key_up(event) {
    const key = this.browser_key_to_input_key(event.code);
    if (key !== undefined) {
      this.key_map.set(key, false);
    }
  }

  handle_mouse_down(event) {
    const key = this.browser_button_to_input_key(event.button);
    if (key !== undefined) {
      this.key_map.set(key, true);
    }
  }

  handle_mouse_up(event) {
    const key = this.browser_button_to_input_key(event.button);
    if (key !== undefined) {
      this.key_map.set(key, false);
    }
  }

  handle_mouse_move(event) {
    this.mouse_x = event.clientX;
    this.mouse_y = event.clientY;
  }

  update(context, window) {
    // Calculate mouse movement
    const delta_x = this.mouse_x - this.last_mouse_x;
    const delta_y = this.mouse_y - this.last_mouse_y;

    this.ranges_array[InputRange.M_x] = delta_x / window.width;
    this.ranges_array[InputRange.M_y] = delta_y / window.height;

    // Apply decay to mouse movement
    this.ranges_array[InputRange.M_x] *= Math.exp(-0.5 * 0.5);
    this.ranges_array[InputRange.M_y] *= Math.exp(-0.5 * 0.5);

    this.last_mouse_x = this.mouse_x;
    this.last_mouse_y = this.mouse_y;

    // Update the input context
    for (let i = 0; i < context.input_states.length; ++i) {
      const state = context.input_states[i];
      switch (state.input_type) {
        case InputType.State:
        case InputType.Action: {
          const input_is_active = this.key_map.get(state.raw_input) || false;
          if (state.input_type === InputType.State) {
            context.set_state(i, input_is_active);
          } else {
            context.set_action(i, input_is_active);
          }
          break;
        }
        case InputType.Range: {
          const range_value = this.ranges_array[state.raw_range];
          context.set_range(i, range_value);
          break;
        }
      }
    }
  }

  browser_key_to_input_key(code) {
    return InputProcessor.key_mapping[code];
  }

  browser_button_to_input_key(button) {
    return InputProcessor.button_mapping[button];
  }
}
