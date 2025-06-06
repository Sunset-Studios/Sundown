import { InputKey, InputRange, InputType } from "./input_types.js";

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
    ShiftLeft: InputKey.K_LShift,
    ShiftRight: InputKey.K_RShift,
    ControlLeft: InputKey.K_LControl,
    ControlRight: InputKey.K_RControl,
    AltLeft: InputKey.K_LAlt,
    AltRight: InputKey.K_RAlt,
    ArrowUp: InputKey.K_Up,
    ArrowDown: InputKey.K_Down,
    ArrowLeft: InputKey.K_Left,
    ArrowRight: InputKey.K_Right,
  };

  static button_mapping = {
    0: InputKey.B_mouse_left,
    1: InputKey.B_mouse_middle,
    2: InputKey.B_mouse_right,
  };

  key_map = new Map();
  ranges_array = new Array(InputRange.NumRanges).fill(0.0);
  mouse_wheel = 0;
  mouse_x = 0;
  mouse_y = 0;
  abs_mouse_x = 0;
  abs_mouse_y = 0;

  constructor() {
    this.handle_key_down = this.handle_key_down.bind(this);
    this.handle_key_up = this.handle_key_up.bind(this);
    this.handle_mouse_down = this.handle_mouse_down.bind(this);
    this.handle_mouse_up = this.handle_mouse_up.bind(this);
    this.handle_mouse_move = this.handle_mouse_move.bind(this);
    this.handle_mouse_wheel = this.handle_mouse_wheel.bind(this);
  }

  init() {
    window.addEventListener("keydown", this.handle_key_down);
    window.addEventListener("keyup", this.handle_key_up);
    window.addEventListener("mousedown", this.handle_mouse_down);
    window.addEventListener("mouseup", this.handle_mouse_up);
    window.addEventListener("mousemove", this.handle_mouse_move);
    window.addEventListener("wheel", this.handle_mouse_wheel);
  }

  shutdown() {
    window.removeEventListener("keydown", this.handle_key_down);
    window.removeEventListener("keyup", this.handle_key_up);
    window.removeEventListener("mousedown", this.handle_mouse_down);
    window.removeEventListener("mouseup", this.handle_mouse_up);
    window.removeEventListener("mousemove", this.handle_mouse_move);
    window.removeEventListener("wheel", this.handle_mouse_wheel);
  }

  shouldPreventDefaultKeyCombo(event) {
    return (event.ctrlKey || event.metaKey) && (event.key === 'w' || event.key === 'a');
  }

  handle_key_down(event) {
    const key = this.browser_key_to_input_key(event.code);
    if (key !== undefined) {
      this.key_map.set(key, true);
    }
    if (this.shouldPreventDefaultKeyCombo(event)) {
      event.preventDefault();
    }
  }

  handle_key_up(event) {
    const key = this.browser_key_to_input_key(event.code);
    if (key !== undefined) {
      this.key_map.set(key, false);
    }
    if (this.shouldPreventDefaultKeyCombo(event)) {
      event.preventDefault();
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
    this.mouse_x = event.movementX;
    this.mouse_y = event.movementY;
    this.abs_mouse_x = event.clientX;
    this.abs_mouse_y = event.clientY;
  }

  handle_mouse_wheel(event) {
    this.mouse_wheel = event.deltaY;
  }

  update(context, delta_time, canvas) {
    this.ranges_array[InputRange.M_wheel] = this.mouse_wheel;

    // Apply exponential decay to mouse wheel motion
    this.mouse_wheel *= Math.exp(-0.5 * 0.5);

    if (this.ranges_array[InputRange.M_xabs] == null) {
      this.ranges_array[InputRange.M_xabs] = this.abs_mouse_x;
    } else {
      this.ranges_array[InputRange.M_xabs] = Math.max(
        0,
        Math.min(
          canvas.width,
          this.ranges_array[InputRange.M_xabs] + this.mouse_x
        )
      );
    }
    if (this.ranges_array[InputRange.M_yabs] == null) {
      this.ranges_array[InputRange.M_yabs] = this.abs_mouse_y;
    } else {
      this.ranges_array[InputRange.M_yabs] = Math.max(
        0,
        Math.min(
          canvas.height,
          this.ranges_array[InputRange.M_yabs] + this.mouse_y
        )
      );
    }

    // Calculate mouse movement
    this.ranges_array[InputRange.M_x] = this.mouse_x / canvas.width;
    this.ranges_array[InputRange.M_y] = this.mouse_y / canvas.height;

    // Apply decay to mouse movement
    this.mouse_x *= Math.exp(-0.5 * 0.5);
    this.mouse_y *= Math.exp(-0.5 * 0.5);

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
