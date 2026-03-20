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
  ranges_array = new Float32Array(InputRange.NumRanges);
  mouse_wheel = 0;
  mouse_x = 0;
  mouse_y = 0;
  abs_mouse_x = 0;
  abs_mouse_y = 0;
  has_pointer_position = false;
  active_touch_identifier = null;
  pointer_events_supported = false;
  touch_event_options = { passive: false };
  suppress_compatibility_mouse_events_until = 0;

  constructor() {
    this.pointer_events_supported = typeof window !== "undefined" && "PointerEvent" in window;

    this.handle_key_down = this.handle_key_down.bind(this);
    this.handle_key_up = this.handle_key_up.bind(this);
    this.handle_mouse_down = this.handle_mouse_down.bind(this);
    this.handle_mouse_up = this.handle_mouse_up.bind(this);
    this.handle_mouse_move = this.handle_mouse_move.bind(this);
    this.handle_mouse_wheel = this.handle_mouse_wheel.bind(this);
    this.handle_pointer_down = this.handle_pointer_down.bind(this);
    this.handle_pointer_up = this.handle_pointer_up.bind(this);
    this.handle_pointer_move = this.handle_pointer_move.bind(this);
    this.handle_pointer_cancel = this.handle_pointer_cancel.bind(this);
    this.handle_touch_start = this.handle_touch_start.bind(this);
    this.handle_touch_move = this.handle_touch_move.bind(this);
    this.handle_touch_end = this.handle_touch_end.bind(this);
    this.handle_touch_cancel = this.handle_touch_cancel.bind(this);
  }

  init() {
    window.addEventListener("keydown", this.handle_key_down);
    window.addEventListener("keyup", this.handle_key_up);

    if (this.pointer_events_supported) {
      window.addEventListener("pointerdown", this.handle_pointer_down);
      window.addEventListener("pointerup", this.handle_pointer_up);
      window.addEventListener("pointermove", this.handle_pointer_move);
      window.addEventListener("pointercancel", this.handle_pointer_cancel);
    } else {
      window.addEventListener("mousedown", this.handle_mouse_down);
      window.addEventListener("mouseup", this.handle_mouse_up);
      window.addEventListener("mousemove", this.handle_mouse_move);
      window.addEventListener("touchstart", this.handle_touch_start, this.touch_event_options);
      window.addEventListener("touchmove", this.handle_touch_move, this.touch_event_options);
      window.addEventListener("touchend", this.handle_touch_end, this.touch_event_options);
      window.addEventListener("touchcancel", this.handle_touch_cancel, this.touch_event_options);
    }

    window.addEventListener("wheel", this.handle_mouse_wheel);
  }

  shutdown() {
    window.removeEventListener("keydown", this.handle_key_down);
    window.removeEventListener("keyup", this.handle_key_up);

    if (this.pointer_events_supported) {
      window.removeEventListener("pointerdown", this.handle_pointer_down);
      window.removeEventListener("pointerup", this.handle_pointer_up);
      window.removeEventListener("pointermove", this.handle_pointer_move);
      window.removeEventListener("pointercancel", this.handle_pointer_cancel);
    } else {
      window.removeEventListener("mousedown", this.handle_mouse_down);
      window.removeEventListener("mouseup", this.handle_mouse_up);
      window.removeEventListener("mousemove", this.handle_mouse_move);
      window.removeEventListener("touchstart", this.handle_touch_start, this.touch_event_options);
      window.removeEventListener("touchmove", this.handle_touch_move, this.touch_event_options);
      window.removeEventListener("touchend", this.handle_touch_end, this.touch_event_options);
      window.removeEventListener("touchcancel", this.handle_touch_cancel, this.touch_event_options);
    }

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
    if (this.should_ignore_compatibility_mouse_event()) {
      return;
    }

    const key = this.browser_button_to_input_key(event.button);
    if (key !== undefined) {
      this.key_map.set(key, true);
    }

    this.update_pointer_position(event.clientX, event.clientY, 0, 0);
  }

  handle_mouse_up(event) {
    if (this.should_ignore_compatibility_mouse_event()) {
      return;
    }

    const key = this.browser_button_to_input_key(event.button);
    if (key !== undefined) {
      this.key_map.set(key, false);
    }

    this.update_pointer_position(event.clientX, event.clientY, 0, 0);
  }

  handle_mouse_move(event) {
    if (this.should_ignore_compatibility_mouse_event()) {
      return;
    }

    this.update_pointer_position(
      event.clientX,
      event.clientY,
      event.movementX,
      event.movementY
    );
  }

  handle_mouse_wheel(event) {
    this.mouse_wheel = event.deltaY;
  }

  handle_pointer_down(event) {
    this.update_pointer_position(event.clientX, event.clientY, 0, 0);

    if (event.pointerType !== "mouse" && !event.isPrimary) {
      return;
    }

    const key = this.browser_button_to_input_key(event.button);
    if (key !== undefined) {
      this.key_map.set(key, true);
    }
  }

  handle_pointer_up(event) {
    this.update_pointer_position(event.clientX, event.clientY, 0, 0);

    if (event.pointerType !== "mouse" && !event.isPrimary) {
      return;
    }

    const key = this.browser_button_to_input_key(event.button);
    if (key !== undefined) {
      this.key_map.set(key, false);
    }
  }

  handle_pointer_move(event) {
    if (event.pointerType !== "mouse" && !event.isPrimary) {
      return;
    }

    const use_native_movement = event.pointerType === "mouse";
    this.update_pointer_position(
      event.clientX,
      event.clientY,
      use_native_movement ? event.movementX : null,
      use_native_movement ? event.movementY : null
    );
  }

  handle_pointer_cancel(event) {
    if (event.pointerType !== "mouse" && !event.isPrimary) {
      return;
    }

    this.key_map.set(InputKey.B_mouse_left, false);
    this.update_pointer_position(event.clientX, event.clientY, 0, 0);
  }

  handle_touch_start(event) {
    const touch = this.claim_active_touch(event.changedTouches);
    if (!touch) {
      return;
    }

    this.mark_touch_activity();
    this.prevent_default_if_possible(event);
    this.key_map.set(InputKey.B_mouse_left, true);
    this.update_pointer_position(touch.clientX, touch.clientY, 0, 0);
  }

  handle_touch_move(event) {
    const touch = this.find_touch_by_identifier(event.touches, this.active_touch_identifier);
    if (!touch) {
      return;
    }

    this.mark_touch_activity();
    this.prevent_default_if_possible(event);
    this.update_pointer_position(touch.clientX, touch.clientY);
  }

  handle_touch_end(event) {
    const touch = this.find_touch_by_identifier(
      event.changedTouches,
      this.active_touch_identifier
    );
    if (!touch) {
      return;
    }

    this.mark_touch_activity();
    this.prevent_default_if_possible(event);
    this.key_map.set(InputKey.B_mouse_left, false);
    this.update_pointer_position(touch.clientX, touch.clientY, 0, 0);
    this.active_touch_identifier = null;
  }

  handle_touch_cancel(event) {
    const touch = this.find_touch_by_identifier(
      event.changedTouches,
      this.active_touch_identifier
    );
    if (!touch) {
      return;
    }

    this.mark_touch_activity();
    this.prevent_default_if_possible(event);
    this.key_map.set(InputKey.B_mouse_left, false);
    this.update_pointer_position(touch.clientX, touch.clientY, 0, 0);
    this.active_touch_identifier = null;
  }

  update_pointer_position(client_x, client_y, movement_x = null, movement_y = null) {
    if (movement_x == null || movement_y == null) {
      if (this.has_pointer_position) {
        this.mouse_x = client_x - this.abs_mouse_x;
        this.mouse_y = client_y - this.abs_mouse_y;
      } else {
        this.mouse_x = 0;
        this.mouse_y = 0;
      }
    } else {
      this.mouse_x = movement_x;
      this.mouse_y = movement_y;
    }

    this.abs_mouse_x = client_x;
    this.abs_mouse_y = client_y;
    this.has_pointer_position = true;
  }

  claim_active_touch(touch_list) {
    if (this.active_touch_identifier != null) {
      return this.find_touch_by_identifier(touch_list, this.active_touch_identifier);
    }

    const touch = touch_list?.[0];
    if (!touch) {
      return null;
    }

    this.active_touch_identifier = touch.identifier;
    return touch;
  }

  find_touch_by_identifier(touch_list, identifier) {
    if (identifier == null || !touch_list) {
      return null;
    }

    for (let i = 0; i < touch_list.length; ++i) {
      if (touch_list[i].identifier === identifier) {
        return touch_list[i];
      }
    }

    return null;
  }

  prevent_default_if_possible(event) {
    if (event.cancelable) {
      event.preventDefault();
    }
  }

  mark_touch_activity() {
    this.suppress_compatibility_mouse_events_until = performance.now() + 100;
  }

  should_ignore_compatibility_mouse_event() {
    return performance.now() < this.suppress_compatibility_mouse_events_until;
  }

  update(context, delta_time, canvas) {
    this.ranges_array[InputRange.M_wheel] = this.mouse_wheel;

    // Apply exponential decay to mouse wheel motion
    this.mouse_wheel *= Math.exp(-1000.0 * 0.5);

    if (this.is_pointer_locked(canvas)) {
      this.ranges_array[InputRange.M_xabs] = Math.max(
        0,
        Math.min(
          canvas.width,
          this.ranges_array[InputRange.M_xabs] + this.mouse_x
        )
      );
      this.ranges_array[InputRange.M_yabs] = Math.max(
        0,
        Math.min(
          canvas.height,
          this.ranges_array[InputRange.M_yabs] + this.mouse_y
        )
      );
    } else if (this.has_pointer_position) {
      this.ranges_array[InputRange.M_xabs] = this.client_to_canvas_x(this.abs_mouse_x, canvas);
      this.ranges_array[InputRange.M_yabs] = this.client_to_canvas_y(this.abs_mouse_y, canvas);
    } else {
      this.ranges_array[InputRange.M_xabs] = 0;
      this.ranges_array[InputRange.M_yabs] = 0;
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

  is_pointer_locked(canvas) {
    return typeof document !== "undefined" && document.pointerLockElement === canvas;
  }

  client_to_canvas_x(client_x, canvas) {
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0) {
      return 0;
    }

    const relative_x = ((client_x - rect.left) / rect.width) * canvas.width;
    return Math.max(0, Math.min(canvas.width, relative_x));
  }

  client_to_canvas_y(client_y, canvas) {
    const rect = canvas.getBoundingClientRect();
    if (rect.height <= 0) {
      return 0;
    }

    const relative_y = ((client_y - rect.top) / rect.height) * canvas.height;
    return Math.max(0, Math.min(canvas.height, relative_y));
  }
}
