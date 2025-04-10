import { Renderer } from "../../renderer/renderer.js";
import { InputProvider } from "../../input/input_provider.js";
import {
  InputRange,
  InputKey,
} from "../../input/input_types.js";
import { SimulationLayer } from "../simulation_layer.js";
import { UIContext, ImmediateUIUpdater } from "../../ui/2d/immediate.js";
import { screen_pos_to_world_pos } from "../../utility/camera.js";
import { SharedViewBuffer, SharedFrameInfoBuffer } from "../shared_data.js";
import { profile_scope } from "../../utility/performance.js";

const ui_processor_pre_update_scope_name = "UIProcessor.pre_update";

export class UIProcessor extends SimulationLayer {
  context = null;
  current_delta_time = 0;
  processed_keys = new Set();

  init() {
    super.init();

    const renderer = Renderer.get();
    this.context = renderer.context_ui;

    this._pre_update_internal = this._pre_update_internal.bind(this);
  }

  pre_update(delta_time) {
    super.pre_update(delta_time);
    this.current_delta_time = delta_time;
    profile_scope(ui_processor_pre_update_scope_name, this._pre_update_internal);
  }

  _pre_update_internal() {
    const renderer = Renderer.get();

    if (!UIContext.input_state.pressed) {
      UIContext.drag_state.active = false;
      UIContext.drag_state.started = false;
      UIContext.drag_state.widget_id = null;
      UIContext.drag_state.timer = 0;
    }

    UIContext.input_state.prev_x = UIContext.input_state.x;
    UIContext.input_state.prev_y = UIContext.input_state.y;
    UIContext.input_state.x = InputProvider.get_range(InputRange.M_xabs);
    UIContext.input_state.y = InputProvider.get_range(InputRange.M_yabs);
    UIContext.input_state.clicked = InputProvider.get_action(InputKey.B_mouse_left);
    UIContext.input_state.pressed = InputProvider.get_state(InputKey.B_mouse_left);
    UIContext.input_state.wheel = InputProvider.get_range(InputRange.M_wheel) * 0.01;
    UIContext.delta_time = this.current_delta_time;

    if (UIContext.drag_state.active) {
      UIContext.drag_state.timer += this.current_delta_time;
    }

    this.processed_keys.clear();

    const keyboard_events = InputProvider.current_dirty_states;
    
    for (let i = 0; i < keyboard_events.length; i++) {
      const key = keyboard_events[i].raw_input;
      
      if (!this.processed_keys.has(key)) {
        const has_state = InputProvider.get_state(key);
        const has_action = InputProvider.get_action(key);
        
        const new_key = UIContext.keyboard_events.allocate();
        new_key.key = key;
        new_key.first = has_action;
        new_key.held = has_state && !has_action;
        new_key.consumed = false;
        new_key.last_change_time = keyboard_events[i].last_change_time;
        
        this.processed_keys.add(key);
      }
    }

    UIContext.input_state.world_position = screen_pos_to_world_pos(
      SharedViewBuffer.get_view_data(0),
      UIContext.input_state.x,
      UIContext.input_state.y,
      renderer.canvas.width,
      renderer.canvas.height,
      500.0 
    );
    SharedFrameInfoBuffer.set_cursor_world_position(UIContext.input_state.world_position);

    this.context.clearRect(0, 0, renderer.canvas.width, renderer.canvas.height);

    ImmediateUIUpdater.update_all(this.current_delta_time);
  }
}
