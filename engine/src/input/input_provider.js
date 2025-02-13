import { InputContext } from "./input_context.js";
import { InputState, InputRange, InputType, InputKey } from "./input_types.js";
import { InputProcessor } from "./input_processor.js";
import { Renderer } from "../renderer/renderer.js";
import { profile_scope } from "../utility/performance.js";

const input_provider_update_scope_name = "InputProvider.update";

export class InputProvider {
  static default_context_instance = null;
  static contexts = [];
  static current_dirty_states = [];
  static processor = null;

  static default_context() {
    if (!this.default_context_instance) {
      this.default_context_instance = new InputContext();
      if (this.default_context_instance.num_states() === 0) {
        const input_states = [];
        for (let i = 0; i < InputKey.NumKeys; ++i) {
          input_states.push(new InputState("", i, InputRange.NumRanges, InputType.State));
          input_states.push(new InputState("", i, InputRange.NumRanges, InputType.Action));
        }
        for (let i = 0; i < InputRange.NumRanges; ++i) {
          input_states.push(new InputState("", InputKey.NumKeys, i, InputType.Range));
        }
        this.default_context_instance.set_states(input_states);
      }
    }
    return this.default_context_instance;
  }

  static setup() {
    this.processor = new InputProcessor();
    this.processor.init();
    this.push_context(this.default_context());
  }

  static update(delta_time) {
    profile_scope(input_provider_update_scope_name, () => {
      if (this.contexts.length > 0) {
        const context = this.contexts[this.contexts.length - 1];

        this.processor.update(context, delta_time, Renderer.get().canvas);

        this.current_dirty_states = [];
        context.visit_dirty_states((state) => {
          this.current_dirty_states.push(state);
        });
      }
    });
  }

  static push_context(context) {
    this.contexts.push(context);
  }

  static pop_context() {
    if (this.contexts.length > 0) {
      return this.contexts.pop();
    }
    return null;
  }

  static get_state(name) {
    return this.current_dirty_states.some(
      (state) =>
        (typeof name === "number" ? state.raw_input === name : state.mapped_name === name) &&
        state.input_type === InputType.State
    );
  }

  static consume_state(name) {
    let index = this.current_dirty_states.findIndex(
      (state) =>
        (typeof name === "number" ? state.raw_input === name : state.mapped_name === name) &&
        state.input_type === InputType.State
    );
    while (index !== -1) {
      this.current_dirty_states.splice(index, 1);
      index = this.current_dirty_states.findIndex(
        (state) =>
          (typeof name === "number" ? state.raw_input === name : state.mapped_name === name) &&
          state.input_type === InputType.State
      );
    }
  }

  static get_action(name) {
    return this.current_dirty_states.some(
      (state) =>
        (typeof name === "number" ? state.raw_input === name : state.mapped_name === name) &&
        state.input_type === InputType.Action
    );
  }

  static consume_action(name) {
    let index = this.current_dirty_states.findIndex(
      (state) =>
        (typeof name === "number" ? state.raw_input === name : state.mapped_name === name) &&
        state.input_type === InputType.Action
    );
    while (index !== -1) {
      this.current_dirty_states.splice(index, 1);
      index = this.current_dirty_states.findIndex(
        (state) =>
          (typeof name === "number" ? state.raw_input === name : state.mapped_name === name) &&
          state.input_type === InputType.Action
      );
    }
  }

  static get_range(name) {
    const index = this.current_dirty_states.findIndex(
      (state) =>
        (typeof name === "number" ? state.raw_range === name : state.mapped_name === name) &&
        state.input_type === InputType.Range
    );
    return index !== -1 ? this.current_dirty_states[index].range_value : 0.0;
  }

  static consume_range(name) {
    let index = this.current_dirty_states.findIndex(
      (state) =>
        (typeof name === "number" ? state.raw_range === name : state.mapped_name === name) &&
        state.input_type === InputType.Range
    );
    while (index !== -1) {
      this.current_dirty_states.splice(index, 1);
      index = this.current_dirty_states.findIndex(
        (state) =>
          (typeof name === "number" ? state.raw_range === name : state.mapped_name === name) &&
          state.input_type === InputType.Range
      );
    }
  }
}
