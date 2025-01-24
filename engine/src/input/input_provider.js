import { InputContext } from './input_context.js';
import { InputState, InputRange, InputType, InputKey } from './input_types.js';
import { InputProcessor } from './input_processor.js';
import { SimulationLayer } from '../core/simulation_layer.js';
import { Renderer } from '../renderer/renderer.js';

export class InputProvider extends SimulationLayer {
    static default_context_instance = null;

    contexts = [];
    current_dirty_states = [];
    b_initialized = false;
    processor = null;

    constructor() {
        if (InputProvider.instance) {
            return InputProvider.instance;
        }
        super();
        InputProvider.instance = this;
    }

    static get() {
        if (!InputProvider.instance) {
            return new InputProvider();
        }
        return InputProvider.instance;
    }

    static default_context() {
        if (!InputProvider.default_context_instance) {
            InputProvider.default_context_instance = new InputContext();
            if (InputProvider.default_context_instance.num_states() === 0) {
                const input_states = [];
                for (let i = 0; i < InputKey.NumKeys; ++i) {
                    input_states.push(new InputState("", i, InputRange.NumRanges, InputType.State));
                    input_states.push(new InputState("", i, InputRange.NumRanges, InputType.Action));
                }
                for (let i = 0; i < InputRange.NumRanges; ++i) {
                    input_states.push(new InputState("", InputKey.NumKeys, i, InputType.Range));
                }
                InputProvider.default_context_instance.set_states(input_states);
            }
        }
        return InputProvider.default_context_instance;
    }

    init() {
        if (!this.b_initialized) {
            this.b_initialized = true;
            this.processor = new InputProcessor();
            this.processor.init();
        }
    }

    pre_update(delta_time) {
        super.pre_update(delta_time);

        performance.mark('input provider update');

        if (this.contexts.length > 0) {
            const context = this.contexts[this.contexts.length - 1];

            this.processor.update(context, delta_time, Renderer.get().canvas);

            this.current_dirty_states = [];
            context.visit_dirty_states((state) => {
                this.current_dirty_states.push(state);
            });
        }
    }

    push_context(context) {
        this.contexts.push(context);
    }

    pop_context() {
        if (this.contexts.length > 0) {
            return this.contexts.pop();
        }
        return null;
    }

    get_state(name) {
        return this.current_dirty_states.some(state => 
            (typeof name === 'number' ? state.raw_input === name : state.mapped_name === name) && 
            state.input_type === InputType.State
        );
    }

    consume_state(name) {
        let index = this.current_dirty_states.findIndex(state => 
            (typeof name === 'number' ? state.raw_input === name : state.mapped_name === name) && 
            state.input_type === InputType.State
        );
        while (index !== -1) {
            this.current_dirty_states.splice(index, 1);
            index = this.current_dirty_states.findIndex(state => 
                (typeof name === 'number' ? state.raw_input === name : state.mapped_name === name) && 
                state.input_type === InputType.State
            );
        }
    }

    get_action(name) {
        return this.current_dirty_states.some(state => 
            (typeof name === 'number' ? state.raw_input === name : state.mapped_name === name) && 
            state.input_type === InputType.Action
        );
    }

    consume_action(name) {
        let index = this.current_dirty_states.findIndex(state => 
            (typeof name === 'number' ? state.raw_input === name : state.mapped_name === name) && 
            state.input_type === InputType.Action
        );
        while (index !== -1) {
            this.current_dirty_states.splice(index, 1);
            index = this.current_dirty_states.findIndex(state => 
                (typeof name === 'number' ? state.raw_input === name : state.mapped_name === name) && 
                state.input_type === InputType.Action
            );
        }
    }

    get_range(name) {
        const index = this.current_dirty_states.findIndex(state => 
            (typeof name === 'number' ? state.raw_range === name : state.mapped_name === name) && 
            state.input_type === InputType.Range
        );
        return index !== -1 ? this.current_dirty_states[index].range_value : 0.0;

    }

    consume_range(name) {
        let index = this.current_dirty_states.findIndex(state => 
            (typeof name === 'number' ? state.raw_range === name : state.mapped_name === name) && 
            state.input_type === InputType.Range
        );
        while (index !== -1) {
            this.current_dirty_states.splice(index, 1);
            index = this.current_dirty_states.findIndex(state => 
                (typeof name === 'number' ? state.raw_range === name : state.mapped_name === name) && 
                state.input_type === InputType.Range
            );
        }
    }
}