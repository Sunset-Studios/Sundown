import { InputContext } from '@/input/input_context.js';
import { InputState, InputRange, InputType, InputKey } from '@/input/input_types.js';
import { InputProcessor } from '@/input/input_processor.js';
import { SimulationLayer } from '@/core/simulation_layer.js';

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

    update(delta_time) {
        super.update(delta_time);

        performance.mark('input provider update');

        if (this.contexts.length > 0) {
            const context = this.contexts[this.contexts.length - 1];

            this.processor.update(context, delta_time);

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

    get_action(name) {
        return this.current_dirty_states.some(state => 
            (typeof name === 'number' ? state.raw_input === name : state.mapped_name === name) && 
            state.input_type === InputType.Action
        );
    }

    get_range(name) {
        const state = this.current_dirty_states.find(state => 
            (typeof name === 'number' ? state.raw_range === name : state.mapped_name === name) && 
            state.input_type === InputType.Range
        );
        return state ? state.range_value : 0.0;
    }
}