export class InputContext {
    input_states = [];
    action_mappings = [];
    state_mappings = [];
    range_mappings = [];
    raw_to_state_mappings = {};
    action_to_state_mappings = {};
    dirty_states = new Set();

    constructor(states = []) {
        this.set_states(states);
    }

    num_states() {
        return this.input_states.length;
    }

    set_states(states) {
        this.input_states = states;
        this.action_mappings = new Array(this.input_states.length).fill(false);
        this.state_mappings = new Array(this.input_states.length).fill(false);
        this.range_mappings = new Array(this.input_states.length).fill(0.0);
        this.raw_to_state_mappings = {};
        this.action_to_state_mappings = {};
        this.dirty_states = new Set();

        for (let i = 0; i < this.input_states.length; ++i) {
            const input_state = this.input_states[i];
            this.raw_to_state_mappings[input_state.raw_input] = i;
            this.action_to_state_mappings[input_state.mapped_name] = i;
        }
    }

    set_state(input_state_index, new_state) {
        console.assert(input_state_index < this.state_mappings.length && input_state_index >= 0, 'Invalid input state index');
        if (this.state_mappings[input_state_index] !== new_state) {
            this.input_states[input_state_index].last_change_time = performance.now();
        }
        this.state_mappings[input_state_index] = new_state;
        if (new_state) {
            this.dirty_states.add(input_state_index);
        }
    }

    set_action(input_state_index, new_action) {
        console.assert(input_state_index < this.action_mappings.length && input_state_index >= 0, 'Invalid input state index');
        if (this.action_mappings[input_state_index] !== new_action) {
            this.input_states[input_state_index].last_change_time = performance.now();
        }
        const action_fired = !this.action_mappings[input_state_index] && new_action;
        this.action_mappings[input_state_index] = new_action;
        if (action_fired) {
            this.dirty_states.add(input_state_index);
        }
    }

    set_range(input_state_index, new_range) {
        console.assert(input_state_index < this.range_mappings.length && input_state_index >= 0, 'Invalid input state index');
        if (this.range_mappings[input_state_index] !== new_range) {
            this.input_states[input_state_index].last_change_time = performance.now();
        }
        if (new_range !== 0.0) {
            this.range_mappings[input_state_index] = new_range;
            this.input_states[input_state_index].range_value = new_range;
            this.dirty_states.add(input_state_index);
        }
    }

    visit_dirty_states(visitor) {
        if (typeof visitor !== 'function') {
            return;
        }

        for (const index of this.dirty_states) {
            console.assert(index < this.input_states.length && index >= 0, 'Invalid dirty state index');
            const state = this.input_states[index];
            visitor(state);
        }

        this.dirty_states.clear();
    }
}
