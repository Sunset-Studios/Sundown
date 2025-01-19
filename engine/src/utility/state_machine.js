/**
 * A flexible finite state machine implementation that supports state transitions,
 * callbacks, and guards.
 */
export class StateMachine {
    constructor(initial_state) {
        this.current_state = initial_state;
        this.states = new Map();
        this.transitions = new Map();
        this.on_transition_callbacks = new Set();
        this._current_state_config = null;
    }

    /**
     * Adds a new state to the state machine
     * @param {string} state_name - The name of the state
     * @param {Object} state_config - Configuration for the state
     * @param {Function} [state_config.on_enter] - Called when entering the state
     * @param {Function} [state_config.on_exit] - Called when exiting the state
     * @param {Function} [state_config.on_update] - Called while in this state
     */
    add_state(state_name, state_config = {}) {
        const config = {
            on_enter: state_config.on_enter || (() => {}),
            on_exit: state_config.on_exit || (() => {}),
            on_update: state_config.on_update || (() => {})
        };
        
        this.states.set(state_name, config);
        
        // Update cache if this is the current state
        if (state_name === this.current_state) {
            this._current_state_config = config;
        }
    }

    /**
     * Adds a transition between states
     * @param {string} from_state - Starting state
     * @param {string} to_state - Target state
     * @param {string} trigger - Name of the trigger that causes this transition
     * @param {Function} [guard] - Optional guard function that must return true to allow transition
     */
    add_transition(from_state, to_state, trigger, guard = null) {
        if (!this.transitions.has(from_state)) {
            this.transitions.set(from_state, new Map());
        }
        
        const state_transitions = this.transitions.get(from_state);
        state_transitions.set(trigger, {
            to_state,
            guard,
            to_state_config: this.states.get(to_state)
        });
    }

    /**
     * Pre-validates if a transition is possible
     * @param {string} trigger - The trigger to validate
     * @param {*} [data] - Optional data to test against guards
     * @returns {boolean} - Whether the transition would be valid
     */
    can_trigger(trigger, data = null) {
        const current_state_transitions = this.transitions.get(this.current_state);
        if (!current_state_transitions || !current_state_transitions.has(trigger)) {
            return false;
        }

        const transition = current_state_transitions.get(trigger);
        return !transition.guard || transition.guard(data);
    }

    /**
     * Triggers a state transition
     * @param {string} trigger - The name of the trigger
     * @param {*} [data] - Optional data to pass to callbacks
     * @returns {boolean} - Whether the transition was successful
     */
    trigger(trigger, data = null) {
        const current_state_transitions = this.transitions.get(this.current_state);
        if (!current_state_transitions || !current_state_transitions.has(trigger)) {
            return false;
        }


        const transition = current_state_transitions.get(trigger);
        
        // Check guard condition if it exists
        if (transition.guard && !transition.guard(data)) {
            return false;
        }

        const from_state = this.current_state;
        const to_state = transition.to_state;

        // Execute exit action of current state
        if (this._current_state_config) {
            this._current_state_config.on_exit(data);
        }

        // Update current state and cache
        this.current_state = to_state;
        this._current_state_config = transition.to_state_config;

        // Execute enter action of new state
        if (this._current_state_config) {
            this._current_state_config.on_enter(data);
        }

        // Notify transition listeners
        if (this.on_transition_callbacks.size > 0) {
            for (const callback of this.on_transition_callbacks) {
                callback(from_state, to_state, trigger, data);
            }
        }

        return true;
    }

    /**
     * Updates the current state
     * @param {*} [data] - Optional data to pass to the update callback
     */
    update(data = null) {
        // Use cached state config for faster updates
        if (this._current_state_config?.on_update) {
            this._current_state_config.on_update(data);
        }
    }

    /**
     * Gets the current state
     * @returns {string} - The current state name
     */
    get_current_state() {
        return this.current_state;
    }

    /**
     * Adds a callback that will be called on every state transition
     * @param {Function} callback - Function to call on state transition
     */
    on_transition(callback) {
        this.on_transition_callbacks.add(callback);
    }

    /**
     * Removes a transition callback
     * @param {Function} callback - The callback to remove
     */
    remove_transition_callback(callback) {
        this.on_transition_callbacks.delete(callback);
    }

    /**
     * Resets the state machine to its initial state
     * @param {string} state - The state to reset to
     */
    reset(state) {
        this.current_state = state;
        this._current_state_config = this.states.get(state);
    }
}