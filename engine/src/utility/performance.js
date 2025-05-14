const performance_mark_names = new Map();
let trace_activated = false;

/**
 * Sets the trace activation state.
 * @param {boolean} value - The new activation state.
 */
export function set_trace_activated(value) {
  trace_activated = value;
}

/**
 * Returns the current trace activation state.
 * @returns {boolean} The current activation state.
 */
export function is_trace_activated() {
  return trace_activated;
}

/**
 * Profiles a scope of code execution. Simply falls through to the callback if tracing is not activated.
 * @param {string} name - The name of the scope.
 * @param {function} fn - The function to profile.
 */
export function profile_scope(name, fn) {
  if (__DEV__ && trace_activated) {
    if (!performance_mark_names.has(name)) {
      performance_mark_names.set(name, {
        start: `${name}_start`,
        end: `${name}_end`,
        measure: `${name}`,
      });
    }
    const mark_names = performance_mark_names.get(name);
    performance.mark(mark_names.start);
  }
  fn();
  if (__DEV__ && trace_activated) {
    const mark_names = performance_mark_names.get(name);
    performance.mark(mark_names.end);
    performance.measure(mark_names.measure, mark_names.start, mark_names.end);
  }
}
