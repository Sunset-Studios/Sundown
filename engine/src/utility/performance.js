const performance_mark_names = new Map();

export function profile_scope(name, fn) {
  if (__DEV__) {
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
  if (__DEV__) {
    const mark_names = performance_mark_names.get(name);
    performance.mark(mark_names.end);
    performance.measure(mark_names.measure, mark_names.start, mark_names.end);
  }
}
