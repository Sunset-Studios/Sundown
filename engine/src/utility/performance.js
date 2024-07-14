export function profile_scope(name, fn) {
    performance.mark(`${name}_start`);
    fn();
    performance.mark(`${name}_end`);
    performance.measure(`${name}`, `${name}_start`, `${name}_end`);
}