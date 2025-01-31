/**
 * Performs a deep clone of a JavaScript object/array/value
 * @param {*} value - The value to clone
 * @returns {*} - The cloned value
 */
export function deep_clone(value) {
    // Handle primitive types and functions
    if (typeof value !== 'object' || value === null) {
        return value;
    }

    // Handle Date objects
    if (value instanceof Date) {
        return new Date(value.getTime());
    }

    // Handle ArrayBuffer
    if (value instanceof ArrayBuffer) {
        const copy = new ArrayBuffer(value.byteLength);
        new Uint8Array(copy).set(new Uint8Array(value));
        return copy;
    }

    // Handle TypedArrays
    if (ArrayBuffer.isView(value)) {
        return value.slice();
    }

    // Handle RegExp objects
    if (value instanceof RegExp) {
        return new RegExp(value);
    }

    // Handle Maps
    if (value instanceof Map) {
        const result = new Map();
        value.forEach((val, key) => {
            result.set(key, deep_clone(val));
        });
        return result;
    }

    // Handle Sets
    if (value instanceof Set) {
        const result = new Set();
        value.forEach((val) => {
            result.add(deep_clone(val));
        });
        return result;

    }

    // Handle Arrays
    if (Array.isArray(value)) {
        return value.map(deep_clone);
    }

    // Handle plain objects
    const result = Object.create(Object.getPrototypeOf(value));
    for (const key in value) {
        if (value.hasOwnProperty(key)) {
            result[key] = deep_clone(value[key]);
        }
    }

    return result;
}
