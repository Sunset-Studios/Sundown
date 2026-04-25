const noop = () => {};

const VERBOSITY_LEVELS = {
  SILENT: 0,
  ERROR: 1,
  WARN: 2,
  INFO: 3,
  DEBUG: 4,
};

const VERBOSITY = VERBOSITY_LEVELS.INFO;

export const log = __DEV__ && VERBOSITY >= VERBOSITY_LEVELS.INFO ? (...args) => console.log(...args) : noop;
export const warn = __DEV__ && VERBOSITY >= VERBOSITY_LEVELS.WARN ? (...args) => console.warn(...args) : noop;
export const error = __DEV__ && VERBOSITY >= VERBOSITY_LEVELS.ERROR ? (...args) => console.error(...args) : noop;

export function format_number(value) {
  return Number(value || 0).toLocaleString();
}

export function format_vec3(v) {
  if (!v) {
    return "0.00, 0.00, 0.00";
  }
  return `${Number(v[0]).toFixed(2)}, ${Number(v[1]).toFixed(2)}, ${Number(v[2]).toFixed(2)}`;
}
