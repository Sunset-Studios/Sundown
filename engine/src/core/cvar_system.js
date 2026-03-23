import { global_dispatcher } from "./dispatcher.js";

export const CVarType = Object.freeze({
  Boolean: "boolean",
  Number: "number",
  Integer: "integer",
  String: "string",
  Enum: "enum",
});

const TRUE_VALUES = new Set(["1", "true", "on", "yes"]);
const FALSE_VALUES = new Set(["0", "false", "off", "no"]);

function normalize_boolean(value) {
  if (typeof value === "boolean") {
    return value;
  }

  if (typeof value === "number") {
    return value !== 0;
  }

  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    if (TRUE_VALUES.has(normalized)) {
      return true;
    }
    if (FALSE_VALUES.has(normalized)) {
      return false;
    }
  }

  throw new Error(`Unable to coerce '${value}' to a boolean.`);
}

function normalize_number(value, integer = false) {
  const parsed = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(parsed)) {
    throw new Error(`Unable to coerce '${value}' to a number.`);
  }

  return integer ? Math.trunc(parsed) : parsed;
}

function create_enum_maps(values) {
  const entries = Array.isArray(values) ? values.map((value) => [String(value), value]) : Object.entries(values);
  const label_to_value = new Map();
  const value_to_label = new Map();

  for (let i = 0; i < entries.length; i++) {
    const [label, value] = entries[i];
    const normalized_label = String(label).toLowerCase();
    label_to_value.set(normalized_label, value);
    if (!value_to_label.has(value)) {
      value_to_label.set(value, String(label));
    }
  }

  return { label_to_value, value_to_label };
}

export class CVarSystem {
  static definitions = new Map();
  static values = new Map();
  static listeners = new Map();

  static register(name, definition = {}) {
    if (!name) {
      throw new Error("CVar registration requires a name.");
    }

    if (this.definitions.has(name)) {
      return this.definitions.get(name);
    }

    const normalized_definition = this._normalize_definition(name, definition);
    this.definitions.set(name, normalized_definition);

    if (!this.values.has(name)) {
      this.values.set(name, normalized_definition.default_value);
    }

    return normalized_definition;
  }

  static register_many(definitions) {
    for (let i = 0; i < definitions.length; i++) {
      const { name, ...definition } = definitions[i];
      this.register(name, definition);
    }
  }

  static has(name) {
    return this.definitions.has(name);
  }

  static get_definition(name) {
    return this.definitions.get(name);
  }

  static get(name, fallback = undefined) {
    if (this.values.has(name)) {
      return this.values.get(name);
    }

    return fallback;
  }

  static set(name, value, options = {}) {
    const definition = this._require_definition(name);
    const next_value = this._coerce_value(value, definition);
    const previous_value = this.values.has(name) ? this.values.get(name) : definition.default_value;

    if (Object.is(previous_value, next_value)) {
      return next_value;
    }

    this.values.set(name, next_value);

    const payload = {
      definition,
      name,
      previous_value,
      value: next_value,
      source: options.source || "runtime",
    };

    const listeners = this.listeners.get(name);
    if (listeners) {
      const callbacks = Array.from(listeners);
      for (let i = 0; i < callbacks.length; i++) {
        callbacks[i](next_value, previous_value, payload);
      }
    }

    global_dispatcher.dispatch("cvar_changed", payload);
    global_dispatcher.dispatch(`cvar:${name}`, payload);

    return next_value;
  }

  static set_from_string(name, raw_value, options = {}) {
    return this.set(name, raw_value, options);
  }

  static toggle(name, options = {}) {
    const definition = this._require_definition(name);
    if (definition.type !== CVarType.Boolean) {
      throw new Error(`CVar '${name}' is not a boolean and cannot be toggled.`);
    }

    return this.set(name, !this.get(name, definition.default_value), {
      ...options,
      source: options.source || "toggle",
    });
  }

  static reset(name, options = {}) {
    const definition = this._require_definition(name);
    return this.set(name, definition.default_value, {
      ...options,
      source: options.source || "reset",
    });
  }

  static reset_all(options = {}) {
    const definitions = Array.from(this.definitions.values());
    for (let i = 0; i < definitions.length; i++) {
      const definition = definitions[i];
      if (options.silent) {
        this.values.set(definition.name, definition.default_value);
      } else {
        this.set(definition.name, definition.default_value, {
          ...options,
          source: options.source || "reset_all",
        });
      }
    }
  }

  static apply_config(config, options = {}) {
    const normalized_config = this._normalize_config(config, options.context);
    if (!normalized_config) {
      return;
    }

    if (Array.isArray(normalized_config.definitions) && normalized_config.definitions.length > 0) {
      this.register_many(normalized_config.definitions);
    }

    const cvars = normalized_config.cvars || normalized_config.values || {};
    const entries = Object.entries(cvars);
    for (let i = 0; i < entries.length; i++) {
      const [name, value] = entries[i];
      this.set(name, value, {
        ...options,
        source: options.source || "config",
      });
    }
  }

  static subscribe(name, callback, invoke_immediately = false) {
    if (!this.listeners.has(name)) {
      this.listeners.set(name, new Set());
    }

    const listeners = this.listeners.get(name);
    listeners.add(callback);

    if (invoke_immediately) {
      callback(this.get(name), undefined, {
        definition: this.get_definition(name),
        name,
        previous_value: undefined,
        source: "subscribe",
        value: this.get(name),
      });
    }

    return () => {
      const current_listeners = this.listeners.get(name);
      if (!current_listeners) {
        return;
      }

      current_listeners.delete(callback);
      if (current_listeners.size === 0) {
        this.listeners.delete(name);
      }
    };
  }

  static list(prefix = "") {
    const definitions = Array.from(this.definitions.values());
    definitions.sort((left, right) => left.name.localeCompare(right.name));

    return definitions
      .filter((definition) => !prefix || definition.name.startsWith(prefix))
      .map((definition) => ({
        definition,
        name: definition.name,
        value: this.get(definition.name, definition.default_value),
      }));
  }

  static format_value(name, value = undefined) {
    const definition = this._require_definition(name);
    const current_value = value !== undefined ? value : this.get(name, definition.default_value);

    if (typeof definition.format === "function") {
      return definition.format(current_value, definition);
    }

    if (definition.type === CVarType.Boolean) {
      return current_value ? "true" : "false";
    }

    if (definition.type === CVarType.Enum && definition.value_to_label.has(current_value)) {
      return definition.value_to_label.get(current_value);
    }

    return String(current_value);
  }

  static _require_definition(name) {
    const definition = this.definitions.get(name);
    if (!definition) {
      throw new Error(`Unknown cvar '${name}'.`);
    }
    return definition;
  }

  static _normalize_config(config, context = {}) {
    if (!config) {
      return null;
    }

    let normalized_config = config;
    if (typeof normalized_config === "function") {
      normalized_config = normalized_config(context);
    }

    if (!normalized_config) {
      return null;
    }

    if (normalized_config.default) {
      return this._normalize_config(normalized_config.default, context);
    }

    if (Array.isArray(normalized_config.definitions) || normalized_config.cvars || normalized_config.values) {
      return normalized_config;
    }

    return {
      cvars: normalized_config,
    };
  }

  static _normalize_definition(name, definition) {
    const normalized_definition = {
      description: "",
      ...definition,
      name,
    };

    if (!normalized_definition.type) {
      normalized_definition.type = this._infer_type(normalized_definition.default_value);
    }

    if (normalized_definition.type === CVarType.Enum) {
      if (!normalized_definition.values) {
        throw new Error(`Enum cvar '${name}' must declare allowed values.`);
      }

      const { label_to_value, value_to_label } = create_enum_maps(normalized_definition.values);
      normalized_definition.label_to_value = label_to_value;
      normalized_definition.value_to_label = value_to_label;

      if (normalized_definition.default_value === undefined) {
        normalized_definition.default_value = value_to_label.keys().next().value;
      }
    } else if (normalized_definition.default_value === undefined) {
      normalized_definition.default_value = this._default_value_for_type(normalized_definition.type);
    }

    normalized_definition.default_value = this._coerce_value(
      normalized_definition.default_value,
      normalized_definition
    );

    return normalized_definition;
  }

  static _infer_type(value) {
    switch (typeof value) {
      case "boolean":
        return CVarType.Boolean;
      case "number":
        return Number.isInteger(value) ? CVarType.Integer : CVarType.Number;
      default:
        return CVarType.String;
    }
  }

  static _default_value_for_type(type) {
    switch (type) {
      case CVarType.Boolean:
        return false;
      case CVarType.Number:
      case CVarType.Integer:
        return 0;
      case CVarType.String:
        return "";
      default:
        return undefined;
    }
  }

  static _coerce_value(value, definition) {
    const source_value =
      typeof definition.parse === "function" ? definition.parse(value, definition) : value;

    let coerced_value = source_value;

    switch (definition.type) {
      case CVarType.Boolean:
        coerced_value = normalize_boolean(source_value);
        break;
      case CVarType.Number:
        coerced_value = normalize_number(source_value);
        break;
      case CVarType.Integer:
        coerced_value = normalize_number(source_value, true);
        break;
      case CVarType.String:
        coerced_value = String(source_value);
        break;
      case CVarType.Enum:
        coerced_value = this._coerce_enum_value(source_value, definition);
        break;
      default:
        break;
    }

    if (typeof definition.validate === "function" && !definition.validate(coerced_value, definition)) {
      throw new Error(`Validation failed for cvar '${definition.name}'.`);
    }

    return coerced_value;
  }

  static _coerce_enum_value(value, definition) {
    if (definition.value_to_label.has(value)) {
      return value;
    }

    if (typeof value === "string") {
      const normalized_label = value.trim().toLowerCase();
      if (definition.label_to_value.has(normalized_label)) {
        return definition.label_to_value.get(normalized_label);
      }

      const numeric_value = Number(value);
      if (!Number.isNaN(numeric_value) && definition.value_to_label.has(numeric_value)) {
        return numeric_value;
      }
    }

    throw new Error(
      `Unable to coerce '${value}' to enum '${definition.name}'. Expected one of: ${Array.from(
        definition.label_to_value.keys()
      ).join(", ")}.`
    );
  }
}
