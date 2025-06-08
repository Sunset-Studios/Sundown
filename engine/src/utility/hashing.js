export function hash_data_map(data_map, initial_key) {
  // Derive a numeric seed from initial_key (supports both numbers and strings)
  let hash = hash_value(initial_key);
  const sorted_keys = Object.keys(data_map).sort();
  for (let i = 0; i < sorted_keys.length; i++) {
    const key = sorted_keys[i];
    const value = data_map[key];
    hash ^= hash_value(key);
    hash = Math.imul(hash, 16777619);
    hash ^= hash_value(value);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

export function hash_data_object(data_object, initial_key) {
  // Derive a numeric seed from initial_key (supports both numbers and strings)
  let hash = hash_value(initial_key);
  const sorted_keys = Object.keys(data_object).sort();
  for (let i = 0; i < sorted_keys.length; i++) {
    const key = sorted_keys[i];
    const value = data_object[key];
    hash ^= hash_value(key);
    hash = Math.imul(hash, 16777619);
    hash ^= hash_value(value);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

export function hash_value(value) {
  if (typeof value === "number") {
    return value;
  } else if (typeof value === "string") {
    return value.split("").reduce((acc, char) => {
      return (acc << 5) - acc + char.charCodeAt(0);
    }, 0);
  } else if (typeof value === "object" && value.config && value.config.name) {
    return value.config.name.split("").reduce((acc, char) => {
      return (acc << 5) - acc + char.charCodeAt(0);
    }, 0);
  } else {
    return 0;
  }
}
