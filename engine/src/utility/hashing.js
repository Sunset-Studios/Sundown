export function hash_data(data_map, initial_hash) {
  let hash = initial_hash;
  for (const [key, value] of data_map) {
    hash = (hash << 5) - hash + hash_value(key);
    hash = (hash << 5) - hash + hash_value(value);
    hash |= 0; // Convert to 32-bit integer
  }
  return hash;
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
