function get_default_base_url() {
  if (typeof document !== "undefined" && document.baseURI) {
    return document.baseURI;
  }
  if (typeof window !== "undefined" && window.location?.href) {
    return window.location.href;
  }
  return null;
}

export function resolve_resource_url(resource_path, base_url = get_default_base_url()) {
  const normalized_path = String(resource_path ?? "");
  if (!base_url) {
    return normalized_path;
  }

  try {
    return new URL(normalized_path, base_url).href;
  } catch {
    return normalized_path;
  }
}

export function is_html_content_type(content_type) {
  return String(content_type ?? "")
    .toLowerCase()
    .includes("text/html");
}

export function serialize_json(value, options = {}) {
  const spacing = options.pretty === true ? 2 : options.spacing;
  return JSON.stringify(value, options.replacer, spacing);
}

export function deserialize_json(payload, label = "JSON payload") {
  let text = payload;
  if (payload instanceof ArrayBuffer) {
    text = new TextDecoder().decode(new Uint8Array(payload));
  } else if (ArrayBuffer.isView(payload)) {
    text = new TextDecoder().decode(
      new Uint8Array(payload.buffer, payload.byteOffset, payload.byteLength)
    );
  }

  if (typeof text !== "string") {
    throw new Error(`${label} must be a string, ArrayBuffer, or typed array.`);
  }

  try {
    return JSON.parse(text);
  } catch (parse_error) {
    throw new Error(`${label} is not valid JSON: ${parse_error?.message ?? parse_error}`);
  }
}

export function read_file(file_path) {
  if (typeof XMLHttpRequest === "undefined") {
    return null;
  }

  try {
    const url = resolve_resource_url(file_path);
    const check_xhr = new XMLHttpRequest();
    check_xhr.open("HEAD", url, false);
    check_xhr.send(null);

    if (check_xhr.status !== 200) {
      return null;
    }

    const get_xhr = new XMLHttpRequest();
    get_xhr.open("GET", url, false);
    get_xhr.send(null);
    if (get_xhr.status !== 200 || get_xhr.responseText.includes("<!DOCTYPE html>")) {
      return null;
    }
    return get_xhr.responseText;
  } catch {
    return null;
  }
}

export async function fetch_resource(file_path, options = {}) {
  try {
    const url = resolve_resource_url(file_path, options.base_url ?? get_default_base_url());
    const response = await fetch(url, options.fetch_options);
    return response.ok ? response : null;
  } catch {
    return null;
  }
}

export async function read_file_async(file_path, options = {}) {
  const response = await fetch_resource(file_path, options);
  if (!response) {
    return null;
  }

  const asset = await response.text();
  if (asset.includes("<!DOCTYPE html>")) {
    return null;
  }
  return asset;
}

export async function read_file_bytes_async(file_path, options = {}) {
  const response = await fetch_resource(file_path, options);
  if (!response || is_html_content_type(response.headers.get("content-type"))) {
    return null;
  }
  return await response.arrayBuffer();
}

export async function read_file_range_async(file_path, start, end, options = {}) {
  const range_start = Math.max(0, Math.floor(Number(start)));
  const range_end = Math.max(range_start, Math.floor(Number(end)));
  const fetch_options = options.fetch_options ?? {};
  const headers = new Headers(fetch_options.headers ?? {});
  headers.set("Range", `bytes=${range_start}-${range_end}`);

  const response = await fetch_resource(file_path, {
    ...options,
    fetch_options: {
      ...fetch_options,
      headers,
    },
  });
  if (!response || is_html_content_type(response.headers.get("content-type"))) {
    return null;
  }

  const payload = new Uint8Array(await response.arrayBuffer());
  const expected_byte_length = range_end - range_start + 1;
  if (response.status === 206) {
    if (payload.byteLength !== expected_byte_length) {
      throw new Error(
        `Range '${range_start}-${range_end}' from '${file_path}' returned ${payload.byteLength} bytes; ${expected_byte_length} were expected.`
      );
    }
    return payload;
  }

  if (payload.byteLength < range_end + 1) {
    throw new Error(`Asset '${file_path}' does not contain range '${range_start}-${range_end}'.`);
  }
  return payload.slice(range_start, range_end + 1);
}

export async function read_json_async(file_path, options = {}) {
  const text = await read_file_async(file_path, options);
  if (text === null) {
    return null;
  }
  return deserialize_json(text, options.label ?? `JSON asset '${file_path}'`);
}

export function resolve_manifest_asset_path(manifest_path, asset_path, fallback_path = null) {
  const resolved_asset_path = asset_path || fallback_path;
  if (!resolved_asset_path) {
    return null;
  }

  const normalized_asset_path = String(resolved_asset_path).replace(/\\/g, "/");
  if (normalized_asset_path.startsWith("/") || /^[a-z][a-z\d+.-]*:/i.test(normalized_asset_path)) {
    return normalized_asset_path;
  }

  const normalized_manifest_path = String(manifest_path ?? "").replace(/\\/g, "/");
  const base_path_index = normalized_manifest_path.lastIndexOf("/");
  const base_path =
    base_path_index >= 0 ? normalized_manifest_path.slice(0, base_path_index + 1) : "";
  return `${base_path}${normalized_asset_path}`;
}

/**
 * Loads a JSON manifest and its referenced binary payload as one serialized
 * asset bundle.
 */
export async function read_binary_manifest_async(manifest_path, options = {}) {
  const label = options.label ?? "Binary asset";
  const manifest_text = await read_file_async(manifest_path, options);
  if (manifest_text === null) {
    if (options.optional) {
      return null;
    }
    throw new Error(`${label} manifest '${manifest_path}' could not be loaded.`);
  }

  const manifest = deserialize_json(manifest_text, `${label} manifest '${manifest_path}'`);
  const binary_path = options.resolve_binary_path?.(manifest, manifest_path);
  if (!binary_path) {
    throw new Error(`${label} manifest '${manifest_path}' is missing its binary asset path.`);
  }

  const binary = await read_file_bytes_async(binary_path, options);
  if (!(binary instanceof ArrayBuffer)) {
    if (options.optional) {
      return null;
    }
    throw new Error(`${label} binary '${binary_path}' could not be loaded.`);
  }

  return {
    manifest,
    binary,
    manifest_path,
    binary_path,
  };
}
