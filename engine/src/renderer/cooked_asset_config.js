const default_cooked_asset_subdir = "cooked";
const default_shader_archive_manifest_name = "shaders.cooked.json";
const default_shader_archive_binary_name = "shaders.cooked.bin";

function normalize_path_segment(value, fallback) {
  const normalized = String(value ?? fallback)
    .replace(/\\/g, "/")
    .replace(/^\/+|\/+$/g, "");
  return normalized.length > 0 ? normalized : fallback;
}

function resolve_config_value(vite_key, process_key, fallback) {
  const vite_value =
    typeof import.meta !== "undefined" && import.meta.env ? import.meta.env[vite_key] : undefined;
  if (vite_value !== undefined && vite_value !== null && vite_value !== "") {
    return vite_value;
  }

  const process_value =
    typeof process !== "undefined" && process.env
      ? process.env[process_key] ?? process.env[vite_key]
      : undefined;
  if (process_value !== undefined && process_value !== null && process_value !== "") {
    return process_value;
  }

  return fallback;
}

export const cooked_asset_subdir = normalize_path_segment(
  resolve_config_value("VITE_COOKED_ASSET_DIR", "SUNDOWN_COOKED_ASSET_DIR", default_cooked_asset_subdir),
  default_cooked_asset_subdir
);

export const shader_archive_manifest_name = normalize_path_segment(
  resolve_config_value(
    "VITE_SHADER_ARCHIVE_MANIFEST_NAME",
    "SUNDOWN_SHADER_ARCHIVE_MANIFEST_NAME",
    default_shader_archive_manifest_name
  ),
  default_shader_archive_manifest_name
);

export const shader_archive_binary_name = normalize_path_segment(
  resolve_config_value(
    "VITE_SHADER_ARCHIVE_BINARY_NAME",
    "SUNDOWN_SHADER_ARCHIVE_BINARY_NAME",
    default_shader_archive_binary_name
  ),
  default_shader_archive_binary_name
);

export function build_cooked_asset_path(file_name) {
  const normalized_file_name = normalize_path_segment(file_name, file_name);
  return cooked_asset_subdir.length > 0
    ? `${cooked_asset_subdir}/${normalized_file_name}`
    : normalized_file_name;
}

export const shader_archive_manifest_asset_path = build_cooked_asset_path(
  shader_archive_manifest_name
);

export const shader_archive_binary_asset_path = build_cooked_asset_path(
  shader_archive_binary_name
);
