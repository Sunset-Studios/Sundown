const default_shader_archive_asset_dir = "engine/shaders";
const default_shader_archive_manifest_name = "shaders.cooked.json";
const default_shader_archive_binary_name = "shaders.cooked.bin";

function normalize_path_segment(value) {
  const normalized = value
    .replace(/\\/g, "/")
    .replace(/^\/+|\/+$/g, "");
  return normalized.length > 0 ? normalized : value;
}

function resolve_config_value(config_key, fallback) {
  let config_value =
    (typeof import.meta !== "undefined" && import.meta.env) ? import.meta.env[config_key] : null;

  if (config_value === undefined || config_value === null || config_value === "") {
    config_value = (typeof process !== "undefined" && process.env) ? process.env[config_key] : null;
  }

  if (config_value === undefined || config_value === null || config_value === "") {
    config_value = fallback;
  }

  return config_value;
}

export const shader_archive_asset_dir = normalize_path_segment(
  resolve_config_value("SUNDOWN_SHADER_ARCHIVE_DIR", default_shader_archive_asset_dir)
);

export const shader_archive_manifest_name = normalize_path_segment(
  resolve_config_value("SUNDOWN_SHADER_ARCHIVE_MANIFEST_NAME", default_shader_archive_manifest_name)
);

export const shader_archive_binary_name = normalize_path_segment(
  resolve_config_value("SUNDOWN_SHADER_ARCHIVE_BINARY_NAME", default_shader_archive_binary_name)
);

export function build_shader_archive_asset_path(file_name, asset_dir = shader_archive_asset_dir) {
  const normalized_file_name = normalize_path_segment(file_name);
  const normalized_asset_dir = normalize_path_segment(asset_dir);
  return `${normalized_asset_dir}/${normalized_file_name}`;
}

export function get_project_shader_root_asset_path(project_root) {
  const normalized_project_root = normalize_path_segment(project_root);
  return `${normalized_project_root}/shaders`;
}

export function get_project_source_root(project_root) {
  return normalize_path_segment(project_root);
}

export const shader_archive_manifest_asset_path = build_shader_archive_asset_path(
  shader_archive_manifest_name
);
