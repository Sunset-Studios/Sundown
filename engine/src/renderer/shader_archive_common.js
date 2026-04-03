const boolean_string = "boolean";
const number_string = "number";
const string_string = "string";

export const ShaderPrecisionProfile = Object.freeze({
  F16: "f16",
  F32: "f32",
});

function normalize_define_value(value) {
  if (value === undefined) {
    return true;
  }

  if (
    typeof value === boolean_string ||
    typeof value === number_string ||
    typeof value === string_string ||
    value === null
  ) {
    return value;
  }

  return JSON.stringify(value);
}

export function canonicalize_shader_defines(defines = {}) {
  if (!defines) {
    return {};
  }

  return Object.fromEntries(
    Object.entries(defines)
      .filter(([key]) => typeof key === string_string && key.length > 0)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, value]) => [key, normalize_define_value(value)])
  );
}

export function create_shader_variant_key(
  file_path,
  defines = {},
  precision_profile = ShaderPrecisionProfile.F32
) {
  const canonical_defines = canonicalize_shader_defines(defines);
  const define_entries = Object.entries(canonical_defines).map(([key, value]) => [key, value]);
  return `${file_path}|${precision_profile}|${JSON.stringify(define_entries)}`;
}
