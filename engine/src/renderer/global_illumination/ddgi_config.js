export const DDGI_DEFAULT_CONFIG = Object.freeze({
  probe_grid_dimensions: Object.freeze([64, 64, 64]),
  probe_spacing: 1.0,
  probe_radius: 0.1,
  max_rays_per_probe: 64,
  probes_per_frame: 1024,
  indirect_boost: 1.0,
  cascade_count: 6,
  cascade_spacing_multiplier: 2.0,
  probe_depth_resolutions: Object.freeze([16, 16, 16, 16, 16, 16]),
  probe_depth_slot_count: 65536,
  probe_depth_slot_retention_frames: 120,
  max_emissive_lights: 32768,
  diffuse_sample_upscale_factor: 1,
  diffuse_atrous_enabled: false,
  diffuse_atrous_pass_count: 3,
  diffuse_atrous_phi_depth: 0.04,
  diffuse_atrous_phi_normal: 64.0,
  diffuse_atrous_luma_sigma: 1.0,
});

const resource_config_keys = new Set([
  "probe_grid_dimensions",
  "probe_spacing",
  "max_rays_per_probe",
  "probes_per_frame",
  "cascade_count",
  "cascade_spacing_multiplier",
  "probe_depth_resolutions",
  "probe_depth_slot_count",
  "max_emissive_lights",
  "diffuse_sample_upscale_factor",
]);

export function clone_ddgi_config_value(value) {
  return Array.isArray(value) ? [...value] : value;
}

export function ddgi_config_values_equal(a, b) {
  if (Array.isArray(a) || Array.isArray(b)) {
    return (
      Array.isArray(a) &&
      Array.isArray(b) &&
      a.length === b.length &&
      a.every((value, index) => Object.is(value, b[index]))
    );
  }
  return Object.is(a, b);
}

export function ddgi_config_change_requires_rebuild(key) {
  return resource_config_keys.has(key);
}

export function create_ddgi_config(overrides = {}) {
  const config = {};
  for (const [key, value] of Object.entries(DDGI_DEFAULT_CONFIG)) {
    config[key] = clone_ddgi_config_value(value);
  }
  for (const [key, value] of Object.entries(overrides)) {
    config[key] = clone_ddgi_config_value(value);
  }
  return config;
}

export function summarize_ddgi_config(config) {
  const dimensions = config.probe_grid_dimensions ?? [0, 0, 0];
  const cascades = Math.max(1, Math.floor(config.cascade_count ?? 1));
  const probes_per_cascade = dimensions[0] * dimensions[1] * dimensions[2];
  const total_probes = probes_per_cascade * cascades;
  const scheduled_probes =
    config.probes_per_frame === 0
      ? total_probes
      : Math.min(total_probes, Math.max(0, Math.floor(config.probes_per_frame ?? 0)));
  return {
    total_probes,
    rays_per_frame: scheduled_probes * Math.max(1, Math.floor(config.max_rays_per_probe ?? 1)),
    depth_slots: Math.min(
      total_probes,
      Math.max(1, Math.floor(config.probe_depth_slot_count ?? 1))
    ),
  };
}
