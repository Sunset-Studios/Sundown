#include "common.wgsl"
#include "gi/scgi_common.wgsl"

@group(1) @binding(0) var<uniform> scgi_params: SCGIParams;
@group(1) @binding(1) var<storage, read_write> surface_cache: array<SurfacePatch>;
@group(1) @binding(2) var<storage, read_write> surface_cache_sh: array<u32>;
@group(1) @binding(3) var<storage, read> active_indices: array<u32>;
@group(1) @binding(4) var<storage, read> counters: SCGICountersReadOnly;
@group(1) @binding(5) var<storage, read> hit_info: array<SCGIHitInfo>;
@group(1) @binding(6) var<storage, read> radiance_info: array<SCGIRadianceInfo>;

// Accumulate one progressive hemisphere sample into each active patch. Early
// samples use a running mean; once max_history_samples is reached, the update
// becomes the configured EMA. Spatial variance is handled by scgi_filter.
@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let active_index = gid.x;
    if (
        active_index >= counters.active_patch_count ||
        radiance_info[active_index].sample_radiance.w <= 0.0
    ) {
        return;
    }

    let patch_index = active_indices[active_index];
    var sample_radiance = radiance_info[active_index].sample_radiance.xyz;
    if (radiance_info[active_index].shadow_radiance.w == 2.0) {
        sample_radiance += radiance_info[active_index].shadow_radiance.xyz;
    }
    sample_radiance = safe_clamp_vec3_max(sample_radiance, SCGI_MAX_RADIANCE);

    let patch_normal = safe_normalize(surface_cache[patch_index].normal_unused.xyz);
    let local_direction = safe_normalize(scgi_world_to_hemisphere(
        hit_info[active_index].ray_direction_primitive.xyz,
        patch_normal
    ));
    let sample_sh = sh_project_onto_l1_rgb(local_direction, sample_radiance * (2.0 * PI));

    let history = surface_cache[patch_index].history;
    let previous_sample_count = history.x;
    let next_sample_count = min(
        previous_sample_count + 1.0,
        max(scgi_params.max_history_samples, 1.0)
    );
    let next_sequence = f32((u32(history.y) + 1u) & 4095u);
    let running_alpha = 1.0 / max(next_sample_count, 1.0);
    let ema_alpha = clamp(1.0 - scgi_params.history_hysteresis, 0.0, 1.0);
    let blend_alpha = max(running_alpha, ema_alpha);

    var result = sample_sh;
    if (previous_sample_count > 0.0) {
        result = sh_l1_rgb_lerp(
            scgi_sh_patch_read(&surface_cache_sh, patch_index),
            sample_sh,
            blend_alpha
        );
    }

    scgi_sh_patch_write(&surface_cache_sh, patch_index, result);
    surface_cache[patch_index].history = vec4<f32>(next_sample_count, next_sequence, 0.0, 0.0);
}
