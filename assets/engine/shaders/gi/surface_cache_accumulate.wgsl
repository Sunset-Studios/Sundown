#include "common.wgsl"
#include "gi/surface_cache_common.wgsl"

@group(1) @binding(0) var<uniform> surface_cache_params: SurfaceCacheParams;
@group(1) @binding(1) var<storage, read_write> surface_cache: array<SurfacePatch>;
@group(1) @binding(2) var<storage, read_write> surface_cache_sh: array<u32>;
@group(1) @binding(3) var<storage, read> active_indices: array<u32>;
@group(1) @binding(4) var<storage, read> counters: SurfaceCacheCountersReadOnly;
@group(1) @binding(5) var<storage, read> hit_info: array<SurfaceCacheHitInfo>;
@group(1) @binding(6) var<storage, read> radiance_info: array<SurfaceCacheRadianceInfo>;

// Reduce one ray batch into a single race-free SH update per active patch.
// Early samples use a running mean; once max_history_samples is reached, the
// update becomes the configured EMA. Spatial variance is handled by the filter.
@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let active_index = gid.x;
    if (active_index >= counters.active_patch_count) {
        return;
    }

    let patch_index = active_indices[active_index];
    let patch_normal = safe_normalize(surface_cache[patch_index].normal_lod.xyz);
    let rays_per_patch = surface_cache_rays_per_patch(surface_cache_params);
    let ray_data_base = active_index * rays_per_patch;
    var sample_sh_sum = sh_l1_rgb_zero();
    var luminance_sum = 0.0;
    var luminance_squared_sum = 0.0;
    var valid_sample_count = 0.0;

    for (var ray_index = 0u; ray_index < rays_per_patch; ray_index = ray_index + 1u) {
        let ray_data_index = ray_data_base + ray_index;
        if (radiance_info[ray_data_index].sample_radiance.w <= 0.0) {
            continue;
        }

        var sample_radiance = radiance_info[ray_data_index].sample_radiance.xyz;
        if (radiance_info[ray_data_index].shadow_radiance.w == 2.0) {
            sample_radiance += radiance_info[ray_data_index].shadow_radiance.xyz;
        }
        sample_radiance = safe_clamp_vec3_max(
            sample_radiance,
            SURFACE_CACHE_MAX_RADIANCE
        );
        let local_direction = safe_normalize(surface_cache_world_to_hemisphere(
            hit_info[ray_data_index].ray_direction_primitive.xyz,
            patch_normal
        ));
        let sampling_weight = max(
            hit_info[ray_data_index].hit_position_sampling_weight.w,
            0.0
        );
        sample_sh_sum = sh_l1_rgb_add(
            sample_sh_sum,
            sh_project_onto_l1_rgb(
                local_direction,
                sample_radiance * sampling_weight
            )
        );
        let sample_luminance = luminance(sample_radiance);
        luminance_sum += sample_luminance;
        luminance_squared_sum += sample_luminance * sample_luminance;
        valid_sample_count += 1.0;
    }

    if (valid_sample_count <= 0.0) {
        return;
    }

    let inverse_sample_count = 1.0 / valid_sample_count;
    let sample_sh = sh_l1_rgb_multiply_scalar(sample_sh_sum, inverse_sample_count);

    let history = surface_cache[patch_index].history;
    let previous_sample_count = history.x;
    let maximum_history = max(surface_cache_params.max_history_samples, 1.0);
    // Keep the true sample total for the article's trace-stopping threshold;
    // only the statistical blend window is bounded by max_history_samples.
    let next_sample_count = previous_sample_count + valid_sample_count;
    let effective_history_count = min(next_sample_count, maximum_history);
    let next_sequence = f32((u32(history.y) + u32(valid_sample_count)) & 4095u);
    let running_alpha = min(
        valid_sample_count / max(effective_history_count, 1.0),
        1.0
    );
    // Once the running-mean history is mature, keep a bounded EMA response so
    // lighting changes do not remain trapped for hundreds of frames.
    let blend_alpha = max(
        running_alpha,
        1.0 - clamp(surface_cache_params.history_hysteresis, 0.0, 0.999)
    );

    var result = sample_sh;
    if (previous_sample_count > 0.0) {
        result = sh_l1_rgb_lerp(
            surface_cache_sh_patch_read(&surface_cache_sh, patch_index),
            sample_sh,
            blend_alpha
        );
    }

    surface_cache_sh_patch_write(&surface_cache_sh, patch_index, result);
    let sample_luminance = luminance_sum * inverse_sample_count;
    let sample_luminance_squared = luminance_squared_sum * inverse_sample_count;
    let moment_alpha = max(running_alpha, blend_alpha);
    let first_moment = mix(history.z, sample_luminance, moment_alpha);
    let second_moment = mix(history.w, sample_luminance_squared, moment_alpha);
    surface_cache[patch_index].history = vec4<f32>(
        next_sample_count,
        next_sequence,
        first_moment,
        second_moment
    );
}
