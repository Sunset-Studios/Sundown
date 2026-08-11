#include "common.wgsl"
#include "gi/surface_cache_common.wgsl"

@group(1) @binding(0) var<uniform> surface_cache_params: SurfaceCacheParams;
@group(1) @binding(1) var<storage, read_write> surface_cache: array<SurfacePatch>;
@group(1) @binding(2) var<storage, read_write> surface_cache_sh: array<u32>;
@group(1) @binding(3) var<storage, read> update_indices: array<u32>;
@group(1) @binding(4) var<storage, read> counters: SurfaceCacheCountersReadOnly;
@group(1) @binding(5) var<storage, read> hit_info: array<SurfaceCacheHitInfo>;
@group(1) @binding(6) var<storage, read> radiance_info: array<SurfaceCacheRadianceInfo>;
@group(1) @binding(7) var<uniform> ray_batch: SurfaceCacheRayBatchParams;

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let active_index = gid.x;
    if (active_index >= surface_cache_ray_batch_patch_count(counters, ray_batch)) {
        return;
    }

    let patch_index = update_indices[active_index];
    let rays_per_patch = surface_cache_ray_batch_rays_per_patch(
        counters,
        ray_batch
    );
    let local_ray_base = active_index * rays_per_patch;

    var sample_sh_sum = sh_l1_rgb_zero();
    var luminance_sum = 0.0;
    var luminance_squared_sum = 0.0;
    var valid_sample_count = 0.0;

    for (var ray_index = 0u; ray_index < rays_per_patch; ray_index = ray_index + 1u) {
        let ray_data_index = surface_cache_ray_batch_data_index(
            local_ray_base + ray_index,
            arrayLength(&radiance_info),
            ray_batch
        );
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
        // Store the cell's directional signal in world space. Ray origins and
        // hemisphere normals intentionally move across the cell, so a basis
        // tied to the current representative surface sample cannot persist.
        let sample_direction = safe_normalize(
            hit_info[ray_data_index].ray_direction_sampling_weight.xyz
        );
        let sampling_weight = max(
            hit_info[ray_data_index].ray_direction_sampling_weight.w,
            0.0
        );
        sample_sh_sum = sh_l1_rgb_add(
            sample_sh_sum,
            sh_project_onto_l1_rgb(
                sample_direction,
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
    let sample_luminance = luminance_sum * inverse_sample_count;
    let sample_luminance_squared = luminance_squared_sum * inverse_sample_count;

    let history = surface_cache[patch_index].history;
    let previous_sample_count = history.x;
    let maximum_history_samples = max(
        surface_cache_params.max_history_samples,
        1.0
    );
    // Bound the statistical memory used by the running mean. Without this,
    // mature cells eventually give a new lighting batch almost zero weight and
    // can take tens of seconds to converge after a lighting change.
    let effective_previous_sample_count = min(
        previous_sample_count,
        maximum_history_samples
    );
    let next_sample_count =
        effective_previous_sample_count + valid_sample_count;
    let next_sequence = f32((u32(history.y) + u32(valid_sample_count)) & 4095u);
    let running_alpha = min(
        valid_sample_count / max(next_sample_count, 1.0),
        1.0
    );

    // A permanent EMA floor leaves stationary Monte Carlo noise visible. Let
    // stable cells use a true progressive mean, and only enable the configured
    // response when the new batch differs from history by more than its
    // estimated sampling error.
    let previous_variance = max(
        history.w - history.z * history.z,
        0.0
    );
    let sample_variance = max(
        sample_luminance_squared - sample_luminance * sample_luminance,
        0.0
    );
    let mean_variance =
        previous_variance / max(effective_previous_sample_count, 1.0) +
        sample_variance / max(valid_sample_count, 1.0);
    let change_threshold = max(3.0 * sqrt(mean_variance), 0.01);
    let history_is_mature =
        previous_sample_count >= maximum_history_samples;
    let lighting_changed = history_is_mature &&
        abs(sample_luminance - history.z) > change_threshold;
    let response_alpha = 1.0 - clamp(
        surface_cache_params.history_hysteresis,
        0.0,
        0.999
    );
    let blend_alpha = max(
        running_alpha,
        select(0.0, response_alpha, lighting_changed)
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

    let first_moment = mix(history.z, sample_luminance, blend_alpha);
    let second_moment = mix(
        history.w,
        sample_luminance_squared,
        blend_alpha
    );
    // When a change is detected, retain a sample count consistent with the
    // faster response so the following stable batches can converge again.
    let responsive_sample_count = valid_sample_count / max(blend_alpha, 1e-6);
    let stored_sample_count = select(
        next_sample_count,
        min(next_sample_count, responsive_sample_count),
        lighting_changed
    );
    surface_cache[patch_index].history = vec4<f32>(
        min(stored_sample_count, maximum_history_samples),
        next_sequence,
        first_moment,
        second_moment
    );
    surface_cache[patch_index].metadata.x = surface_cache_params.frame_index;
    // A large bootstrap batch improves the first estimate without jumping the
    // footprint across every intermediate LOD in one frame.
    let footprint_history_increment = min(
        valid_sample_count,
        max(surface_cache_params.rays_per_patch, 1.0)
    );
    surface_cache[patch_index].metadata.w = min(
        surface_cache[patch_index].metadata.w + footprint_history_increment,
        max(surface_cache_params.history_footprint_end_samples, 1.0)
    );
}
