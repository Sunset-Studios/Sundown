#include "common.wgsl"
#include "gi/surface_cache_common.wgsl"

@group(1) @binding(0) var<uniform> surface_cache_params: SurfaceCacheParams;
@group(1) @binding(1) var<storage, read_write> surface_cache: array<SurfacePatchReadOnly>;
@group(1) @binding(2) var<storage, read_write> surface_cache_sh: array<u32>;
@group(1) @binding(3) var<storage, read_write> surface_cache_sh_filtered: array<u32>;
@group(1) @binding(4) var<storage, read> active_indices: array<u32>;
@group(1) @binding(5) var<storage, read> counters: SurfaceCacheCountersReadOnly;
@group(1) @binding(6) var<storage, read> surface_cache_hashmap: array<HashMapEntry>;

#include "gi/surface_cache_lookup.wgsl"

const SURFACE_CACHE_FILTER_WARMUP_START: f32 = 8.0;
const SURFACE_CACHE_FILTER_WARMUP_END: f32 = 64.0;
const SURFACE_CACHE_FILTER_ERROR_START: f32 = 0.04;
const SURFACE_CACHE_FILTER_ERROR_END: f32 = 0.20;
const SURFACE_CACHE_FILTER_SIGNAL_FLOOR: f32 = 0.05;
const SURFACE_CACHE_FILTER_RADIANCE_FLOOR: f32 = 0.08;
const SURFACE_CACHE_FILTER_RADIANCE_SIGMA_SCALE: f32 = 3.0;
const SURFACE_CACHE_FILTER_TEMPORAL_RESPONSE: f32 = 0.05;
const SURFACE_CACHE_FILTER_CHANGE_START: f32 = 0.10;
const SURFACE_CACHE_FILTER_CHANGE_END: f32 = 0.50;
const SURFACE_CACHE_FILTER_EDGE_CONFIDENCE_START: f32 = 16.0;
const SURFACE_CACHE_FILTER_EDGE_CONFIDENCE_END: f32 = 64.0;
const SURFACE_CACHE_FILTER_CONFIDENCE_START: f32 = 4.0;
const SURFACE_CACHE_FILTER_CONFIDENCE_END: f32 = 32.0;
const SURFACE_CACHE_FILTER_CONFIDENCE_FLOOR: f32 = 0.0625;
const SURFACE_CACHE_FILTER_TEMPORAL_HISTORY_START: f32 = 16.0;
const SURFACE_CACHE_FILTER_IDLE_RESPONSE_SCALE: f32 = 0.25;

struct SurfaceCacheFilterStatistics {
    effective_sample_count: f32,
    mean_luminance: f32,
    standard_error: f32,
    relative_standard_error: f32,
};

// Convert the stored per-ray luminance moments into uncertainty in the patch
// mean. Dividing by the bounded statistical history is important: raw variance
// describes the lighting distribution, while variance of the mean describes
// how much spatial support the reconstructed irradiance still needs.
fn surface_cache_filter_statistics(history: vec4<f32>) -> SurfaceCacheFilterStatistics {
    let maximum_history = max(surface_cache_params.max_history_samples, 1.0);
    let effective_sample_count = clamp(history.x, 1.0, maximum_history);
    let mean_luminance = max(history.z, 0.0);
    let variance = max(history.w - history.z * history.z, 0.0);
    let standard_error = sqrt(variance / effective_sample_count);
    let relative_standard_error = standard_error / max(
        mean_luminance,
        SURFACE_CACHE_FILTER_SIGNAL_FLOOR
    );
    return SurfaceCacheFilterStatistics(
        effective_sample_count,
        mean_luminance,
        standard_error,
        relative_standard_error
    );
}

fn surface_cache_filter_strength(stats: SurfaceCacheFilterStatistics) -> f32 {
    let warmup_strength = 1.0 - smoothstep(
        SURFACE_CACHE_FILTER_WARMUP_START,
        SURFACE_CACHE_FILTER_WARMUP_END,
        stats.effective_sample_count
    );
    let variance_strength = smoothstep(
        SURFACE_CACHE_FILTER_ERROR_START,
        SURFACE_CACHE_FILTER_ERROR_END,
        stats.relative_standard_error
    );
    return clamp(max(warmup_strength, variance_strength), 0.0, 1.0);
}

// A statistically significant luminance difference is treated as a real
// irradiance boundary. Noisy estimates get a wider acceptance interval, while
// converged neighbors must closely agree before sharing their SH coefficients.
fn surface_cache_filter_radiance_weight(
    center: SurfaceCacheFilterStatistics,
    neighbor: SurfaceCacheFilterStatistics
) -> f32 {
    let luminance_delta = abs(neighbor.mean_luminance - center.mean_luminance);
    let combined_standard_error = sqrt(
        center.standard_error * center.standard_error +
        neighbor.standard_error * neighbor.standard_error
    );
    let signal_scale = max(
        max(center.mean_luminance, neighbor.mean_luminance),
        SURFACE_CACHE_FILTER_SIGNAL_FLOOR
    );
    let radiance_sigma = max(
        combined_standard_error * SURFACE_CACHE_FILTER_RADIANCE_SIGMA_SCALE,
        signal_scale * SURFACE_CACHE_FILTER_RADIANCE_FLOOR
    );
    let normalized_delta = luminance_delta / max(radiance_sigma, 1e-4);
    return exp(-0.5 * normalized_delta * normalized_delta);
}

// Denoise the persistent cache rather than the final image. The filter keeps
// converged centers untouched, then continuously increases the existing 3x3
// tangent-plane support as uncertainty rises. Geometry weights prevent surface
// leaks; statistical radiance weights preserve genuine lighting boundaries.
// Raw and filtered buffers remain separate, so invocation order cannot feed
// filtered data back into the same dispatch.
@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let active_index = gid.x;
    if (active_index >= counters.active_patch_count) {
        return;
    }

    let patch_index = active_indices[active_index];
    let center_patch = surface_cache[patch_index];
    let center_sh = surface_cache_sh_patch_read(&surface_cache_sh, patch_index);
    if (center_patch.history.x <= 0.0) {
        surface_cache_sh_patch_write(&surface_cache_sh_filtered, patch_index, center_sh);
        return;
    }

    let center_stats = surface_cache_filter_statistics(center_patch.history);
    let filter_strength = surface_cache_filter_strength(center_stats);
    if (filter_strength <= 1e-4) {
        surface_cache_sh_patch_write(&surface_cache_sh_filtered, patch_index, center_sh);
        return;
    }

    let center_position = center_patch.position_frame.xyz;
    let center_normal = safe_normalize(center_patch.normal_cell_exponent.xyz);
    let cell_exponent = surface_cache_grid_key_cell_exponent(center_patch.grid_key);
    let cell_size = surface_cache_cell_size(cell_exponent);
    let center_descriptor = center_patch.grid_key.xyz;
    let quantized_normal = surface_cache_quantize_normal(center_normal);

    let absolute_normal = abs(center_normal);
    let dominant_axis = select(
        select(2u, 1u, absolute_normal.y >= absolute_normal.z),
        0u,
        absolute_normal.x >= max(absolute_normal.y, absolute_normal.z)
    );
    var center_tangent_cell = center_descriptor.xy;
    if (dominant_axis == 0u) {
        center_tangent_cell = center_descriptor.yz;
    } else if (dominant_axis == 1u) {
        center_tangent_cell = center_descriptor.xz;
    }

    var filtered_sh = sh_l1_rgb_zero();
    var weight_sum = 0.0;
    let plane_sigma = max(cell_size * 0.35, 0.001);
    let spatial_sigma = 0.85;
    let inverse_spatial_variance = 1.0 / (spatial_sigma * spatial_sigma);

    for (var tap_y: i32 = -1; tap_y <= 1; tap_y = tap_y + 1) {
        for (var tap_x: i32 = -1; tap_x <= 1; tap_x = tap_x + 1) {
            let tap_offset = vec2<i32>(tap_x, tap_y);
            let descriptor = surface_cache_corner_descriptor(
                center_position,
                center_normal,
                center_tangent_cell + tap_offset,
                dominant_axis,
                cell_size,
                surface_cache_params
            );
            var neighbor_index_i = i32(patch_index);
            if (tap_x != 0 || tap_y != 0) {
                neighbor_index_i = surface_cache_find_patch(
                    descriptor,
                    quantized_normal,
                    cell_exponent
                );
            }
            if (neighbor_index_i < 0) {
                continue;
            }

            let neighbor_index = u32(neighbor_index_i);
            let neighbor_patch = surface_cache[neighbor_index];
            let sample_count = neighbor_patch.history.x;
            if (sample_count <= 0.0) {
                continue;
            }

            let neighbor_normal = safe_normalize(
                neighbor_patch.normal_cell_exponent.xyz
            );
            let normal_alignment = clamp(
                (dot(center_normal, neighbor_normal) - 0.75) * 4.0,
                0.0,
                1.0
            );
            let plane_distance = abs(dot(
                neighbor_patch.position_frame.xyz - center_position,
                center_normal
            ));
            let normalized_plane_distance = plane_distance / plane_sigma;
            let plane_weight = exp(
                -0.5 * normalized_plane_distance * normalized_plane_distance
            );
            let spatial_distance_squared = f32(tap_x * tap_x + tap_y * tap_y);
            let spatial_weight = exp(
                -0.5 * spatial_distance_squared * inverse_spatial_variance
            );
            let neighbor_stats = surface_cache_filter_statistics(
                neighbor_patch.history
            );
            let sample_confidence = smoothstep(
                SURFACE_CACHE_FILTER_CONFIDENCE_START,
                SURFACE_CACHE_FILTER_CONFIDENCE_END,
                neighbor_stats.effective_sample_count
            );
            let variance_confidence = 1.0 / (
                1.0 + 4.0 * neighbor_stats.relative_standard_error
            );
            var confidence_weight = clamp(
                mix(
                    SURFACE_CACHE_FILTER_CONFIDENCE_FLOOR,
                    1.0,
                    sample_confidence * variance_confidence
                ),
                SURFACE_CACHE_FILTER_CONFIDENCE_FLOOR,
                1.0
            );
            var radiance_weight = surface_cache_filter_radiance_weight(
                center_stats,
                neighbor_stats
            );
            // A low-variance dark result is not a trustworthy lighting edge
            // while either patch is still young. Let geometry-compatible
            // neighbors bootstrap it, then restore the bilateral edge stop as
            // both estimates converge.
            let edge_confidence = smoothstep(
                SURFACE_CACHE_FILTER_EDGE_CONFIDENCE_START,
                SURFACE_CACHE_FILTER_EDGE_CONFIDENCE_END,
                min(
                    center_stats.effective_sample_count,
                    neighbor_stats.effective_sample_count
                )
            );
            radiance_weight = mix(1.0, radiance_weight, edge_confidence);
            if (neighbor_index == patch_index) {
                radiance_weight = 1.0;
            }
            let weight = spatial_weight
                * plane_weight
                * normal_alignment * normal_alignment
                * confidence_weight
                * radiance_weight;
            if (weight <= 1e-5) {
                continue;
            }

            let neighbor_sh = surface_cache_rotate_sh_between_hemispheres(
                surface_cache_sh_patch_read(&surface_cache_sh, neighbor_index),
                neighbor_normal,
                center_normal
            );
            filtered_sh = sh_l1_rgb_add(
                filtered_sh,
                sh_l1_rgb_multiply_scalar(neighbor_sh, weight)
            );
            weight_sum += weight;
        }
    }

    // The center patch is always a valid member of its own footprint, but keep
    // a raw fallback so malformed or transient metadata cannot write NaNs.
    var spatial_result = center_sh;
    if (weight_sum > 1e-5) {
        let neighborhood_sh = sh_l1_rgb_multiply_scalar(
            filtered_sh,
            1.0 / weight_sum
        );
        spatial_result = sh_l1_rgb_lerp(
            center_sh,
            neighborhood_sh,
            filter_strength
        );
    }

    // Reuse the previous filtered value as cache-space temporal support. This
    // costs no additional resource and suppresses the frame-to-frame changes
    // caused by phased patch updates. Large irradiance changes raise the
    // response immediately so genuine lighting transitions do not linger.
    var result = spatial_result;
    let has_filtered_history = center_patch.history.x >=
        SURFACE_CACHE_FILTER_TEMPORAL_HISTORY_START;
    if (has_filtered_history) {
        let previous_filtered_sh = surface_cache_sh_patch_read(
            &surface_cache_sh_filtered,
            patch_index
        );
        let previous_luminance = luminance(
            surface_cache_evaluate_local_sh_irradiance(previous_filtered_sh)
        );
        let current_luminance = luminance(
            surface_cache_evaluate_local_sh_irradiance(spatial_result)
        );
        let relative_change = abs(current_luminance - previous_luminance) / max(
            max(current_luminance, previous_luminance),
            SURFACE_CACHE_FILTER_SIGNAL_FLOOR
        );
        let change_response = smoothstep(
            SURFACE_CACHE_FILTER_CHANGE_START,
            SURFACE_CACHE_FILTER_CHANGE_END,
            relative_change
        );
        // High variance makes a one-frame luminance jump weak evidence of a
        // real lighting change. Suppress that fast path until the patch's
        // estimate becomes statistically reliable.
        let change_confidence = 1.0 / (
            1.0 + 4.0 * center_stats.relative_standard_error
        );
        var temporal_response = max(
            mix(1.0, SURFACE_CACHE_FILTER_TEMPORAL_RESPONSE, filter_strength),
            change_response * change_confidence
        );
        let updated_this_frame = u32(center_patch.metadata.x) ==
            u32(surface_cache_params.frame_index);
        temporal_response *= select(
            SURFACE_CACHE_FILTER_IDLE_RESPONSE_SCALE,
            1.0,
            updated_this_frame
        );
        result = sh_l1_rgb_lerp(
            previous_filtered_sh,
            spatial_result,
            temporal_response
        );
    }
    surface_cache_sh_patch_write(
        &surface_cache_sh_filtered,
        patch_index,
        result
    );
}
