#include "common.wgsl"
#include "gi/scgi_common.wgsl"

@group(1) @binding(0) var<uniform> scgi_params: SCGIParams;
@group(1) @binding(1) var<storage, read_write> surface_cache: array<SurfacePatchReadOnly>;
@group(1) @binding(2) var<storage, read_write> surface_cache_sh: array<u32>;
@group(1) @binding(3) var<storage, read_write> surface_cache_sh_filtered: array<u32>;
@group(1) @binding(4) var<storage, read> active_indices: array<u32>;
@group(1) @binding(5) var<storage, read> counters: SCGICountersReadOnly;

#include "gi/scgi_cache_lookup.wgsl"

// Denoise the persistent cache rather than the final image. Each active patch
// gathers a 5x5 neighborhood in its tangent plane. Spatial, surface-depth,
// normal, and convergence weights reject unrelated geometry while giving the
// progressive one-ray-per-cell estimator enough support to suppress persistent
// high-frequency noise. Neighbor SH is rotated into the center patch's
// hemisphere before accumulation. Raw and filtered buffers remain separate,
// so results never depend on GPU invocation order.
@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let active_index = gid.x;
    if (active_index >= counters.active_patch_count) {
        return;
    }

    let patch_index = active_indices[active_index];
    let center_patch = surface_cache[patch_index];
    let center_sh = scgi_sh_patch_read(&surface_cache_sh, patch_index);
    if (center_patch.history.x <= 0.0) {
        scgi_sh_patch_write(&surface_cache_sh_filtered, patch_index, center_sh);
        return;
    }

    let center_position = center_patch.position_frame.xyz;
    let center_normal = safe_normalize(center_patch.normal_unused.xyz);
    let lod = min(
        u32(max(center_patch.material_props.w, 0.0)),
        u32(scgi_params.surface_cache_lod_count) - 1u
    );
    let cell_size = scgi_lod_cell_size(lod, scgi_params);
    let center_descriptor = scgi_quantize_position(center_position, lod, scgi_params);
    let quantized_normal = scgi_quantize_normal(center_normal);

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
    let spatial_sigma = 1.25;
    let inverse_spatial_variance = 1.0 / (spatial_sigma * spatial_sigma);

    for (var tap_y: i32 = -2; tap_y <= 2; tap_y = tap_y + 1) {
        for (var tap_x: i32 = -2; tap_x <= 2; tap_x = tap_x + 1) {
            let tap_offset = vec2<i32>(tap_x, tap_y);
            let descriptor = scgi_surface_corner_descriptor(
                center_position,
                center_normal,
                center_tangent_cell + tap_offset,
                dominant_axis,
                cell_size
            );
            let neighbor_index_i = scgi_find_patch(descriptor, quantized_normal, lod);
            if (neighbor_index_i < 0) {
                continue;
            }

            let neighbor_index = u32(neighbor_index_i);
            let neighbor_patch = surface_cache[neighbor_index];
            let sample_count = neighbor_patch.history.x;
            if (sample_count <= 0.0) {
                continue;
            }

            let neighbor_normal = safe_normalize(neighbor_patch.normal_unused.xyz);
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
            let confidence_weight = clamp(sample_count / 8.0, 0.25, 1.0);
            let weight = spatial_weight
                * plane_weight
                * normal_alignment * normal_alignment
                * confidence_weight;
            if (weight <= 1e-5) {
                continue;
            }

            let neighbor_sh = scgi_rotate_sh_between_hemispheres(
                scgi_sh_patch_read(&surface_cache_sh, neighbor_index),
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
    var result = center_sh;
    if (weight_sum > 1e-5) {
        result = sh_l1_rgb_multiply_scalar(filtered_sh, 1.0 / weight_sum);
    }
    scgi_sh_patch_write(
        &surface_cache_sh_filtered,
        patch_index,
        result
    );
}
