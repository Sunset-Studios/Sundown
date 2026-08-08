#include "common.wgsl"
#include "gi/surface_cache_common.wgsl"
#include "gi/surface_cache_lookup.wgsl"

@group(1) @binding(0) var<uniform> surface_cache_params: SurfaceCacheParams;
@group(1) @binding(1) var<storage, read_write> surface_cache: array<SurfacePatchReadOnly>;
@group(1) @binding(2) var<storage, read_write> surface_cache_sh: array<u32>;
@group(1) @binding(3) var<storage, read_write> surface_cache_sh_filtered: array<u32>;
@group(1) @binding(4) var<storage, read> active_indices: array<u32>;
@group(1) @binding(5) var<storage, read> counters: SurfaceCacheCountersReadOnly;
@group(1) @binding(6) var<storage, read> surface_cache_hashmap: array<HashMapEntry>;

const SURFACE_CACHE_FILTER_OFFSETS = array<vec2<i32>, 4>(
    vec2<i32>(-1, 0),
    vec2<i32>(1, 0),
    vec2<i32>(0, -1),
    vec2<i32>(0, 1)
);

// Cache-space filtering only suppresses independent patch noise before the
// cache is reused recursively. The small mature-patch floor removes residual
// cell noise without adding another temporal loop in cache space.
@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let active_index = gid.x;
    if (active_index >= counters.active_patch_count) {
        return;
    }

    let patch_index = active_indices[active_index];
    let center_patch = surface_cache[patch_index];
    let center_sh = surface_cache_sh_patch_read(&surface_cache_sh, patch_index);
    let center_sample_count = center_patch.history.x;
    if (center_sample_count <= 0.0) {
        surface_cache_sh_patch_write(
            &surface_cache_sh_filtered,
            patch_index,
            center_sh
        );
        return;
    }

    let young_patch_strength = 1.0 - smoothstep(
        8.0,
        64.0,
        center_sample_count
    );
    let filter_strength = mix(0.2, 1.0, young_patch_strength);
    let center_position = center_patch.position_frame.xyz;
    let center_normal = safe_normalize(center_patch.normal_cell_exponent.xyz);
    let cell_exponent = surface_cache_grid_key_cell_exponent(center_patch.grid_key);
    let cell_size = surface_cache_cell_size(cell_exponent);
    let quantized_normal = surface_cache_quantize_normal(center_normal);
    let dominant_axis = surface_cache_dominant_axis(center_normal);
    let center_tangent_cell = vec2<i32>(surface_cache_tangent_components(
        vec3<f32>(center_patch.grid_key.xyz),
        dominant_axis
    ));

    var sh_sum = center_sh;
    var weight_sum = 1.0;
    let plane_scale = max(cell_size * 0.35, 0.001);
    for (var tap_index = 0u; tap_index < 4u; tap_index = tap_index + 1u) {
        let descriptor = surface_cache_corner_descriptor(
            center_position,
            center_normal,
            center_tangent_cell + SURFACE_CACHE_FILTER_OFFSETS[tap_index],
            dominant_axis,
            cell_size,
            surface_cache_params
        );
        let neighbor_index_i = surface_cache_find_patch(
            descriptor,
            quantized_normal,
            cell_exponent
        );
        if (neighbor_index_i < 0) {
            continue;
        }

        let neighbor_index = u32(neighbor_index_i);
        let neighbor_patch = surface_cache[neighbor_index];
        if (neighbor_patch.history.x < SURFACE_CACHE_MIN_QUERY_SAMPLES) {
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
        let normalized_plane_distance = plane_distance / plane_scale;
        let plane_weight = exp(
            -0.5 * normalized_plane_distance * normalized_plane_distance
        );
        let sample_confidence = clamp(
            neighbor_patch.history.x / 32.0,
            0.25,
            1.0
        );
        let weight = 0.5
            * plane_weight
            * normal_alignment * normal_alignment
            * sample_confidence;
        if (weight <= 1e-5) {
            continue;
        }

        let neighbor_sh = surface_cache_rotate_sh_between_hemispheres(
            surface_cache_sh_patch_read(&surface_cache_sh, neighbor_index),
            neighbor_normal,
            center_normal
        );
        sh_sum = sh_l1_rgb_add(
            sh_sum,
            sh_l1_rgb_multiply_scalar(neighbor_sh, weight)
        );
        weight_sum += weight;
    }

    let neighborhood_sh = sh_l1_rgb_multiply_scalar(
        sh_sum,
        1.0 / weight_sum
    );
    surface_cache_sh_patch_write(
        &surface_cache_sh_filtered,
        patch_index,
        sh_l1_rgb_lerp(center_sh, neighborhood_sh, filter_strength)
    );
}
