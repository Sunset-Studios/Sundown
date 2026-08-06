// =============================================================================
// DDGI Depth Moment Update
// Runs after SH accumulate and updates octahedral depth moments.
// One invocation owns one packed word (two directional depth texels), gathers
// all rays that map to either texel, and performs one race-free u32 write.
// =============================================================================
#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read> probe_ray_data: DDGIProbeRayDataBufferReadOnlyHeader;
@group(1) @binding(2) var<storage, read> probe_update_indices: array<u32>;
@group(1) @binding(3) var<storage, read> probe_history_valid: array<f32>;
@group(1) @binding(4) var<storage, read_write> probe_depth_moments: array<u32>;
@group(1) @binding(5) var<storage, read> probe_depth_slots: array<u32>;

const DDGI_VISIBILITY_HYSTERESIS = 0.985;
const DDGI_LARGE_GEOMETRY_CHANGE_SPACING_FRACTION = 0.5;
const DDGI_LARGE_GEOMETRY_CHANGE_DISTANCE_FRACTION = 0.25;
const DDGI_LARGE_GEOMETRY_CHANGE_HYSTERESIS_SCALE = 0.5;

fn ddgi_depth_texel_for_direction(ray_dir: vec3<f32>, depth_res: u32) -> u32 {
    let ray_dir_n = safe_normalize(ray_dir);
    let uv = encode_octahedral_normalized(ray_dir_n) * f32(depth_res);
    let max_coord = i32(depth_res) - 1;
    let texel_coord = vec2<u32>(clamp(vec2<i32>(uv), vec2<i32>(0), vec2<i32>(max_coord)));
    return ddgi_depth_texel_id(texel_coord, depth_res);
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let probe_slot = gid.x;
    let word_idx = gid.y;
    let rays_per_probe = ddgi_max_rays_per_probe(&ddgi_params);
    let active_probe_count = probe_ray_data.header.active_ray_count / rays_per_probe;

    if (probe_slot >= active_probe_count) {
        return;
    }

    let ray_base = probe_slot * rays_per_probe;
    let probe_index = probe_update_indices[probe_slot];

    let depth_slot = ddgi_depth_slot_for_probe(&probe_depth_slots, probe_index);
    if (depth_slot == INVALID_IDX) {
        return;
    }

    let depth_base = ddgi_depth_base_for_slot(&ddgi_params, depth_slot);
    let depth_res = ddgi_depth_resolution_for_probe(&ddgi_params, probe_index);
    let depth_texel_count = depth_res * depth_res;
    let depth_word_count = (depth_texel_count + 1u) / 2u;
    if (word_idx >= depth_word_count) {
        return;
    }

    let spacing = ddgi_probe_spacing_from_index(&ddgi_params, probe_index);
    let miss_distance = ddgi_probe_miss_distance(&ddgi_params, probe_index);
    let has_local_history = probe_history_valid[probe_index] > 0.0;
    var packed_word = probe_depth_moments[depth_base + word_idx];

    for (var lane = 0u; lane < 2u; lane = lane + 1u) {
        let texel_idx = word_idx * 2u + lane;
        if (texel_idx >= depth_texel_count) {
            continue;
        }

        var t_sum = 0.0;
        var t2_sum = 0.0;
        var sample_count = 0u;
        for (var ray_i = 0u; ray_i < rays_per_probe; ray_i = ray_i + 1u) {
            let ray_index = ray_base + ray_i;
            let hit = probe_ray_data.rays[ray_index];
            if (ddgi_depth_texel_for_direction(ddgi_probe_ray_stored_direction(hit), depth_res) != texel_idx) {
                continue;
            }

            let t_raw = hit.hit_distance;
            let is_valid_hit = hit.prim_store != INVALID_IDX;
            let t_sample = min(select(miss_distance, abs(t_raw), is_valid_hit), miss_distance);
            t_sum += t_sample;
            t2_sum += t_sample * t_sample;
            sample_count = sample_count + 1u;
        }

        if (sample_count == 0u) {
            continue;
        }

        let inv_sample_count = 1.0 / f32(sample_count);
        let t = t_sum * inv_sample_count;
        let t2 = t2_sum * inv_sample_count;
        let packed_prev = ddgi_depth_word_texel(packed_word, texel_idx);
        let has_moment_history =
            has_local_history && packed_prev != DDGI_DEPTH_MOMENTS_INVALID_TEXEL;
        let prev = ddgi_depth_moments_unpack(
            packed_prev,
            spacing,
            miss_distance
        );

        // Lighting changes never enter this classification. Only a meaningful
        // local directional depth shift temporarily reduces hysteresis.
        let geometry_change_threshold = max(
            spacing * DDGI_LARGE_GEOMETRY_CHANGE_SPACING_FRACTION,
            max(prev.x, spacing) * DDGI_LARGE_GEOMETRY_CHANGE_DISTANCE_FRACTION
        );
        let large_geometry_change =
            has_moment_history && abs(t - prev.x) > geometry_change_threshold;
        var visibility_hysteresis =
            select(0.0, DDGI_VISIBILITY_HYSTERESIS, has_moment_history);
        if (large_geometry_change) {
            visibility_hysteresis *= DDGI_LARGE_GEOMETRY_CHANGE_HYSTERESIS_SCALE;
        }

        let result = mix(vec2<f32>(t, t2), prev, visibility_hysteresis);
        let packed_texel = ddgi_depth_moments_pack(
            result.x,
            result.y,
            spacing,
            miss_distance
        );
        packed_word = ddgi_depth_word_replace_texel(packed_word, texel_idx, packed_texel);
    }

    probe_depth_moments[depth_base + word_idx] = packed_word;
}
