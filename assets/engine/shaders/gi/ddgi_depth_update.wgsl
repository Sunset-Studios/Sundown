// =============================================================================
// DDGI Depth Moment Update
// Runs after SH accumulate and updates octahedral depth moments.
// One invocation owns one probe depth texel and gathers all rays that map there,
// avoiding packed-moment write races when multiple rays hit the same texel.
// =============================================================================
#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read> probe_ray_data: DDGIProbeRayDataBufferReadOnlyHeader;
@group(1) @binding(2) var<storage, read> probe_history_valid: array<f32>;
@group(1) @binding(3) var<storage, read_write> probe_depth_moments: array<u32>;

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
    let texel_idx = gid.y;
    let rays_per_probe = ddgi_max_rays_per_probe(&ddgi_params);
    let active_probe_count = probe_ray_data.header.active_ray_count / rays_per_probe;

    if (probe_slot >= active_probe_count) {
        return;
    }

    let ray_base = probe_slot * rays_per_probe;
    let probe_index = probe_ray_data.rays[ray_base].meta_u32.x;

    let depth_base = ddgi_depth_base_for_probe(&ddgi_params, probe_index);
    let depth_res = ddgi_depth_resolution_for_probe(&ddgi_params, probe_index);
    let depth_texel_count = depth_res * depth_res;
    if (texel_idx >= depth_texel_count) {
        return;
    }

    let spacing = ddgi_probe_spacing_from_index(&ddgi_params, probe_index);
    let max_dim = max(
        ddgi_params.probe_grid_dims.x,
        max(ddgi_params.probe_grid_dims.y, ddgi_params.probe_grid_dims.z)
    );
    let miss_distance = max(1.0, spacing * max_dim * 2.0);

    var t_sum = 0.0;
    var t2_sum = 0.0;
    var sample_count = 0u;
    for (var ray_i = 0u; ray_i < rays_per_probe; ray_i = ray_i + 1u) {
        let ray_index = ray_base + ray_i;
        let hit = probe_ray_data.rays[ray_index];
        if (ddgi_depth_texel_for_direction(hit.ray_dir_prim.xyz, depth_res) != texel_idx) {
            continue;
        }

        let t_raw = hit.hit_pos_t.w;
        let is_valid_hit = hit.state_u32.w != INVALID_IDX;
        let t = min(select(miss_distance, abs(t_raw), is_valid_hit), miss_distance);
        t_sum += t;
        t2_sum += t * t;
        sample_count = sample_count + 1u;
    }

    if (sample_count == 0u) {
        return;
    }

    let inv_sample_count = 1.0 / f32(sample_count);
    let t = t_sum * inv_sample_count;
    let t2 = t2_sum * inv_sample_count;

    let moment_idx = depth_base + texel_idx;
    let prev = ddgi_depth_moments_unpack(probe_depth_moments[moment_idx]);
    let has_local_history = probe_history_valid[probe_index] > 0.0;

    // Lighting changes never enter this classification. Only a meaningful local
    // directional depth shift temporarily reduces the otherwise high visibility
    // hysteresis. As the texel converges, the test naturally returns to the
    // stable baseline after a few updates.
    let geometry_change_threshold = max(
        spacing * DDGI_LARGE_GEOMETRY_CHANGE_SPACING_FRACTION,
        max(prev.x, spacing) * DDGI_LARGE_GEOMETRY_CHANGE_DISTANCE_FRACTION
    );
    let large_geometry_change = has_local_history && abs(t - prev.x) > geometry_change_threshold;
    var visibility_hysteresis = select(0.0, DDGI_VISIBILITY_HYSTERESIS, has_local_history);
    if (large_geometry_change) {
        visibility_hysteresis *= DDGI_LARGE_GEOMETRY_CHANGE_HYSTERESIS_SCALE;
    }

    let next = vec2<f32>(t, t2);
    let result = mix(next, prev, visibility_hysteresis);
    probe_depth_moments[moment_idx] = ddgi_depth_moments_pack(result.x, result.y);
}
