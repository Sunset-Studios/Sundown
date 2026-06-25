// =============================================================================
// DDGI Depth Moment Update — 1 thread per ray
// Runs after SH accumulate and updates octahedral depth moments.
// Better GPU occupancy than per-probe loop (many more threads, coalesced reads).
// =============================================================================
#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read> probe_ray_data: DDGIProbeRayDataBufferReadOnlyHeader;
@group(1) @binding(2) var<storage, read> probe_history_valid: array<f32>;
@group(1) @binding(3) var<storage, read_write> probe_depth_moments: array<u32>;

fn depth_moment_update_at(
    moment_idx: u32,
    t: f32,
    t2: f32,
    visibility_hysteresis: f32
) {
    let prev = ddgi_depth_moments_unpack(probe_depth_moments[moment_idx]);
    let next = vec2<f32>(t, t2);
    let result = mix(next, prev, visibility_hysteresis);
    probe_depth_moments[moment_idx] = ddgi_depth_moments_pack(
        result.x,
        result.y
    );
}

const DDGI_VISIBILITY_HYSTERESIS = 0.985;
const DDGI_LARGE_GEOMETRY_CHANGE_SPACING_FRACTION = 0.5;
const DDGI_LARGE_GEOMETRY_CHANGE_DISTANCE_FRACTION = 0.25;
const DDGI_LARGE_GEOMETRY_CHANGE_HYSTERESIS_SCALE = 0.5;

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let active_ray_count = probe_ray_data.header.active_ray_count;
    if (gid.x >= active_ray_count) {
        return;
    }

    let ray_index = gid.x;
    let hit = probe_ray_data.rays[ray_index];
    let probe_index = hit.meta_u32.x;

    let depth_base = ddgi_depth_base_for_probe(&ddgi_params, probe_index);
    let depth_res = ddgi_depth_resolution_for_probe(&ddgi_params, probe_index);
    let depth_res_f = f32(depth_res);
    let spacing = ddgi_probe_spacing_from_index(&ddgi_params, probe_index);
    let max_dim = max(
        ddgi_params.probe_grid_dims.x,
        max(ddgi_params.probe_grid_dims.y, ddgi_params.probe_grid_dims.z)
    );
    let miss_distance = max(1.0, spacing * max_dim * 2.0);

    let ray_dir_n = safe_normalize(hit.ray_dir_prim.xyz);
    let uv = encode_octahedral_normalized(ray_dir_n) * depth_res_f;
    let max_coord = i32(depth_res) - 1;
    let texel_coord = vec2<u32>(clamp(vec2<i32>(uv), vec2<i32>(0), vec2<i32>(max_coord)));
    let texel_idx = ddgi_depth_texel_id(texel_coord, depth_res);

    let t_raw = hit.hit_pos_t.w;
    let is_valid_hit = hit.state_u32.w != INVALID_IDX;
    let t = min(select(miss_distance, abs(t_raw), is_valid_hit), miss_distance);
    let t2 = t * t;

    let moment_idx = depth_base + texel_idx;
    let prev = ddgi_depth_moments_unpack(probe_depth_moments[moment_idx]);
    let has_local_history = probe_history_valid[probe_index] > 0.0;

    // Lighting changes never enter this classification. Only a meaningful local
    // directional depth shift (a door or moving wall) temporarily reduces the
    // otherwise high visibility hysteresis. As the texel converges, the test
    // naturally returns to the stable baseline after a few updates.
    let geometry_change_threshold = max(
        spacing * DDGI_LARGE_GEOMETRY_CHANGE_SPACING_FRACTION,
        max(prev.x, spacing) * DDGI_LARGE_GEOMETRY_CHANGE_DISTANCE_FRACTION
    );
    let large_geometry_change = has_local_history && abs(t - prev.x) > geometry_change_threshold;
    var visibility_hysteresis = select(0.0, DDGI_VISIBILITY_HYSTERESIS, has_local_history);
    if (large_geometry_change) {
        visibility_hysteresis *= DDGI_LARGE_GEOMETRY_CHANGE_HYSTERESIS_SCALE;
    }

    // Stable baseline: lerp(new visibility, old visibility, hysteresis).
    depth_moment_update_at(moment_idx, t, t2, visibility_hysteresis);
}
