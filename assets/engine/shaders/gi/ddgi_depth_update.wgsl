// =============================================================================
// DDGI Depth Moment Update — 1 thread per ray
// Runs after SH accumulate; reads per-probe alpha and updates octahedral depth.
// Better GPU occupancy than per-probe loop (many more threads, coalesced reads).
// =============================================================================
#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read> probe_ray_data: DDGIProbeRayDataBufferReadOnlyHeader;
@group(1) @binding(2) var<storage, read> probe_alpha: array<f32>;
@group(1) @binding(3) var<storage, read_write> probe_depth_moments: array<u32>;

fn depth_moment_update_at(
    moment_idx: u32,
    t: f32,
    t2: f32,
    update_alpha: f32
) {
    let prev = ddgi_depth_moments_unpack(probe_depth_moments[moment_idx]);
    probe_depth_moments[moment_idx] = ddgi_depth_moments_pack(
        prev.x + (t - prev.x) * update_alpha,
        prev.y + (t2 - prev.y) * update_alpha
    );
}

@compute @workgroup_size(256, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let active_ray_count = probe_ray_data.header.active_ray_count;
    if (gid.x >= active_ray_count) {
        return;
    }

    let ray_index = gid.x;
    let hit = probe_ray_data.rays[ray_index];
    let probe_index = hit.meta_u32.x;

    let alpha = probe_alpha[probe_index];
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
    depth_moment_update_at(moment_idx, t, t2, alpha);
}
