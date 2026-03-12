// =============================================================================
// RTAO TEMPORAL PASS
// =============================================================================
//
// Reuses and reprojects the previous frame's AO history:
//   - Reproject history using screen-space motion vectors (NDC velocity)
//   - Neighborhood clamp to reduce ghosting (clamp history to 3x3 current min/max)
//   - Exponential moving average: blend current with reprojected history
//
// =============================================================================

#include "common.wgsl"

@group(1) @binding(0) var ao_current: texture_2d<f32>;
@group(1) @binding(1) var ao_history: texture_2d<f32>;
@group(1) @binding(2) var motion_texture: texture_2d<f32>;
@group(1) @binding(3) var ao_output: texture_storage_2d<r32float, write>;

// Blend factor: higher = more current frame (faster response), lower = more history (smoother)
const RTAO_TEMPORAL_BLEND = 0.1;

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let resolution = vec2<u32>(u32(frame_info.resolution.x), u32(frame_info.resolution.y));
    if (gid.x >= resolution.x || gid.y >= resolution.y) {
        return;
    }

    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) / vec2<f32>(f32(resolution.x), f32(resolution.y));

    let current_ao = textureLoad(ao_current, coord, 0).r;
    let motion = textureLoad(motion_texture, coord, 0).xy;
    let prev_uv = uv + vec2<f32>(-0.5 * motion.x, 0.5 * motion.y);
    let prev_uv_in_bounds = all(prev_uv >= vec2<f32>(0.0)) && all(prev_uv <= vec2<f32>(1.0));

    var neigh_min = current_ao;
    var neigh_max = current_ao;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let tap = vec2<i32>(
                clamp(coord.x + dx, 0, i32(resolution.x) - 1),
                clamp(coord.y + dy, 0, i32(resolution.y) - 1)
            );
            let tap_ao = textureLoad(ao_current, tap, 0).r;
            neigh_min = min(neigh_min, tap_ao);
            neigh_max = max(neigh_max, tap_ao);
        }
    }

    let prev_coord = uv_to_coord(prev_uv, resolution);
    let history_ao = textureLoad(ao_history, prev_coord, 0).r;
    let history_sample = select(current_ao, history_ao, prev_uv_in_bounds);
    let history_clamped = clamp(history_sample, neigh_min, neigh_max);

    let blend_alpha = select(1.0, RTAO_TEMPORAL_BLEND, prev_uv_in_bounds);
    let ao_out = mix(history_clamped, current_ao, blend_alpha);

    textureStore(ao_output, coord, vec4<f32>(ao_out, 0.0, 0.0, 1.0));
}
