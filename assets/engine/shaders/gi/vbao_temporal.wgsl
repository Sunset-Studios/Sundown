// VBAO temporal: reproject history with motion, clamp to 3x3 neighborhood, blend.
// Keeps history clamping to reduce ghosting; no world-space validation.

#include "common.wgsl"

struct VBAOSettings {
    radius: f32,
    bias: f32,
    slice_count: f32,
    sample_count: f32,
    max_radius_px: f32,
    thickness: f32,
    temporal_response: f32,
    denoise_radius: f32,
    denoise_position_sigma: f32,
    denoise_normal_power: f32,
    denoise_ao_sigma: f32,
    denoise_direction: vec2<f32>,
    denoise_radius_px: f32,
};

@group(1) @binding(0) var current_ao_tex: texture_2d<f32>;
@group(1) @binding(1) var history_ao_tex: texture_2d<f32>;
@group(1) @binding(2) var motion_tex: texture_2d<f32>;
@group(1) @binding(3) var ao_output: texture_storage_2d<r32float, write>;
@group(1) @binding(4) var<uniform> settings: VBAOSettings;

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(current_ao_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let coord = vec2<i32>(gid.xy);
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) /
        vec2<f32>(f32(dims.x), f32(dims.y));

    let current_ao = textureLoad(current_ao_tex, coord, 0).r;

    // 3x3 neighborhood for history clamping
    var neigh_min = current_ao;
    var neigh_max = current_ao;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let tap = vec2<i32>(
                clamp(coord.x + dx, 0, i32(dims.x) - 1),
                clamp(coord.y + dy, 0, i32(dims.y) - 1)
            );
            let tap_ao = textureLoad(history_ao_tex, tap, 0).r;
            neigh_min = min(neigh_min, tap_ao);
            neigh_max = max(neigh_max, tap_ao);
        }
    }

    let motion = textureLoad(motion_tex, coord, 0).xy;
    let prev_uv = uv + vec2<f32>(-0.5 * motion.x, 0.5 * motion.y);
    let prev_in_bounds = all(prev_uv >= vec2<f32>(0.0)) && all(prev_uv <= vec2<f32>(1.0));

    let prev_coord = uv_to_coord(prev_uv, dims);
    let history_ao_raw = textureLoad(history_ao_tex, prev_coord, 0).r;
    let history_ao_sample = select(current_ao, history_ao_raw, prev_in_bounds);
    let history_ao_clamped = clamp(history_ao_sample, neigh_min, neigh_max);

    let blend_alpha = select(1.0, settings.temporal_response, prev_in_bounds);
    let ao_value = mix(history_ao_clamped, current_ao, blend_alpha);

    textureStore(ao_output, coord, vec4<f32>(ao_value, 0.0, 0.0, 1.0));
}
