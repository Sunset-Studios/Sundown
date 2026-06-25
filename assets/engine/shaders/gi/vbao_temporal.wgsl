// VBAO temporal: reproject history with motion, clamp to 3x3 neighborhood, blend.
// Keeps history clamping to reduce ghosting; no world-space validation.

#include "common.wgsl"

struct VBAOSettings {
    radius: f32,
    strength: f32,
    bias: f32,
    slice_count: f32,
    sample_count: f32,
    thickness: f32,
    temporal_response: f32,
};

@group(1) @binding(0) var current_ao_tex: texture_2d<f32>;
@group(1) @binding(1) var history_ao_tex: texture_2d<f32>;
@group(1) @binding(2) var motion_tex: texture_2d<f32>;
@group(1) @binding(3) var ao_output: texture_storage_2d<r32float, write>;
@group(1) @binding(4) var<uniform> settings: VBAOSettings;

fn sample_history_bilinear(prev_uv: vec2<f32>, dims: vec2<u32>) -> f32 {
    let history_pos = prev_uv * vec2<f32>(f32(dims.x), f32(dims.y)) - vec2<f32>(0.5);
    let base = vec2<i32>(floor(history_pos));
    let frac = fract(history_pos);

    let p00 = vec2<i32>(
        clamp(base.x, 0, i32(dims.x) - 1),
        clamp(base.y, 0, i32(dims.y) - 1)
    );
    let p10 = vec2<i32>(
        clamp(base.x + 1, 0, i32(dims.x) - 1),
        clamp(base.y, 0, i32(dims.y) - 1)
    );
    let p01 = vec2<i32>(
        clamp(base.x, 0, i32(dims.x) - 1),
        clamp(base.y + 1, 0, i32(dims.y) - 1)
    );
    let p11 = vec2<i32>(
        clamp(base.x + 1, 0, i32(dims.x) - 1),
        clamp(base.y + 1, 0, i32(dims.y) - 1)
    );

    let h00 = textureLoad(history_ao_tex, p00, 0).r;
    let h10 = textureLoad(history_ao_tex, p10, 0).r;
    let h01 = textureLoad(history_ao_tex, p01, 0).r;
    let h11 = textureLoad(history_ao_tex, p11, 0).r;

    return mix(mix(h00, h10, frac.x), mix(h01, h11, frac.x), frac.y);
}

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

    // Use relaxed neighborhood clipping. A single-frame VBAO result is intentionally
    // sparse, so strict min/max clipping throws away useful accumulated history.
    var neigh_min = current_ao;
    var neigh_max = current_ao;
    var neigh_sum = 0.0;
    var neigh_sum_sq = 0.0;
    var neigh_count = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let tap = vec2<i32>(
                clamp(coord.x + dx, 0, i32(dims.x) - 1),
                clamp(coord.y + dy, 0, i32(dims.y) - 1)
            );
            let tap_ao = textureLoad(current_ao_tex, tap, 0).r;
            neigh_min = min(neigh_min, tap_ao);
            neigh_max = max(neigh_max, tap_ao);
            neigh_sum += tap_ao;
            neigh_sum_sq += tap_ao * tap_ao;
            neigh_count += 1.0;
        }
    }

    let motion = textureLoad(motion_tex, coord, 0).xy;
    let prev_uv = uv + vec2<f32>(-0.5 * motion.x, 0.5 * motion.y);
    let prev_in_bounds = all(prev_uv >= vec2<f32>(0.0)) && all(prev_uv <= vec2<f32>(1.0));

    let history_ao_raw = sample_history_bilinear(prev_uv, dims);
    let history_ao_sample = select(current_ao, history_ao_raw, prev_in_bounds);
    let neigh_mean = neigh_sum / neigh_count;
    let neigh_variance = max(neigh_sum_sq / neigh_count - neigh_mean * neigh_mean, 0.0);
    let neigh_stddev = sqrt(neigh_variance);
    let range_padding = max(0.05, max(neigh_max - neigh_min, neigh_stddev * 2.0) * 0.5);
    let history_ao_clamped = clamp(
        history_ao_sample,
        max(0.0, neigh_min - range_padding),
        min(1.0, neigh_max + range_padding)
    );

    let blend_alpha = select(1.0, settings.temporal_response, prev_in_bounds);
    let ao_value = mix(history_ao_clamped, current_ao, blend_alpha);

    textureStore(ao_output, coord, vec4<f32>(ao_value, 0.0, 0.0, 1.0));
}
