#include "common.wgsl"

@group(1) @binding(0) var resolve_texture: texture_2d<f32>;
@group(1) @binding(1) var history_texture_prev: texture_2d<f32>;
@group(1) @binding(2) var motion_texture: texture_2d<f32>;
@group(1) @binding(3) var smra_texture: texture_2d<f32>;
@group(1) @binding(4) var out_temporal: texture_storage_2d<rgba16float, write>;

const FLT_EPS = 1e-6;
const TEMPORAL_RESPONSE_MIN = 0.1;
const TEMPORAL_RESPONSE_MAX = 0.25;
// Expand resolve AABB so we reject ghosting but don't over-clamp and kill accumulation.
const AABB_EXPAND_FACTOR = 0.1;
const AABB_EXPAND_MIN = 0.02;

fn clip_aabb(aabb_min: vec3<f32>, aabb_max: vec3<f32>, p: vec4<f32>, q: vec4<f32>) -> vec4<f32> {
    let p_clip = 0.5 * (aabb_max + aabb_min);
    let e_clip = 0.5 * (aabb_max - aabb_min) + vec3<f32>(FLT_EPS);

    let v_clip = q - vec4<f32>(p_clip, p.w);
    let v_unit = v_clip.xyz / e_clip;
    let a_unit = abs(v_unit);
    let ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));

    return select(q, vec4<f32>(p_clip, p.w) + v_clip / ma_unit, ma_unit > 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let resolution = vec2<u32>(u32(frame_info.resolution.x), u32(frame_info.resolution.y));
    if (gid.x >= resolution.x || gid.y >= resolution.y) {
        return;
    }

    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) / vec2<f32>(f32(resolution.x), f32(resolution.y));

    let current = textureLoad(resolve_texture, coord, 0);
    let roughness = clamp(textureLoad(smra_texture, coord, 0).g, 0.0, 1.0);
    let current_confidence = clamp(current.a, 0.0, 1.0);

    let motion = textureLoad(motion_texture, coord, 0).xy;
    let prev_uv = uv + vec2<f32>(-0.5 * motion.x, 0.5 * motion.y);
    let prev_uv_in_bounds = all(prev_uv >= vec2<f32>(0.0)) && all(prev_uv <= vec2<f32>(1.0));

    let prev_coord = uv_to_coord(prev_uv, resolution);
    let history = textureLoad(history_texture_prev, prev_coord, 0);
    let history_confidence = select(0.0, clamp(history.a, 0.0, 1.0), prev_uv_in_bounds);

    var neigh_min = current.rgb;
    var neigh_max = current.rgb;
    var neigh_avg = vec3<f32>(0.0);

    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let tap = vec2<i32>(
                clamp(coord.x + x, 0, i32(resolution.x) - 1),
                clamp(coord.y + y, 0, i32(resolution.y) - 1)
            );
            let tap_color = textureLoad(resolve_texture, tap, 0).rgb;
            neigh_min = min(neigh_min, tap_color);
            neigh_max = max(neigh_max, tap_color);
            neigh_avg += tap_color;
        }
    }
    neigh_avg *= (1.0 / 9.0);

    // Widen resolve AABB so clipping rejects ghosting but doesn't crush valid history (noise).
    let extent = neigh_max - neigh_min;
    let margin = max(extent * AABB_EXPAND_FACTOR, vec3<f32>(AABB_EXPAND_MIN));
    let clip_min = neigh_min - margin;
    let clip_max = neigh_max + margin;

    let clipped_history = clip_aabb(
        clip_min,
        clip_max,
        vec4<f32>(clamp(neigh_avg, neigh_min, neigh_max), current_confidence),
        history
    );
    let history_sample = select(current, clipped_history, prev_uv_in_bounds);

    let lum_current = luminance(current.rgb);
    let lum_history = luminance(history_sample.rgb);
    let unbiased_diff = abs(lum_current - lum_history) / max(lum_current, max(lum_history, 0.2));
    let stability = (1.0 - unbiased_diff) * (1.0 - unbiased_diff);
    let roughness_response = mix(0.85, 1.15, roughness);
    var feedback = mix(
        TEMPORAL_RESPONSE_MIN,
        TEMPORAL_RESPONSE_MAX,
        stability * current_confidence * roughness_response
    );
    feedback = select(1.0, clamp(feedback, TEMPORAL_RESPONSE_MIN, TEMPORAL_RESPONSE_MAX), prev_uv_in_bounds);

    let temporal_rgb = mix(history_sample.rgb, current.rgb, feedback);
    let temporal_confidence = max(current_confidence, history_confidence * (1.0 - feedback));

    textureStore(out_temporal, coord, vec4<f32>(temporal_rgb, temporal_confidence));
}
