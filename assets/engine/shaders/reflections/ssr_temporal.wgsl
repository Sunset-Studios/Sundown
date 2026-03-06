#include "common.wgsl"

@group(1) @binding(0) var resolve_texture: texture_2d<f32>;
@group(1) @binding(1) var history_texture_prev: texture_2d<f32>;
@group(1) @binding(2) var motion_texture: texture_2d<f32>;
@group(1) @binding(3) var smra_texture: texture_2d<f32>;
@group(1) @binding(4) var out_temporal: texture_storage_2d<rgba16float, write>;

const FLT_EPS = 1e-6;
const TEMPORAL_RESPONSE_MIN = 0.0;
const TEMPORAL_RESPONSE_MAX = 0.2;

fn clip_aabb(aabb_min: vec3f, aabb_max: vec3f, p: vec4f, q: vec4f) -> vec4f {
    let p_clip = 0.5 * (aabb_max + aabb_min);
    let e_clip = 0.5 * (aabb_max - aabb_min) + vec3f(FLT_EPS);

    let v_clip = q - vec4f(p_clip, p.w);
    let v_unit = v_clip.xyz / e_clip;
    let a_unit = abs(v_unit);
    let ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));

    return select(q, vec4f(p_clip, p.w) + v_clip / ma_unit, ma_unit > 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let resolution = vec2<u32>(u32(frame_info.resolution.x), u32(frame_info.resolution.y));
    if (gid.x >= resolution.x || gid.y >= resolution.y) {
        return;
    }

    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let uv = (vec2f(f32(gid.x), f32(gid.y)) + 0.5) / vec2f(f32(resolution.x), f32(resolution.y));

    let current = textureLoad(resolve_texture, coord, 0);
    let roughness = clamp(textureLoad(smra_texture, coord, 0).g, 0.0, 1.0);

    let motion = textureLoad(motion_texture, coord, 0).xy;
    let prev_uv = uv + vec2f(-0.5 * motion.x, 0.5 * motion.y);
    let prev_uv_in_bounds = all(prev_uv >= vec2f(0.0)) && all(prev_uv <= vec2f(1.0));

    let prev_coord = uv_to_coord(prev_uv, resolution);
    let history = textureLoad(history_texture_prev, prev_coord, 0);

    var neigh_min = current.rgb;
    var neigh_max = current.rgb;
    var neigh_avg = vec3f(0.0);

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

    let clipped_history = clip_aabb(neigh_min, neigh_max, vec4f(clamp(neigh_avg, neigh_min, neigh_max), current.a), history);
    let history_sample = select(current, clipped_history, prev_uv_in_bounds);

    let lum_current = luminance(current.rgb);
    let lum_history = luminance(history_sample.rgb);
    let unbiased_diff = abs(lum_current - lum_history) / max(lum_current, max(lum_history, 0.2));
    let unbiased_weight = 1.0 - unbiased_diff;
    let feedback = mix(TEMPORAL_RESPONSE_MIN, TEMPORAL_RESPONSE_MAX, unbiased_weight * unbiased_weight);

    let temporal = mix(history_sample, current, feedback);

    textureStore(out_temporal, coord, temporal);
}
