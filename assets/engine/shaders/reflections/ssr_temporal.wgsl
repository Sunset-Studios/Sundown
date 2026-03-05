#include "common.wgsl"

@group(1) @binding(0) var resolve_texture: texture_2d<f32>;
@group(1) @binding(1) var history_texture_prev: texture_2d<f32>;
@group(1) @binding(2) var raycast_hit_texture: texture_2d<f32>;
@group(1) @binding(3) var raycast_mask_texture: texture_2d<f32>;
@group(1) @binding(4) var curr_normal_texture: texture_2d<f32>;
@group(1) @binding(5) var curr_position_texture: texture_2d<f32>;
@group(1) @binding(6) var prev_normal_texture: texture_2d<f32>;
@group(1) @binding(7) var prev_position_texture: texture_2d<f32>;
@group(1) @binding(8) var motion_texture: texture_2d<f32>;
@group(1) @binding(9) var smra_texture: texture_2d<f32>;
@group(1) @binding(10) var lighting_history_texture: texture_2d<f32>;
@group(1) @binding(11) var out_temporal: texture_storage_2d<rgba16float, write>;
@group(1) @binding(12) var out_history: texture_storage_2d<rgba16float, write>;

const FLT_EPS = 1e-6;
const TEMPORAL_RESPONSE_MIN = 0.85;
const TEMPORAL_RESPONSE_MAX = 1.0;

fn uv_to_coord(uv: vec2f, resolution: vec2<u32>) -> vec2<i32> {
    let max_coord = vec2f(f32(max(1u, resolution.x) - 1u), f32(max(1u, resolution.y) - 1u));
    return vec2<i32>(clamp(uv * max_coord, vec2f(0.0), max_coord));
}

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

    let lum_current = luminance(current.rgb);
    let lum_history = luminance(clipped_history.rgb);
    let unbiased_diff = abs(lum_current - lum_history) / max(lum_current, max(lum_history, 0.2));
    let unbiased_weight = 1.0 - unbiased_diff;
    let k_feedback = mix(TEMPORAL_RESPONSE_MIN, TEMPORAL_RESPONSE_MAX, unbiased_weight * unbiased_weight);

    let temporal = mix(current, clipped_history, k_feedback);

    let ray_mask = textureLoad(raycast_mask_texture, coord, 0).r;
    let ray_hit = textureLoad(raycast_hit_texture, coord, 0);
    let hit_valid = ray_mask > 1e-4 && all(ray_hit.xy >= vec2f(0.0)) && all(ray_hit.xy <= vec2f(1.0));

    let confidence = max(current.a, max(history.a * 0.98, select(0.0, ray_mask, hit_valid)));

    let fallback_color = textureSampleLevel(lighting_history_texture, clamped_sampler, prev_uv, roughness * 4.0).rgb * 0.2;
    let final_color = mix(fallback_color, temporal.rgb, clamp(confidence * 1.5, 0.0, 1.0));

    let out_sample = vec4f(final_color, clamp(confidence, 0.0, 1.0));
    textureStore(out_temporal, coord, out_sample);
    textureStore(out_history, coord, out_sample);
}

