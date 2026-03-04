#include "common.wgsl"

@group(1) @binding(0) var trace_texture: texture_2d<f32>;
@group(1) @binding(1) var history_texture: texture_2d<f32>;
@group(1) @binding(2) var curr_normal_texture: texture_2d<f32>;
@group(1) @binding(3) var curr_position_texture: texture_2d<f32>;
@group(1) @binding(4) var prev_normal_texture: texture_2d<f32>;
@group(1) @binding(5) var prev_position_texture: texture_2d<f32>;
@group(1) @binding(6) var motion_texture: texture_2d<f32>;
@group(1) @binding(7) var smra_texture: texture_2d<f32>;
@group(1) @binding(8) var lighting_history_texture: texture_2d<f32>;
@group(1) @binding(9) var out_reflections: texture_storage_2d<rgba16float, write>;
@group(1) @binding(10) var out_history: texture_storage_2d<rgba16float, write>;

fn uv_to_coord(uv: vec2f, resolution: vec2<u32>) -> vec2<i32> {
    let max_coord = vec2f(f32(max(1u, resolution.x) - 1u), f32(max(1u, resolution.y) - 1u));
    return vec2<i32>(clamp(uv * max_coord, vec2f(0.0), max_coord));
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let resolution = vec2<u32>(u32(frame_info.resolution.x), u32(frame_info.resolution.y));
    if (gid.x >= resolution.x || gid.y >= resolution.y) {
        return;
    }

    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let trace = textureLoad(trace_texture, coord, 0);
    let curr_normal = safe_normalize(textureLoad(curr_normal_texture, coord, 0).xyz);
    let curr_position = textureLoad(curr_position_texture, coord, 0).xyz;
    let roughness = clamp(textureLoad(smra_texture, coord, 0).g, 0.0, 1.0);

    let uv = vec2f(f32(gid.x) / max(1.0, f32(resolution.x - 1u)), f32(gid.y) / max(1.0, f32(resolution.y - 1u)));
    let motion = textureLoad(motion_texture, coord, 0).xy;
    let prev_uv = uv + vec2f(-0.5 * motion.x, 0.5 * motion.y);
    let prev_uv_in_bounds = all(prev_uv >= vec2f(0.0)) && all(prev_uv <= vec2f(1.0));

    var history_color = vec3f(0.0);
    var history_conf = 0.0;
    var history_valid = false;
    if (prev_uv_in_bounds) {
        let prev_coord = uv_to_coord(prev_uv, resolution);
        let prev_sample = textureLoad(history_texture, prev_coord, 0);
        let prev_normal = safe_normalize(textureLoad(prev_normal_texture, prev_coord, 0).xyz);
        let prev_pos = textureLoad(prev_position_texture, prev_coord, 0).xyz;
        history_valid = distance(curr_position, prev_pos) < (0.06 + roughness * 0.25) && dot(curr_normal, prev_normal) > (0.65 - roughness * 0.3);
        history_color = prev_sample.rgb;
        history_conf = prev_sample.a;
    }

    let center = trace.rgb;
    var neigh_min = center;
    var neigh_max = center;
    let radius = i32(1 + i32(round(roughness * 3.0)));
    for (var y = -radius; y <= radius; y++) {
        for (var x = -radius; x <= radius; x++) {
            let tap = vec2<i32>(clamp(i32(gid.x) + x, 0, i32(resolution.x) - 1), clamp(i32(gid.y) + y, 0, i32(resolution.y) - 1));
            let tap_trace = textureLoad(trace_texture, tap, 0).rgb;
            let tap_normal = safe_normalize(textureLoad(curr_normal_texture, tap, 0).xyz);
            let tap_pos = textureLoad(curr_position_texture, tap, 0).xyz;
            let edge_weight = pow(max(dot(curr_normal, tap_normal), 0.0), 8.0) * exp(-distance(curr_position, tap_pos) * 18.0);
            let weighted = mix(center, tap_trace, edge_weight);
            neigh_min = min(neigh_min, weighted);
            neigh_max = max(neigh_max, weighted);
        }
    }

    let clamped_history = clamp(history_color, neigh_min, neigh_max);
    let temporal_alpha = mix(0.25, 0.08, roughness) * (1.0 - history_conf * 0.6);
    let blended = select(trace.rgb, mix(clamped_history, trace.rgb, temporal_alpha), history_valid);

    let fallback_uv = select(uv, prev_uv, prev_uv_in_bounds);
    let fallback_color = textureSampleLevel(lighting_history_texture, clamped_sampler, fallback_uv, roughness * 4.0).rgb * 0.25;
    let confidence = max(trace.a, select(0.0, history_conf * 0.95, history_valid));
    let final_color = mix(fallback_color, blended, clamp(confidence * 1.5, 0.0, 1.0));

    let out_sample = vec4f(final_color, clamp(confidence, 0.0, 1.0));
    textureStore(out_reflections, coord, out_sample);
    textureStore(out_history, coord, out_sample);
}
