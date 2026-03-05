#include "common.wgsl"

@group(1) @binding(0) var temporal_texture: texture_2d<f32>;
@group(1) @binding(1) var curr_normal_texture: texture_2d<f32>;
@group(1) @binding(2) var curr_position_texture: texture_2d<f32>;
@group(1) @binding(3) var smra_texture: texture_2d<f32>;
@group(1) @binding(4) var out_reflections: texture_storage_2d<rgba16float, write>;

fn luma(v: vec3f) -> f32 {
    return dot(v, vec3f(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let resolution = vec2<u32>(u32(frame_info.resolution.x), u32(frame_info.resolution.y));
    if (gid.x >= resolution.x || gid.y >= resolution.y) {
        return;
    }

    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let center = textureLoad(temporal_texture, coord, 0);
    let center_normal = safe_normalize(textureLoad(curr_normal_texture, coord, 0).xyz);

    if (length(center_normal) < 1e-5) {
        textureStore(out_reflections, coord, vec4f(0.0));
        return;
    }

    let center_pos = textureLoad(curr_position_texture, coord, 0).xyz;
    let roughness = clamp(textureLoad(smra_texture, coord, 0).g, 0.0, 1.0);

    var samples = array<vec4f, 9>();
    var idx = 0u;

    let normal_threshold = mix(0.98, 0.7, roughness);
    let position_threshold = mix(0.05, 0.4, roughness);

    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let tap = vec2<i32>(
                clamp(coord.x + x, 0, i32(resolution.x) - 1),
                clamp(coord.y + y, 0, i32(resolution.y) - 1)
            );

            let tap_sample = textureLoad(temporal_texture, tap, 0);
            let tap_normal = safe_normalize(textureLoad(curr_normal_texture, tap, 0).xyz);
            let tap_pos = textureLoad(curr_position_texture, tap, 0).xyz;

            let geom_match =
                dot(center_normal, tap_normal) > normal_threshold &&
                distance(center_pos, tap_pos) < position_threshold;

            samples[idx] = select(center, tap_sample, geom_match);
            idx += 1u;
        }
    }

    for (var i = 0u; i < 9u; i++) {
        for (var j = i + 1u; j < 9u; j++) {
            if (luma(samples[j].rgb) < luma(samples[i].rgb)) {
                let tmp = samples[i];
                samples[i] = samples[j];
                samples[j] = tmp;
            }
        }
    }

    let median = samples[4];
    let blur_strength = mix(0.1, 0.65, roughness);
    let out_color = mix(center.rgb, median.rgb, blur_strength);
    let out_alpha = max(center.a, median.a);

    textureStore(out_reflections, coord, vec4f(out_color, out_alpha));
}

