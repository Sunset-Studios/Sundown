#include "common.wgsl"

@group(1) @binding(0) var temporal_texture: texture_2d<f32>;
@group(1) @binding(1) var smra_texture: texture_2d<f32>;
@group(1) @binding(2) var out_reflections: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let resolution = vec2<u32>(u32(frame_info.resolution.x), u32(frame_info.resolution.y));
    if (gid.x >= resolution.x || gid.y >= resolution.y) {
        return;
    }

    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let center = textureLoad(temporal_texture, coord, 0);
    let roughness = clamp(textureLoad(smra_texture, coord, 0).g, 0.0, 1.0);

    var samples = array<vec4f, 9>();
    var idx = 0u;

    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let tap = vec2<i32>(
                clamp(coord.x + x, 0, i32(resolution.x) - 1),
                clamp(coord.y + y, 0, i32(resolution.y) - 1)
            );

            samples[idx] = textureLoad(temporal_texture, tap, 0);
            idx += 1u;
        }
    }

    for (var i = 0u; i < 9u; i++) {
        for (var j = i + 1u; j < 9u; j++) {
            if (luminance(samples[j].rgb) < luminance(samples[i].rgb)) {
                let tmp = samples[i];
                samples[i] = samples[j];
                samples[j] = tmp;
            }
        }
    }

    let median = samples[4];
    let blur_strength = mix(0.1, 0.65, roughness);
    let out_sample = mix(center, median, blur_strength);

    textureStore(out_reflections, coord, out_sample);
}

