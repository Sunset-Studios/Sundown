#include "common.wgsl"

@group(1) @binding(0) var temporal_texture: texture_2d<f32>;
@group(1) @binding(1) var smra_texture: texture_2d<f32>;
@group(1) @binding(2) var out_reflections: texture_storage_2d<rgba16float, write>;

// Optimal 9-element sorting network (25 compare-swaps instead of 36+ from selection sort).
fn swap_if_less(s: ptr<function, array<vec4<f32>, 9>>, a: u32, b: u32) {
    let la = luminance((*s)[a].rgb);
    let lb = luminance((*s)[b].rgb);
    if (lb < la) {
        let tmp = (*s)[a];
        (*s)[a] = (*s)[b];
        (*s)[b] = tmp;
    }
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let resolution = vec2<u32>(u32(frame_info.resolution.x), u32(frame_info.resolution.y));
    if (gid.x >= resolution.x || gid.y >= resolution.y) {
        return;
    }

    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let center = textureLoad(temporal_texture, coord, 0);
    let roughness = clamp(textureLoad(smra_texture, coord, 0).g, 0.0, 1.0);

    var samples = array<vec4<f32>, 9>();
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

    swap_if_less(&samples, 0u, 1u);
    swap_if_less(&samples, 2u, 3u);
    swap_if_less(&samples, 4u, 5u);
    swap_if_less(&samples, 6u, 7u);
    swap_if_less(&samples, 1u, 3u);
    swap_if_less(&samples, 0u, 2u);
    swap_if_less(&samples, 5u, 7u);
    swap_if_less(&samples, 4u, 6u);
    swap_if_less(&samples, 1u, 2u);
    swap_if_less(&samples, 5u, 6u);
    swap_if_less(&samples, 0u, 4u);
    swap_if_less(&samples, 3u, 7u);
    swap_if_less(&samples, 1u, 5u);
    swap_if_less(&samples, 2u, 6u);
    swap_if_less(&samples, 3u, 5u);
    swap_if_less(&samples, 2u, 3u);
    swap_if_less(&samples, 4u, 5u);
    swap_if_less(&samples, 3u, 4u);
    swap_if_less(&samples, 6u, 8u);
    swap_if_less(&samples, 5u, 8u);
    swap_if_less(&samples, 4u, 8u);
    swap_if_less(&samples, 3u, 8u);
    swap_if_less(&samples, 2u, 8u);
    swap_if_less(&samples, 1u, 8u);
    swap_if_less(&samples, 0u, 8u);

    let median = samples[4];
    let blur_strength = mix(0.1, 0.95, roughness);
    let out_sample = mix(center, median, blur_strength);

    textureStore(out_reflections, coord, out_sample);
}

