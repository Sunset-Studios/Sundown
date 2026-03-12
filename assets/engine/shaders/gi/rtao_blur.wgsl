// =============================================================================
// RTAO DEPTH-AWARE SPATIAL BLUR
// =============================================================================
//
// Final filtering step: subtle 3x3 spatial blur that respects depth edges.
// Uses the depth buffer for depth weighting to avoid bleeding across
// depth discontinuities. Kept cheap (3x3, single pass) and subtle.
//
// =============================================================================

#include "common.wgsl"

@group(1) @binding(0) var ao_input: texture_2d<f32>;
@group(1) @binding(1) var depth_texture: texture_2d<f32>;
@group(1) @binding(2) var ao_output: texture_storage_2d<r32float, write>;

// Depth difference sigma (in depth buffer units; works with reverse-Z)
const DEPTH_SIGMA = 0.002;
// Spatial falloff sigma in pixels (Gaussian radius; ~0.8 gives soft 3x3)
const SPATIAL_SIGMA = 0.8;
// How much to blend blurred result with center (1.0 = full blur, 0.0 = none)
const BLUR_STRENGTH = 1.0;

fn depth_weight(center_depth: f32, sample_depth: f32) -> f32 {
    let d = center_depth - sample_depth;
    let sigma2 = DEPTH_SIGMA * DEPTH_SIGMA * 2.0;
    return exp(-(d * d) / sigma2);
}

fn spatial_weight(dx: i32, dy: i32) -> f32 {
    let d2 = f32(dx * dx + dy * dy);
    let sigma2 = SPATIAL_SIGMA * SPATIAL_SIGMA * 2.0;
    return exp(-d2 / sigma2);
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let resolution = vec2<u32>(u32(frame_info.resolution.x), u32(frame_info.resolution.y));
    if (gid.x >= resolution.x || gid.y >= resolution.y) {
        return;
    }

    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let center_ao = textureLoad(ao_input, coord, 0).r;
    let center_depth = textureLoad(depth_texture, coord, 0).r;

    var w_sum = 1.0;
    var ao_sum = center_ao;

    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let tap = vec2<i32>(
                clamp(coord.x + dx, 0, i32(resolution.x) - 1),
                clamp(coord.y + dy, 0, i32(resolution.y) - 1)
            );
            let tap_depth = textureLoad(depth_texture, tap, 0).r;
            let w_depth = depth_weight(center_depth, tap_depth);
            let w_spatial = spatial_weight(dx, dy);
            let w = w_depth * w_spatial;
            let tap_ao = textureLoad(ao_input, tap, 0).r;
            ao_sum += w * tap_ao;
            w_sum += w;
        }
    }

    let blurred = ao_sum / w_sum;
    let ao_out = mix(center_ao, blurred, BLUR_STRENGTH);

    textureStore(ao_output, coord, vec4<f32>(ao_out, 0.0, 0.0, 1.0));
}
