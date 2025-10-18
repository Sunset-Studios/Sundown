// =============================================================================
// Path Trace Output Pass
// - Writes accumulated results for ALL pixels every frame
// - This ensures coherent output even when trace_rate > 1
// - Prevents checkerboard flickering from partial updates
// =============================================================================
#include "common.wgsl"

struct PathShade {
    path_weight: vec4<f32>,
    rng_sample_count_frame_stamp: vec4<f32>,
    throughput: vec4<f32>,
    reservoir_radiance_m: vec4<f32>,
    reservoir_direction_w: vec4<f32>,
}

@group(1) @binding(0) var<storage, read> path_shade: array<PathShade>;
@group(1) @binding(1) var output_tex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(output_tex);
    
    if (gid.x >= res.x || gid.y >= res.y) { return; }
    
    let pixel_index = gid.y * res.x + gid.x;
    let shade = path_shade[pixel_index];
    
    // Write current accumulated average for this pixel
    let sample_count = max(shade.rng_sample_count_frame_stamp.y, 1.0);
    let accumulated_avg = shade.throughput.xyz / sample_count;
    let safe_output = safe_clamp_vec3(accumulated_avg);
    
    textureStore(output_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4f(safe_output, 1.0));
}

