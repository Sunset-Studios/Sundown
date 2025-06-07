#include "common.wgsl"

// ------------------------------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------------------------------ 
struct HZBParams {
    input_image_size: vec2<f32>,
    output_image_size: vec2<f32>,
}

// ------------------------------------------------------------------------------------
// Compute Shader
// ------------------------------------------------------------------------------------ 

@group(1) @binding(0) var input_texture: texture_2d<f32>;
@group(1) @binding(1) var output_texture: texture_storage_2d<r32float, write>;
@group(1) @binding(2) var<uniform> params: HZBParams;

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= u32(params.output_image_size.x) || global_id.y >= u32(params.output_image_size.y)) {
        return;
    }

    let input_texel_size = 1.0 / params.input_image_size;
    let output_texel_size = 1.0 / params.output_image_size;
    let base_uv = (vec2f(global_id.xy) + vec2f(0.5)) * output_texel_size;

    var d00 = textureSampleLevel(input_texture, non_filtering_sampler, base_uv, 0).r;
    var d10 = textureSampleLevel(input_texture, non_filtering_sampler, base_uv + vec2f(input_texel_size.x, 0.0), 0).r;
    var d01 = textureSampleLevel(input_texture, non_filtering_sampler, base_uv + vec2f(0.0, input_texel_size.y), 0).r;
    var d11 = textureSampleLevel(input_texture, non_filtering_sampler, base_uv + vec2f(input_texel_size.x, input_texel_size.y), 0).r;

    let min_depth = min(d00, min(d10, min(d01, d11)));

    textureStore(output_texture, global_id.xy, vec4<f32>(min_depth));
}