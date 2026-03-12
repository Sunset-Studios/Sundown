#include "common.wgsl"

struct LightingMipParams {
    input_image_size: vec2<f32>,
    output_image_size: vec2<f32>,
}

@group(1) @binding(0) var input_texture: texture_2d<f32>;
@group(1) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var<uniform> params: LightingMipParams;

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= u32(params.output_image_size.x) || global_id.y >= u32(params.output_image_size.y)) {
        return;
    }

    let input_texel_size = 1.0 / params.input_image_size;
    let output_texel_size = 1.0 / params.output_image_size;
    let uv = (vec2<f32>(global_id.xy) + 0.5) * output_texel_size;

    let c00 = textureSampleLevel(input_texture, non_filtering_sampler, uv + vec2<f32>(-0.5 * input_texel_size.x, -0.5 * input_texel_size.y), 0.0).rgb;
    let c10 = textureSampleLevel(input_texture, non_filtering_sampler, uv + vec2<f32>( 0.5 * input_texel_size.x, -0.5 * input_texel_size.y), 0.0).rgb;
    let c01 = textureSampleLevel(input_texture, non_filtering_sampler, uv + vec2<f32>(-0.5 * input_texel_size.x,  0.5 * input_texel_size.y), 0.0).rgb;
    let c11 = textureSampleLevel(input_texture, non_filtering_sampler, uv + vec2<f32>( 0.5 * input_texel_size.x,  0.5 * input_texel_size.y), 0.0).rgb;

    let color = 0.25 * (c00 + c10 + c01 + c11);
    textureStore(output_texture, global_id.xy, vec4<f32>(color, 1.0));
}
