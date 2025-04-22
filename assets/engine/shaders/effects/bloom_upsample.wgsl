#include "common.wgsl"

struct BloomBlurConstants {
    input_texture_size: vec2<f32>,
    output_texture_size: vec2<f32>,
    bloom_filter_radius: f32,
    mip_index: u32,
}

@group(1) @binding(0) var input_texture: texture_2d<f32>;
@group(1) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var<uniform> bloom_blur_constants: BloomBlurConstants;

// Adapted from https://learnopengl.com/Guest-Articles/2022/Phys.-Based-Bloom
// This shader performs upsampling on a texture,
// as taken from Call Of Duty method, presented at ACM Siggraph 2014.
// This particular method was customly designed to eliminate
// "pulsating artifacts and temporal stability issues".

@compute @workgroup_size(16, 16, 1)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let target_texel_size = 1.0 / bloom_blur_constants.output_texture_size;

    let tex_coord = (vec2f(global_id.xy) + vec2f(0.5)) * target_texel_size;

    if (global_id.x < u32(bloom_blur_constants.output_texture_size.x) &&
        global_id.y < u32(bloom_blur_constants.output_texture_size.y)) {
        // The filter kernel is applied with a radius, specified in texture
        // coordinates, so that the radius will vary across mip resolutions.
        let x = bloom_blur_constants.bloom_filter_radius;
        let y = bloom_blur_constants.bloom_filter_radius;

        // Take 9 samples around current texel:
        // a - b - c
        // d - e - f
        // g - h - i
        // === ('e' is the current texel) ===
        let a = textureSampleLevel(input_texture, clamped_sampler, vec2f(tex_coord.x - x, tex_coord.y + y), 0).rgb;
        let b = textureSampleLevel(input_texture, clamped_sampler, vec2f(tex_coord.x,     tex_coord.y + y), 0).rgb;
        let c = textureSampleLevel(input_texture, clamped_sampler, vec2f(tex_coord.x + x, tex_coord.y + y), 0).rgb;

        let d = textureSampleLevel(input_texture, clamped_sampler, vec2f(tex_coord.x - x, tex_coord.y), 0).rgb;
        let e = textureSampleLevel(input_texture, clamped_sampler, vec2f(tex_coord.x,     tex_coord.y), 0).rgb;
        let f = textureSampleLevel(input_texture, clamped_sampler, vec2f(tex_coord.x + x, tex_coord.y), 0).rgb;

        let g = textureSampleLevel(input_texture, clamped_sampler, vec2f(tex_coord.x - x, tex_coord.y - y), 0).rgb;
        let h = textureSampleLevel(input_texture, clamped_sampler, vec2f(tex_coord.x,     tex_coord.y - y), 0).rgb;
        let i = textureSampleLevel(input_texture, clamped_sampler, vec2f(tex_coord.x + x, tex_coord.y - y), 0).rgb;

        // Apply weighted distribution, by using a 3x3 tent filter:
        //  1   | 1 2 1 |
        // -- * | 2 4 2 |
        // 16   | 1 2 1 |
        var upsample_color = e * 4.0;
        upsample_color += (b + d + f + h) * 2.0;
        upsample_color += (a + c + g + i);
        upsample_color *= 1.0 / 16.0;

        textureStore(output_texture, vec2i(global_id.xy), vec4f(upsample_color, 1.0));
    }
}