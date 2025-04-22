#include "common.wgsl"

struct BloomBlurConstants {
    input_texture_size: vec2<f32>,
    output_texture_size: vec2<f32>,
    blur_radius: f32,
    mip_index: u32,
}

@group(1) @binding(0) var input_texture: texture_2d<f32>;
@group(1) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var<uniform> bloom_blur_constants: BloomBlurConstants;

// Adapted from https://learnopengl.com/Guest-Articles/2022/Phys.-Based-Bloom
// This shader performs downsampling on a texture,
// as taken from Call Of Duty method, presented at ACM Siggraph 2014.
// This particular method was customly designed to eliminate
// "pulsating artifacts and temporal stability issues".

@compute @workgroup_size(16, 16, 1)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let source_texel_size = 1.0 / bloom_blur_constants.input_texture_size;
    let target_texel_size = 1.0 / bloom_blur_constants.output_texture_size;

    let texture_coord = (vec2f(global_id.xy) + vec2f(0.5)) * target_texel_size;

    if (global_id.x < u32(bloom_blur_constants.output_texture_size.x) &&
        global_id.y < u32(bloom_blur_constants.output_texture_size.y)) {
        // Take 13 samples around current texel:
        // a - b - c
        // - j - k -
        // d - e - f
        // - l - m -
        // g - h - i
        // === ('e' is the current texel) ===
        let a = textureSampleLevel(input_texture, clamped_sampler, vec2f(texture_coord.x - 2.0 * source_texel_size.x, texture_coord.y + 2.0 * source_texel_size.y), 0).rgb;
        let b = textureSampleLevel(input_texture, clamped_sampler, vec2f(texture_coord.x, texture_coord.y + 2.0 * source_texel_size.y), 0).rgb;
        let c = textureSampleLevel(input_texture, clamped_sampler, vec2f(texture_coord.x + 2.0 * source_texel_size.x, texture_coord.y + 2.0 * source_texel_size.y), 0).rgb;

        let d = textureSampleLevel(input_texture, clamped_sampler, vec2f(texture_coord.x - 2.0 * source_texel_size.x, texture_coord.y), 0).rgb;
        let e = textureSampleLevel(input_texture, clamped_sampler, vec2f(texture_coord.x, texture_coord.y), 0).rgb;
        let f = textureSampleLevel(input_texture, clamped_sampler, vec2f(texture_coord.x + 2.0 * source_texel_size.x, texture_coord.y), 0).rgb;

        let g = textureSampleLevel(input_texture, clamped_sampler, vec2f(texture_coord.x - 2.0 * source_texel_size.x, texture_coord.y - 2.0 * source_texel_size.y), 0).rgb;
        let h = textureSampleLevel(input_texture, clamped_sampler, vec2f(texture_coord.x, texture_coord.y - 2.0 * source_texel_size.y), 0).rgb;
        let i = textureSampleLevel(input_texture, clamped_sampler, vec2f(texture_coord.x + 2.0 * source_texel_size.x, texture_coord.y - 2.0 * source_texel_size.y), 0).rgb;

        let j = textureSampleLevel(input_texture, clamped_sampler, vec2f(texture_coord.x - 1.0 * source_texel_size.x, texture_coord.y + 1.0 * source_texel_size.y), 0).rgb;
        let k = textureSampleLevel(input_texture, clamped_sampler, vec2f(texture_coord.x + 1.0 * source_texel_size.x, texture_coord.y + 1.0 * source_texel_size.y), 0).rgb;
        let l = textureSampleLevel(input_texture, clamped_sampler, vec2f(texture_coord.x - 1.0 * source_texel_size.x, texture_coord.y - 1.0 * source_texel_size.y), 0).rgb;
        let m = textureSampleLevel(input_texture, clamped_sampler, vec2f(texture_coord.x + 1.0 * source_texel_size.x, texture_coord.y - 1.0 * source_texel_size.y), 0).rgb;

        // Apply weighted distribution:
        // 0.5 + 0.125 + 0.125 + 0.125 + 0.125 = 1
        // a,b,d,e * 0.125
        // b,c,e,f * 0.125
        // d,e,g,h * 0.125
        // e,f,h,i * 0.125
        // j,k,l,m * 0.5
        // This shows 5 square areas that are being sampled. But some of them overlap,
        // so to have an energy preserving downsample we need to make some adjustments.
        // The weights are the distributed, so that the sum of j,k,l,m (e.g.)
        // contribute 0.5 to the final color output. The code below is written
        // to effectively yield this sum. We get:
        // 0.125*5 + 0.03125*4 + 0.0625*4 = 1

        var downsample_color = e * 0.125;
        downsample_color += (a + c + g + i) * 0.03125;
        downsample_color += (b + d + f + h) * 0.0625;
        downsample_color += (j + k + l + m) * 0.125;
        downsample_color = max(downsample_color, vec3f(0.0001));

        textureStore(output_texture, vec2i(global_id.xy), vec4f(downsample_color, 1.0));
    }
}