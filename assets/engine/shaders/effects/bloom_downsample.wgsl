#include "common.wgsl"

struct BloomPassParams {
    source_texel_size_and_scale: vec4<f32>,
    curve_threshold: vec4<f32>,
    bloom_color_and_exposure: vec4<f32>,
    output_texture_size: vec4<f32>,
}

@group(1) @binding(0) var input_texture: texture_2d<f32>;
@group(1) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var<uniform> bloom_params: BloomPassParams;

fn downsample_filter_high(tex: texture_2d<f32>, uv: vec2<f32>, texel_size: vec2<f32>) -> vec3<f32> {
    let d = texel_size.xyxy * vec4<f32>(-1.0, -1.0, 1.0, 1.0);

    let s1 = safe_clamp_vec3(textureSampleLevel(tex, clamped_sampler, uv + d.xy, 0.0).rgb);
    let s2 = safe_clamp_vec3(textureSampleLevel(tex, clamped_sampler, uv + d.zy, 0.0).rgb);
    let s3 = safe_clamp_vec3(textureSampleLevel(tex, clamped_sampler, uv + d.xw, 0.0).rgb);
    let s4 = safe_clamp_vec3(textureSampleLevel(tex, clamped_sampler, uv + d.zw, 0.0).rgb);

    let s1w = 1.0 / (max3(s1) + 1.0);
    let s2w = 1.0 / (max3(s2) + 1.0);
    let s3w = 1.0 / (max3(s3) + 1.0);
    let s4w = 1.0 / (max3(s4) + 1.0);
    let inv_weight_sum = 1.0 / (s1w + s2w + s3w + s4w);

    return (s1 * s1w + s2 * s2w + s3 * s3w + s4 * s4w) * inv_weight_sum;
}

fn downsample_filter(tex: texture_2d<f32>, uv: vec2<f32>, texel_size: vec2<f32>) -> vec3<f32> {
    let d = texel_size.xyxy * vec4<f32>(-1.0, -1.0, 1.0, 1.0);

    var s = safe_clamp_vec3(textureSampleLevel(tex, clamped_sampler, uv + d.xy, 0.0).rgb);
    s += safe_clamp_vec3(textureSampleLevel(tex, clamped_sampler, uv + d.zy, 0.0).rgb);
    s += safe_clamp_vec3(textureSampleLevel(tex, clamped_sampler, uv + d.xw, 0.0).rgb);
    s += safe_clamp_vec3(textureSampleLevel(tex, clamped_sampler, uv + d.zw, 0.0).rgb);

    return s * 0.25;
}

@compute @workgroup_size(16, 16, 1)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_size = bloom_params.output_texture_size.xy;
    if (global_id.x >= u32(output_size.x) || global_id.y >= u32(output_size.y)) {
        return;
    }

    let uv = (vec2<f32>(global_id.xy) + vec2<f32>(0.5)) / output_size;
    let texel_size = bloom_params.source_texel_size_and_scale.xy;

#if HIGH_QUALITY_DOWNSAMPLE
    let downsample_color = downsample_filter_high(input_texture, uv, texel_size);
#else
    let downsample_color = downsample_filter(input_texture, uv, texel_size);
#endif

    textureStore(output_texture, vec2i(global_id.xy), vec4<f32>(downsample_color, 1.0));
}
