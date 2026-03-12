#include "common.wgsl"

struct BloomPassParams {
    source_texel_size_and_scale: vec4<f32>,
    curve_threshold: vec4<f32>,
    bloom_color_and_exposure: vec4<f32>,
    output_texture_size: vec4<f32>,
}

@group(1) @binding(0) var input_texture: texture_2d<f32>;
@group(1) @binding(1) var base_texture: texture_2d<f32>;
@group(1) @binding(2) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(3) var<uniform> bloom_params: BloomPassParams;

fn upsample_filter_high(tex: texture_2d<f32>, uv: vec2<f32>, texel_size: vec2<f32>, sample_scale: f32) -> vec3<f32> {
    let d = texel_size.xyxy * vec4<f32>(1.0, 1.0, -1.0, 0.0) * sample_scale;

    var s = safe_clamp_vec3(textureSampleLevel(tex, clamped_sampler, uv - d.xy, 0.0).rgb);
    s += safe_clamp_vec3(textureSampleLevel(tex, clamped_sampler, uv - d.wy, 0.0).rgb) * 2.0;
    s += safe_clamp_vec3(textureSampleLevel(tex, clamped_sampler, uv - d.zy, 0.0).rgb);

    s += safe_clamp_vec3(textureSampleLevel(tex, clamped_sampler, uv + d.zw, 0.0).rgb) * 2.0;
    s += safe_clamp_vec3(textureSampleLevel(tex, clamped_sampler, uv, 0.0).rgb) * 4.0;
    s += safe_clamp_vec3(textureSampleLevel(tex, clamped_sampler, uv + d.xw, 0.0).rgb) * 2.0;

    s += safe_clamp_vec3(textureSampleLevel(tex, clamped_sampler, uv + d.zy, 0.0).rgb);
    s += safe_clamp_vec3(textureSampleLevel(tex, clamped_sampler, uv + d.wy, 0.0).rgb) * 2.0;
    s += safe_clamp_vec3(textureSampleLevel(tex, clamped_sampler, uv + d.xy, 0.0).rgb);

    return s * (1.0 / 16.0);
}

@compute @workgroup_size(16, 16, 1)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_size = bloom_params.output_texture_size.xy;
    if (global_id.x >= u32(output_size.x) || global_id.y >= u32(output_size.y)) {
        return;
    }

    let uv = (vec2<f32>(global_id.xy) + vec2<f32>(0.5)) / output_size;
    let source_texel_size = bloom_params.source_texel_size_and_scale.xy;
    let sample_scale = bloom_params.source_texel_size_and_scale.z;

    let blur = upsample_filter_high(input_texture, uv, source_texel_size, sample_scale);
    let base = safe_clamp_vec3(textureSampleLevel(base_texture, clamped_sampler, uv, 0.0).rgb);

    textureStore(output_texture, vec2i(global_id.xy), vec4<f32>(base + blur, 1.0));
}
