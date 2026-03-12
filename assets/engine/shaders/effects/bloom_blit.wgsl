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

fn median_vec3(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    return a + b + c - min(min(a, b), c) - max(max(a, b), c);
}

@compute @workgroup_size(16, 16, 1)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_size = bloom_params.output_texture_size.xy;
    if (global_id.x >= u32(output_size.x) || global_id.y >= u32(output_size.y)) {
        return;
    }

    let uv = (vec2<f32>(global_id.xy) + vec2<f32>(0.5)) / output_size;
    let texel_size = bloom_params.source_texel_size_and_scale.xy;
    let d = vec3<f32>(texel_size.x, texel_size.y, 0.0);

    let s0 = safe_clamp_vec3(textureSampleLevel(input_texture, clamped_sampler, uv, 0.0).rgb);
    let s1 = safe_clamp_vec3(textureSampleLevel(input_texture, clamped_sampler, uv - d.xz, 0.0).rgb);
    let s2 = safe_clamp_vec3(textureSampleLevel(input_texture, clamped_sampler, uv + d.xz, 0.0).rgb);
    let s3 = safe_clamp_vec3(textureSampleLevel(input_texture, clamped_sampler, uv - d.zy, 0.0).rgb);
    let s4 = safe_clamp_vec3(textureSampleLevel(input_texture, clamped_sampler, uv + d.zy, 0.0).rgb);
    var color = median_vec3(median_vec3(s0, s1, s2), s3, s4);

    var brightness = max3(color);
    var rq = clamp(brightness - bloom_params.curve_threshold.x, 0.0, bloom_params.curve_threshold.y);
    rq = bloom_params.curve_threshold.z * rq * rq;

    color *= max(rq, brightness - bloom_params.curve_threshold.w) / max(1e-5, brightness);

    let clamp_intensity = bloom_params.source_texel_size_and_scale.w;
    if (clamp_intensity > 0.0) {
        brightness = max(1e-5, max3(color));
        color *= 1.0 - max(0.0, brightness - clamp_intensity) / brightness;
    }

    textureStore(output_texture, vec2i(global_id.xy), vec4<f32>(color, 1.0));
}
