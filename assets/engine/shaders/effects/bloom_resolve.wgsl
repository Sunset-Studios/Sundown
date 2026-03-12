#include "common.wgsl"
#include "postprocess_common.wgsl"

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) instance_index: u32,
}

struct BloomResolveConstants {
    source_texel_size_and_scale: vec4<f32>,
    curve_threshold: vec4<f32>,
    bloom_color_and_exposure: vec4<f32>,
}

@group(1) @binding(0) var scene_color: texture_2d<f32>;
@group(1) @binding(1) var bloom_brightness: texture_2d<f32>;
@group(1) @binding(2) var<uniform> bloom_resolve_constants: BloomResolveConstants;

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

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = vec2<f32>(in.uv);
    let scene = safe_clamp_vec3(textureSample(scene_color, global_sampler, uv).rgb);
    let bloom = upsample_filter_high(
        bloom_brightness,
        uv,
        bloom_resolve_constants.source_texel_size_and_scale.xy,
        bloom_resolve_constants.source_texel_size_and_scale.z,
    );
    var color = scene + bloom * bloom_resolve_constants.bloom_color_and_exposure.xyz;
    color = reinhard_tonemapping(color, bloom_resolve_constants.bloom_color_and_exposure.w);

    return vec4<f32>(vec4<f32>(color, 1.0));
}
