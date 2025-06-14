#include "common.wgsl"
#include "postprocess_common.wgsl"

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<precision_float>,
    @location(1) @interpolate(flat) instance_index: u32,
}

struct BloomResolveConstants {
    exposure: f32,
    bloom_intensity: f32,
    bloom_threshold: f32,
    bloom_knee: f32,
}

@group(1) @binding(0) var scene_color: texture_2d<f32>;
@group(1) @binding(1) var bloom_brightness: texture_2d<f32>;
@group(1) @binding(2) var<uniform> bloom_resolve_constants: BloomResolveConstants;

fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn apply_bloom(scene: vec3<f32>, bloom: vec3<f32>, intensity: f32, threshold: f32, knee: f32) -> vec3<f32> {
    let scene_luminance = luminance(scene);
    
    // Soft threshold
    let soft_threshold = smoothstep(threshold - knee, threshold + knee, scene_luminance);
    
    // Non-linear intensity scaling based on scene brightness
    let adjusted_intensity = intensity * pow(soft_threshold, 2.0);
    
    return mix(scene, scene + bloom * adjusted_intensity, soft_threshold);
}

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4<precision_float> {
    let uv = vec2<f32>(in.uv);
    var color = textureSample(scene_color, global_sampler, uv).rgb;
    let bloom_color = textureSample(bloom_brightness, global_sampler, uv).rgb;

    color = apply_bloom(
        color, 
        bloom_color, 
        bloom_resolve_constants.bloom_intensity,
        bloom_resolve_constants.bloom_threshold,
        bloom_resolve_constants.bloom_knee
    );
    
    color = reinhard_tonemapping(color, bloom_resolve_constants.exposure);

    return vec4<precision_float>(vec4<f32>(color, 1.0));
}