#include "common.wgsl"

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2<precision_float>,
    @location(1) @interpolate(flat) instance_index: u32,
};

@group(1) @binding(0) var debug_texture: texture_2d<f32>;

@fragment
fn fs(input: VertexOutput) -> @location(0) vec4<f32> {
    // Sample motion vector from RG channels (in NDC space: -1 to 1 range)
    let motion_emissive = textureSample(debug_texture, non_filtering_sampler, input.uv);
    let motion_vector = motion_emissive.rg;
    let magnitude = length(motion_vector) * 10.0;
    let color = vec3f(motion_vector, magnitude);
    return vec4f(color, 1.0);
}

