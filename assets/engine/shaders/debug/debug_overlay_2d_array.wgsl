#include "common.wgsl"

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2<precision_float>,
    @location(1) @interpolate(flat) instance_index: u32,
};

@group(1) @binding(0) var debug_texture: texture_2d_array<f32>;

@fragment
fn fs(input: VertexOutput) -> @location(0) vec4<f32> {
  return textureSample(debug_texture, non_filtering_sampler, input.uv);
} 