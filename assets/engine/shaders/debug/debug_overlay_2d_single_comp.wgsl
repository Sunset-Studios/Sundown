#include "common.wgsl"

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) instance_index: u32,
};

@group(1) @binding(0) var debug_texture: texture_2d<f32>;

@fragment
fn fs(input: VertexOutput) -> @location(0) vec4<f32> {
  let val = textureSample(debug_texture, non_filtering_sampler, input.uv);
  return vec4<f32>(val.r, val.r, val.r, 1.0);
} 