#include "common.wgsl"

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2<precision_float>,
  @location(1) @interpolate(flat) instance_index: u32,
};

@group(1) @binding(0) var debug_texture: texture_cube<f32>;

@fragment
fn fs(input: VertexOutput) -> @location(0) vec4<f32> {
  // Map uv from [0,1] to [-1,1] and use z = 1.0 to form a direction
  let direction = normalize(vec3<f32>(input.uv * 2.0 - vec2<f32>(1.0, 1.0), 1.0));
  return textureSample(debug_texture, non_filtering_sampler, direction);
} 