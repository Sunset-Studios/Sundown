#include "common.wgsl"

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2<precision_float>,
  @location(1) @interpolate(flat) instance_index: u32,
};

@group(1) @binding(0) var debug_texture: texture_depth_2d;

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  let dims = textureDimensions(debug_texture);
  let coord = vec2<i32>(input.uv * vec2<f32>(dims));
  let depth_val = textureLoad(debug_texture, coord, 0);
  return vec4<f32>(depth_val, depth_val, depth_val, 1.0);
} 