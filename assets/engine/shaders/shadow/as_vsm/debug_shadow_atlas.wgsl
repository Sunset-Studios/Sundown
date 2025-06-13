// Debug view for AS-VSM shadow atlas (first layer)
#include "common.wgsl"

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2<precision_float>,
    @location(1) @interpolate(flat) instance_index: u32,
};

@group(1) @binding(0) var shadow_atlas: texture_depth_2d_array;

@fragment fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
  let dims = textureDimensions(shadow_atlas, 0);
  let coord = vec2<i32>(
      i32(in.uv.x * f32(dims.x)),
      i32(in.uv.y * f32(dims.y))
  );
  // sample the correct array slice
  let depth = textureLoad(shadow_atlas, coord, i32(in.instance_index), 0);

  let view = view_buffer[frame_info.view_index];
  let lin_depth = linearize_depth(depth, view.near, view.far) / 100.0;

  return vec4<f32>(lin_depth, lin_depth, lin_depth, 1.0);
} 