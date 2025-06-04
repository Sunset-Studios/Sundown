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
  let d_nl = textureLoad(shadow_atlas, coord, i32(in.instance_index), 0);

  // 1) NDC z in [-1,1]
  let z_ndc = d_nl * 2.0 - 1.0;

  // 2) view-space z
  let view_index = frame_info.view_index;
  let view = view_buffer[view_index];
  let near = view.near;
  let far = view.far;
  let linear_z = (2.0 * near * far) / (far + near - z_ndc * (far - near));

  // 3) normalize to [0,1] for display
  let norm_z = (linear_z - near) / (far - near);

  return vec4<f32>(norm_z, norm_z, norm_z, 1.0);
} 