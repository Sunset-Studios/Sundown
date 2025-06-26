#include "common.wgsl"
#include "lighting_common.wgsl"
#include "shadow/shadows_common.wgsl"

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2<precision_float>,
    @location(1) @interpolate(flat) instance_index: u32,
};

@group(1) @binding(0) var dummy_depth_image: texture_depth_2d;

@fragment
fn fs(input: VertexOutput) -> @location(0) vec4<f32> {
#if SHADOWS_ENABLED
  let depth_sample = textureSample(dummy_depth_image, non_filtering_sampler, input.uv);
  let near = view_buffer[frame_info.view_index].near;
  let far = view_buffer[frame_info.view_index].far;
  let lin_depth = linearize_depth(depth_sample, near, far);
  return vec4<f32>(lin_depth, lin_depth, lin_depth, 1.0);
#else
  return vec4<f32>(0.0);
#endif
} 