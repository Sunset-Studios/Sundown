#include "common.wgsl"
#include "lighting_common.wgsl"
#include "shadow/shadows_common.wgsl"

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) instance_index: u32,
};

@group(1) @binding(0) var dummy_depth_image: texture_2d<f32>;

@fragment
fn fs(input: VertexOutput) -> @location(0) vec4<f32> {
#if SHADOWS_ENABLED
  let depth_sample = textureSample(dummy_depth_image, non_filtering_sampler, input.uv).r;
  return vec4<f32>(depth_sample, depth_sample, depth_sample, 1.0);
#else
  return vec4<f32>(0.0);
#endif
} 