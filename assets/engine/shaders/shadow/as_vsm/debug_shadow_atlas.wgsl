// Debug view for AS-VSM shadow atlas (first layer)
#include "common.wgsl"
#include "shadow/shadows_common.wgsl"

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) instance_index: u32,
};

@group(1) @binding(0) var<storage, read> shadow_atlas_depth: array<u32>;
@group(1) @binding(1) var<uniform> vsm_settings: ASVSMSettings;

@fragment fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
  let dims = u32(vsm_settings.physical_dim);
  let uv = in.uv * 0.5 + 0.5;
  let coord = vec2<u32>(
      u32(uv.x * f32(dims)),
      u32(uv.y * f32(dims))
  );

  let pool_index = 0u;
  let linear_index = pool_index * dims * dims + coord.y * dims + coord.x;
  let packed_depth = shadow_atlas_depth[linear_index];
  let depth_clip = unpack_depth(packed_depth);

  return vec4<f32>(depth_clip, depth_clip, depth_clip, 1.0);
} 