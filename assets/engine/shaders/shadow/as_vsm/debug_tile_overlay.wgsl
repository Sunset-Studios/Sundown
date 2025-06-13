#include "common.wgsl"
#include "lighting_common.wgsl"

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2<precision_float>,
    @location(1) @interpolate(flat) instance_index: u32,
};

@group(1) @binding(0) var world_position_tex: texture_2d<f32>;
@group(1) @binding(1) var<storage, read> settings: ASVSMSettings;
@group(1) @binding(2) var<storage, read> dense_lights_buffer: array<Light>;

// Simple hash function to generate pseudo random colors from tile id
fn hash_u32(val: u32) -> vec3<f32> {
  var x = val ^ (val >> 16u);
  x = x * 0x7feb352du;
  x = x ^ (x >> 15u);
  x = x * 0x846ca68bu;
  x = x ^ (x >> 16u);
  let r = f32((x & 0xFFu)) / 255.0;
  let g = f32((x >> 8u) & 0xFFu) / 255.0;
  let b = f32((x >> 16u) & 0xFFu) / 255.0;
  return vec3<f32>(r, g, b);
}

@fragment
fn fs(input: VertexOutput) -> @location(0) vec4<f32> {
  let tile_size = u32(settings.tile_size);
  let virtual_dim = u32(settings.virtual_dim);
  if (tile_size == 0u || virtual_dim == 0u) {
    return vec4<f32>(0.0);
  }

  // Sample world position; if w == 0 (no geometry), discard
  let world_pos_sample = textureSample(world_position_tex, non_filtering_sampler, input.uv);
  if (all(world_pos_sample.xyz == vec3<f32>(0.0))) {
    return vec4<f32>(0.0);
  }

  let world_pos = world_pos_sample.xyz;

  // Retrieve view index for first shadow-casting light
  let light = dense_lights_buffer[0u];
  let view_idx = u32(light.view_index);

  // Project world position into light clip space
  let clip = view_buffer[view_idx].view_projection_matrix * vec4<f32>(world_pos, 1.0);
  let ndc = clip.xyz / clip.w;

  // Compute virtual pixel coordinates in virtual shadow map
  let virtual_pixel = (ndc.xy * vec2<f32>(0.5) + vec2<f32>(0.5)) * f32(virtual_dim);
  let tile_xy = vec2<u32>(virtual_pixel + 0.5) / tile_size;

  let tile_id = tile_xy.y * u32(settings.virtual_tiles_per_row) + tile_xy.x;

  let color = hash_u32(tile_id);

  return vec4<f32>(color, 1.0);
} 