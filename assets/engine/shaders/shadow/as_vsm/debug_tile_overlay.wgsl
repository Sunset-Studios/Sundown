#include "common.wgsl"
#include "lighting_common.wgsl"
#include "shadow/shadows_common.wgsl"

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2<precision_float>,
    @location(1) @interpolate(flat) instance_index: u32,
};

@group(1) @binding(0) var page_table: texture_storage_2d_array<r32uint, read>;
@group(1) @binding(1) var world_position_tex: texture_2d<f32>;
@group(1) @binding(2) var<uniform> vsm_settings: ASVSMSettings;
@group(1) @binding(3) var<storage, read> light_view_buffer: array<u32>;

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
#if SHADOWS_ENABLED
  let tile_size = u32(vsm_settings.tile_size);
  let virtual_dim = u32(vsm_settings.virtual_dim);
  if (tile_size == 0u || virtual_dim == 0u) {
    return vec4<f32>(0.0);
  }

  // Sample world position; if w == 0 (no geometry), discard
  let world_pos_sample = textureSample(world_position_tex, global_sampler, input.uv);
  if (all(world_pos_sample.xyz == vec3<f32>(0.0))) {
    return vec4<f32>(0.0);
  }

  let view_idx        = light_view_buffer[0u];
  let camera_vp       = view_buffer[frame_info.view_index].view_projection_matrix;
  let clipmap0_vp     = view_buffer[view_idx].view_projection_matrix;
  let world_pos       = vec4<f32>(world_pos_sample.xyz, 1.0);

  let vtile_info      = vsm_world_to_virtual_tile(world_pos, camera_vp, clipmap0_vp, vsm_settings);

  let vtr             = u32(vsm_settings.virtual_tiles_per_row);
  let tile_id         = vtile_info.clipmap_index * vtr * vtr + vtile_info.tile_coords.y * vtr + vtile_info.tile_coords.x;

  // Hash colour encodes tile id & lod (mix into value)
  let base_color     = hash_u32(tile_id);
  let lod_factor     = f32(vtile_info.clipmap_index) / f32(vsm_settings.max_lods - 1.0);
  let color          = mix(base_color, vec3<f32>(lod_factor, 0.0, 1.0 - lod_factor), 0.35);

  return vec4<f32>(color, 1.0);
#else
  return vec4<f32>(0.0);
#endif
} 