#include "common.wgsl"
#include "lighting_common.wgsl"
#include "shadow/shadows_sampling.wgsl"

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) instance_index: u32,
};

@group(1) @binding(0) var page_table: texture_storage_2d_array<r32uint, read>;
@group(1) @binding(1) var depth_texture: texture_2d<f32>;
@group(1) @binding(2) var<uniform> vsm_settings: ASVSMSettings;
@group(1) @binding(3) var<storage, read> light_view_buffer: array<u32>;
@group(1) @binding(4) var<storage, read> shadow_atlas_depth: array<u32>;
@group(1) @binding(5) var page_offset: texture_storage_2d_array<rgba32float, read>;

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

  let view_index = u32(frame_info.view_index);
  let tex_depth = textureSample(depth_texture, non_filtering_sampler, input.uv);
  if (tex_depth.r >= 1.0) {
    return vec4<f32>(0.0);
  }

  let view_idx      = light_view_buffer[0u];
  let clipmap0_vp   = view_buffer[view_idx].view_projection_matrix;
  let camera_vp     = view_buffer[view_index].view_projection_matrix;
  let world_pos     = vec4<f32>(reconstruct_world_position(input.uv, tex_depth.r, view_index), 1.0);

  let vtile_info    = vsm_world_to_virtual_tile(world_pos, camera_vp, clipmap0_vp, vsm_settings);

  // Hash colour encodes tile id & lod (mix into value)
  let base_color    = hash_u32(vtile_info.tile_id);

  // let virtual_tiles_per_row = u32(vsm_settings.virtual_tiles_per_row);
  // let tile_coords_for_color = vtile_info.tile_coords;
  // // Map tile_id to a color that increases monotonically with tile_id, wrapping in a visually distinct way
  // let col = tile_coords_for_color.x % virtual_tiles_per_row;
  // let row = tile_coords_for_color.y % virtual_tiles_per_row;
  // let base_color = vec3<f32>(
  //   f32(col) / f32(virtual_tiles_per_row), 
  //   f32(row) / f32(virtual_tiles_per_row), 
  //   0.0
  // );

  let lod_factor    = f32(vtile_info.clipmap_index) / f32(vsm_settings.max_lods);
  var color         = mix(base_color, vec3<f32>(lod_factor, 0.0, 1.0 - lod_factor), 0.35);

  // Compute depth
  let depth         = vsm_shadow_depth(
                          world_pos,
                          vec3<f32>(0.0, 0.0, 0.0),
                          vec3<f32>(0.0, 0.0, 0.0),
                          view_idx,
                          0u,
                          page_offset,
                          vsm_settings
                      );

  let filter_res    = vsm_sample_shadow(
                          depth,
                          world_pos,
                          vec3<f32>(0.0, 0.0, 0.0),
                          vec3<f32>(0.0, 0.0, 0.0),
                          view_idx,
                          0u,
                          page_table,
                          vsm_settings
                      );

  let shadow_factor  = select(1.0, max(f32(filter_res.depth), 0.3), filter_res.valid);

  let tile_coords = vtile_info.tile_coords.xy;
  let tile_size_f = f32(vsm_settings.tile_size);

  // Compute local UV within the tile (range [0, 1])
  let tile_uv = fract(vec2<f32>(vtile_info.local_pixel) / tile_size_f);

  // Compute distance to nearest edge in texels
  let edge_dist = min(min(tile_uv.x, 1.0 - tile_uv.x), min(tile_uv.y, 1.0 - tile_uv.y)) * tile_size_f;

  // Outline thickness in texels
  let outline_thickness = 1.5;

  // Visualise physical id (x & y) combined
  let e = textureLoad(page_table, vtile_info.tile_coords, vtile_info.clipmap_index).r;
  let is_dirty = (e & pte_dirty_mask) != 0u;

  // If dirty and near edge, blend in white
  let outline = select(0.0, 1.0, is_dirty && (edge_dist < outline_thickness));
  color = mix(color, vec3<f32>(1.0), outline);

  return vec4<f32>(color * shadow_factor, 1.0);
#else
  return vec4<f32>(0.0);
#endif
} 
