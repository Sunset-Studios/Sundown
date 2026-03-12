#include "common.wgsl"
#include "lighting_common.wgsl"
#include "shadow/shadows_sampling.wgsl"

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) instance_index: u32,
};

@group(1) @binding(0) var page_table: texture_storage_2d_array<r32uint, read>;
@group(1) @binding(1) var world_position_tex: texture_2d<f32>;
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

  // Sample world position; if w == 0 (no geometry), discard
  let world_pos_sample = textureSample(world_position_tex, non_filtering_sampler, input.uv);
  if (all(world_pos_sample.xyz == vec3<f32>(0.0))) {
    return vec4<f32>(0.0);
  }

  let view_idx      = light_view_buffer[0u];
  let clipmap0_vp   = view_buffer[view_idx].view_projection_matrix;
  let camera_vp     = view_buffer[u32(frame_info.view_index)].view_projection_matrix;
  let world_pos     = vec4<f32>(world_pos_sample.xyz, 1.0);

  let vtile_info    = vsm_world_to_virtual_tile(world_pos, camera_vp, clipmap0_vp, vsm_settings);
  let ptile_info    = vsm_vtile_to_ptile(vtile_info, vsm_settings, 0u, page_table);

  // Hash colour encodes tile id & lod (mix into value)
  let base_color    = hash_u32(vtile_info.tile_id);
  let lod_factor    = f32(vtile_info.clipmap_index) / f32(vsm_settings.max_lods);

  // Color logic: gray if not dirty, red if dirty
  let gray = mix(vec3<f32>(0.7), vec3<f32>(1.0), base_color.x * 0.2 + 0.1); // random gray shade
  let red = mix(vec3<f32>(1.0, 0.2, 0.2), vec3<f32>(1.0, 0.5, 0.5), base_color.x * 0.2 + 0.1); // random red shade
  let color = select(gray, red, ptile_info.is_dirty);

  // Compute depth
  let depth         = vsm_shadow_depth(
                          vec4<f32>(world_pos_sample.xyz, 1.0),
                          vec3<f32>(0.0, 0.0, 0.0),
                          vec3<f32>(0.0, 0.0, 0.0),
                          view_idx,
                          0u,
                          page_offset,
                          vsm_settings
                      );

  let filter_res    = vsm_sample_shadow(
                          depth,
                          vec4<f32>(world_pos_sample.xyz, 1.0),
                          vec3<f32>(0.0, 0.0, 0.0),
                          vec3<f32>(0.0, 0.0, 0.0),
                          view_idx,
                          0u,
                          page_table,
                          vsm_settings
                      );

  let shadow_factor  = select(1.0, max(f32(filter_res.depth), 0.3), filter_res.valid);

  return vec4<f32>(color * shadow_factor, 1.0);
#else
  return vec4<f32>(0.0);
#endif
} 