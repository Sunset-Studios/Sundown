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
@group(1) @binding(2) var<storage, read> settings: ASVSMSettings;
@group(1) @binding(3) var<storage, read> dense_lights_buffer: array<Light>;

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
  let tile_size = u32(settings.tile_size);
  let virtual_dim = u32(settings.virtual_dim);
  if (tile_size == 0u || virtual_dim == 0u) {
    return vec4<f32>(0.0);
  }

  // Sample world position; if w == 0 (no geometry), discard
  let world_pos_sample = textureSample(world_position_tex, global_sampler, input.uv);
  if (all(world_pos_sample.xyz == vec3<f32>(0.0))) {
    return vec4<f32>(0.0);
  }

  let world_pos = world_pos_sample.xyz;

  let light     = dense_lights_buffer[0u];
  let view_idx  = u32(light.view_index);

  // Determine clip-map level using new helper
  let camera_view_index = frame_info.view_index;
  let clipmap_index = clamp(
    vsm_calculate_clipmap_index_from_world_pos(
      world_pos,
      view_buffer[camera_view_index].view_projection_matrix,
    ),
    0u,
    u32(settings.max_lods) - 1u,
  );

  // Transform world position into clip-space for that clip-map
  let sample_clip = vsm_calculate_sample_clip_value_from_world_pos(
    world_pos,
    clipmap_index,
    view_buffer[view_idx].view_projection_matrix,
  );

  // Virtual-texture pixel coords in clip-map 0 space
  var uv              = sample_clip.xy * 0.5 + 0.5;            // [-1,1] → [0,1]
  let virtual_pixel   = uv * settings.virtual_dim;

  // Compute virtual-tile coordinates
  let tile_xy_f       = (virtual_pixel + vec2<f32>(0.5)) / settings.tile_size;
  let tile_xy_i       = vec2<i32>(floor(tile_xy_f));

  // wrap
  let vtr_i           = i32(settings.virtual_tiles_per_row);
  let tcx_i           = ((tile_xy_i.x % vtr_i) + vtr_i) % vtr_i;
  let tcy_i           = ((tile_xy_i.y % vtr_i) + vtr_i) % vtr_i;

  let vtr             = u32(settings.virtual_tiles_per_row);
  let tcx             = u32(tcx_i);
  let tcy             = u32(tcy_i);

  let base_index      = clipmap_index * vtr * vtr;
  let tile_id         = base_index + tcy * vtr + tcx;

  // Hash colour encodes tile id & lod (mix into value)
  let base_color     = hash_u32(tile_id);
  let lod_factor     = f32(clipmap_index) / f32(settings.max_lods - 1.0);
  let color          = mix(base_color, vec3<f32>(lod_factor, 0.0, 1.0 - lod_factor), 0.35);

  return vec4<f32>(color, 1.0);
#else
  return vec4<f32>(0.0);
#endif
} 