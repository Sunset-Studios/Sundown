// AS-VSM Stage B: Screen-space Feedback
// Categorises each pixel into a virtual tile & marks it in the bitmask.
#include "common.wgsl"
#include "lighting_common.wgsl"
#include "shadow/shadows_common.wgsl"

@group(1) @binding(0) var camera_depth: texture_2d<f32>;
@group(1) @binding(1) var shadow_atlas: texture_depth_2d_array;
@group(1) @binding(2) var page_table: texture_storage_2d_array<r32uint, read>;
@group(1) @binding(3) var<storage, read> vsm_settings: ASVSMSettings;
@group(1) @binding(4) var<storage, read_write> bitmask: array<atomic<u32>>;
@group(1) @binding(5) var<storage, read> dense_lights_buffer: array<Light>;
@group(1) @binding(6) var<storage, read> dense_shadow_casting_lights_buffer: array<u32>;
@group(1) @binding(7) var<storage, read> light_count_buffer: array<u32>;

fn world_to_light_xy(p: vec3<f32>, r0: vec3<f32>, r1: vec3<f32>) -> vec2<f32> {
  return vec2<f32>(dot(r0, p), dot(r1, p));
}

@compute @workgroup_size(8, 8, 4)
fn cs(@builtin(global_invocation_id) id: vec3<u32>) {
#if SHADOWS_ENABLED
  // ------------------------------------------------------------------
  // Screen bounds check
  // ------------------------------------------------------------------
  let dims = textureDimensions(camera_depth);
  if (id.x >= u32(dims.x) || id.y >= u32(dims.y)) {
    return;
  }

  let light_count = light_count_buffer[0u];
  if (id.z >= light_count) {
    return;
  }

  // ------------------------------------------------------------------
  // Depth reconstruction → world position
  // ------------------------------------------------------------------
  let depth_sample = textureLoad(camera_depth, vec2<i32>(id.xy), 0).r;
  if (depth_sample == 1.0) {
    return; // Far plane, skip
  }

  // Pixel → NDC
  let ndc_xy = ((vec2<f32>(id.xy) + 0.5) / vec2<f32>(dims.xy)) * 2.0 - 1.0;
  let clip_pos = vec4<f32>(ndc_xy, depth_sample * 2.0 - 1.0, 1.0);

  let view_idx = frame_info.view_index;
  let inv_vp   = view_buffer[view_idx].inverse_view_projection_matrix;
  var world_pos = inv_vp * clip_pos;
  world_pos /= world_pos.w;

  // ------------------------------------------------------------------
  // Per-light processing
  // ------------------------------------------------------------------
  let dense_light_index          = id.z;
  let light                      = dense_lights_buffer[dense_light_index];

  if ((light.activated == 0.0) || (light.shadow_casting == 0.0)) {
    return;
  }

  let light_view_index = u32(light.view_index);

  // ------------------------------------------------------------------
  // Use new VSM helpers to choose clip-map and translate coordinates
  // ------------------------------------------------------------------
  let clipmap_index = clamp(
    vsm_calculate_clipmap_index_from_world_pos(
      world_pos.xyz,
      view_buffer[frame_info.view_index].view_projection_matrix,
    ),
    0u,
    u32(vsm_settings.max_lods) - 1u,
  );

  // Clip-space position for this clip-map (already accounting for camera translation)
  let sample_clip = vsm_calculate_sample_clip_value_from_world_pos(
    world_pos.xyz,
    clipmap_index,
    view_buffer[light_view_index].view_projection_matrix,
  );

  // Convert to virtual-texture pixel coordinates
  let uv              = sample_clip.xy * 0.5 + 0.5;          // [-1,1] → [0,1]
  let virtual_pixel   = uv * vsm_settings.virtual_dim;           // pixel units in virtual texture

  // Derive virtual-tile coordinates
  let tile_xy_f       = (virtual_pixel + vec2<f32>(0.5)) / vsm_settings.tile_size;
  let tile_xy_i       = vec2<i32>(floor(tile_xy_f));

  let vtr_i           = i32(vsm_settings.virtual_tiles_per_row);
  let tcx_i           = ((tile_xy_i.x % vtr_i) + vtr_i) % vtr_i;
  let tcy_i           = ((tile_xy_i.y % vtr_i) + vtr_i) % vtr_i;

  let vtr             = u32(vsm_settings.virtual_tiles_per_row);
  let tcx             = u32(tcx_i);
  let tcy             = u32(tcy_i);

  let base_index      = clipmap_index * vtr * vtr;
  let tile_id         = base_index + tcy * vtr + tcx;

  // ------------------------------------------------------------------
  // Mark the tile in the bitmask
  // ------------------------------------------------------------------
  let word_index = tile_id >> 5u;
  let bit_index  = tile_id & 31u;
  let mask       = 1u << bit_index;
  atomicOr(&bitmask[word_index], mask);
#endif
} 