// AS-VSM Stage A: Screen-space Feedback
// Categorises each pixel into a virtual tile & marks it in the bitmask.
#include "common.wgsl"
#include "lighting_common.wgsl"
#include "shadow/shadows_common.wgsl"

@group(1) @binding(0) var camera_depth: texture_2d<f32>;
@group(1) @binding(1) var<uniform> vsm_settings: ASVSMSettings;
@group(1) @binding(2) var<storage, read_write> bitmask: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read> light_view_buffer: array<u32>;
@group(1) @binding(4) var<storage, read> light_shadow_idx_buffer: array<u32>;
@group(1) @binding(5) var<storage, read> light_count_buffer: array<u32>;

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
  // Per-light processing
  // ------------------------------------------------------------------
  let light_index       = id.z;

  let light_view_index  = light_view_buffer[light_index];
  if (light_view_index == 0xffffffffu) {
    return;
  }

  let shadow_index = light_shadow_idx_buffer[light_index];
  if (shadow_index == 0xffffffffu) {
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
  let clip_pos = vec4<f32>(ndc_xy, depth_sample, 1.0);

  let inv_vp   = view_buffer[frame_info.view_index].inverse_view_projection_matrix;
  var world_pos = inv_vp * clip_pos;
  world_pos /= world_pos.w;

  // ------------------------------------------------------------------
  // Use VSM helpers to choose clip-map and translate coordinates
  // ------------------------------------------------------------------
  let clipmap_index = clamp(
    vsm_calculate_clipmap_index_from_world_pos(
      world_pos,
      view_buffer[frame_info.view_index].view_projection_matrix,
    ),
    0u,
    u32(vsm_settings.max_lods) - 1u,
  );

  // Clip-space position for this clip-map (already accounting for camera translation)
  let sample_clip = vsm_calculate_sample_clip_value_from_world_pos(
    world_pos,
    clipmap_index,
    view_buffer[light_view_index].view_projection_matrix,
  );

  // Convert to virtual-texture pixel coordinates
  let uv              = sample_clip.xy * 0.5 + 0.5;
  let wrapped_uv      = fract(uv);
  let virtual_pixel   = wrapped_uv * vsm_settings.virtual_dim;           // pixel units in virtual texture

  // Derive virtual-tile coordinates using signed integers to handle negatives correctly.
  let tile_xy_f     = (virtual_pixel + vec2<f32>(0.5)) / vsm_settings.tile_size;
  let tile_coords     = vec2<u32>(floor(tile_xy_f));

  let word_and_mask = vsm_get_virtual_tile_word_and_mask(
    tile_coords,
    clipmap_index,
    shadow_index,
    vsm_settings
  );

  let word = word_and_mask.x;
  let mask = word_and_mask.y;

  atomicOr(&bitmask[word], mask);
#endif
} 