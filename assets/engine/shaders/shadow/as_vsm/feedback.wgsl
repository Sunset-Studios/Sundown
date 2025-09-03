// AS-VSM Stage A: Screen-space Feedback
// Categorises each pixel into a virtual tile & marks it in the bitmask.
#include "common.wgsl"
#include "lighting_common.wgsl"
#include "shadow/shadows_common.wgsl"

@group(1) @binding(0) var position_texture: texture_2d<f32>;
@group(1) @binding(1) var<uniform> vsm_settings: ASVSMSettings;
@group(1) @binding(2) var<storage, read_write> bitmask: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read> light_view_buffer: array<u32>;
@group(1) @binding(4) var<storage, read> light_shadow_idx_buffer: array<u32>;
@group(1) @binding(5) var<storage, read> light_count_buffer: array<u32>;
@group(1) @binding(6) var page_table: texture_storage_2d_array<r32uint, read_write>;

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) id: vec3<u32>) {
#if SHADOWS_ENABLED
  // ------------------------------------------------------------------
  // Screen bounds check
  // ------------------------------------------------------------------
  let dims = textureDimensions(position_texture);
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
  let light_view_index  = light_view_buffer[id.z];
  if (light_view_index == 0xffffffffu) {
    return;
  }

  let shadow_index = light_shadow_idx_buffer[id.z];
  if (shadow_index == 0xffffffffu) {
    return;
  }

  let uv = vec2<f32>(id.xy) / vec2<f32>(dims.xy);

  let texture_pos = textureSampleLevel(position_texture, non_filtering_sampler, uv, 0.0);
  if (texture_pos.w == 0.0) {
    return;
  }

  let world_pos = vec4<f32>(texture_pos.xyz, 1.0);
  let clipmap0_vp = view_buffer[light_view_index].view_projection_matrix;
  let camera_vp   = view_buffer[u32(frame_info.view_index)].view_projection_matrix;

  let vtile_info = vsm_world_to_virtual_tile(
    world_pos,
    camera_vp,
    clipmap0_vp,
    vsm_settings
  );

  let word_and_mask = vsm_get_virtual_tile_word_and_mask(
    vtile_info.tile_coords,
    vtile_info.clipmap_index,
    shadow_index,
    vsm_settings
  );
  let word = word_and_mask.x;
  let mask = word_and_mask.y;

  // Skip RMW if the bit is already set
  let prev = atomicLoad(&bitmask[word]);
  let already_set = (prev & mask) != 0u;
  if (!already_set) {
    atomicOr(&bitmask[word], mask);
  }
#endif
} 