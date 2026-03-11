// AS-VSM Stage A: Screen-space Feedback
// Categorises each pixel into a virtual tile & marks it in the bitmask.
#include "common.wgsl"
#include "lighting_common.wgsl"
#include "shadow/shadows_common.wgsl"

@group(1) @binding(0) var depth_texture: texture_2d<f32>;
@group(1) @binding(1) var<uniform> vsm_settings: ASVSMSettings;
@group(1) @binding(2) var<storage, read_write> bitmask: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read> light_view_buffer: array<u32>;
@group(1) @binding(4) var<storage, read> light_shadow_idx_buffer: array<u32>;
@group(1) @binding(5) var<storage, read> dense_lights_buffer: DenseLightsBuffer;
@group(1) @binding(6) var page_table: texture_storage_2d_array<r32uint, read_write>;

@compute @workgroup_size(8, 8, 1)
fn cs(
  @builtin(global_invocation_id) id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(subgroup_invocation_id) lane_id: u32,
  @builtin(subgroup_size) subgroup_size: u32
) {
#if SHADOWS_ENABLED
  let local_linear = local_id.y * 8u + local_id.x;
  let warp_ctx = make_warp_ctx(local_linear, lane_id, subgroup_size);

  // ------------------------------------------------------------------
  // Compute word/mask for this thread; use (0, 0) when no contribution
  // so all lanes participate in subgroup ops (no early returns before atomics).
  // ------------------------------------------------------------------
  var word = 0u;
  var mask = 0u;
  var contribute = false;

  let dims = textureDimensions(depth_texture);
  let light_count = dense_lights_buffer.header.light_count;
  let light_view_index = light_view_buffer[id.z];
  let shadow_index = light_shadow_idx_buffer[id.z];
  let view_index = u32(frame_info.view_index);

  if (id.x < u32(dims.x) && id.y < u32(dims.y) && id.z < light_count && light_view_index != 0xffffffffu && shadow_index != 0xffffffffu) {
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(dims.xy);
    let depth = textureSampleLevel(depth_texture, non_filtering_sampler, uv, 0.0).r;
    let position = reconstruct_world_position(uv, depth, view_index);
    if (depth < 1.0) {
      let clipmap0_vp = view_buffer[light_view_index].view_projection_matrix;
      let camera_vp = view_buffer[view_index].view_projection_matrix;
      let vtile_info = vsm_world_to_virtual_tile(
        vec4<f32>(position.xyz, 1.0),
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
      word = word_and_mask.x;
      mask = word_and_mask.y;
      contribute = true;
    }
  }

  // Subgroup-level OR: all lanes participate. When all contributing lanes
  // target the same word, combine masks and have the leader do one atomicOr.
  let first_word = warp_broadcast_first_u32(warp_ctx, word);
  let same_word = warp_all(warp_ctx, !contribute || (word == first_word));
  let combined_mask = warp_or(warp_ctx, mask);

  if (same_word) {
    if (is_warp_leader(warp_ctx) && combined_mask != 0u) {
      atomicOr(&bitmask[first_word], combined_mask);
    }
  } else {
    if (contribute) {
      atomicOr(&bitmask[word], mask);
    }
  }
#endif
} 