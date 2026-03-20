#include "common.wgsl"
#include "lighting_common.wgsl"
#include "shadow/shadows_common.wgsl"

// ------------------------------------------------------------------------------------
// Buffers
// ------------------------------------------------------------------------------------ 

@group(1) @binding(0) var<uniform> vsm_settings: ASVSMSettings;
@group(1) @binding(1) var<storage, read> bitmask: array<u32>;
@group(1) @binding(2) var page_table: texture_storage_2d_array<r32uint, read_write>;
@group(1) @binding(3) var page_offset: texture_storage_2d_array<rgba32float, write>;
@group(1) @binding(4) var<storage, read> light_idx_buf: array<u32>;
@group(1) @binding(5) var<storage, read> lights: array<Light>;

// ------------------------------------------------------------------------------------
// Helper Functions
// ------------------------------------------------------------------------------------ 

// Function to mark a tile
fn mark_tile(tile_coords: vec2<u32>, clipmap_index: u32, shadow_index: u32, view: ptr<function, View>) {
    let slice = shadow_index * u32(vsm_settings.max_lods) + clipmap_index;
    let word_and_mask = vsm_get_virtual_tile_word_and_mask(
        tile_coords,
        clipmap_index,
        shadow_index,
        vsm_settings
    );
    let word = word_and_mask.x;
    let mask = word_and_mask.y;
    let is_visible = (bitmask[word] & mask) != 0u;
    if (is_visible) {
        var pte = textureLoad(page_table, tile_coords, slice).r;
        pte = vsm_pte_mark_dirty(pte);
        textureStore(page_table, tile_coords, slice, vec4<u32>(pte));
        textureStore(page_offset, tile_coords, slice, vec4<f32>(view.view_matrix[3]));
    }
}

// ------------------------------------------------------------------------------------
// Compute Shader
// ------------------------------------------------------------------------------------ 

@compute @workgroup_size(8, 8, 4)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
#if SHADOWS_ENABLED
  let vtpr = u32(vsm_settings.virtual_tiles_per_row);
  let max_lods = u32(vsm_settings.max_lods);

  if (global_id.x >= vtpr || global_id.y >= vtpr) {
    return;
  }

  let light_index = global_id.z / max_lods;
  let clipmap_index = global_id.z % max_lods;

  let light_index_proper = light_idx_buf[light_index];
  let light = lights[light_index_proper];

  let shadow_index = u32(light.shadow_index);
  if (shadow_index == 0xffffffffu) {
    return;
  }

  let view_index = u32(light.view_index);
  if (view_index == 0xffffffffu) {
    return;
  }

  if (u32(light.shadows_dirty) == 0u) {
    return;
  }

  var view = view_buffer[view_index];

  let tile_coords = vec2<u32>(global_id.xy);
  mark_tile(tile_coords, clipmap_index, shadow_index, &view);
#endif
}