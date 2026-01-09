#include "common.wgsl"
#include "lighting_common.wgsl"
#include "shadow/shadows_common.wgsl"

@group(1) @binding(0) var page_table: texture_storage_2d_array<r32uint, read_write>;
@group(1) @binding(1) var<storage, read_write> lru: array<atomic<u32>>;
@group(1) @binding(2) var<uniform> vsm_settings: ASVSMSettings;
@group(1) @binding(3) var<storage, read> bitmask: array<u32>;
@group(1) @binding(4) var<storage, read> light_shadow_idx_buffer: array<u32>;
@group(1) @binding(5) var<storage, read> dense_lights_buffer: DenseLightsBuffer;
@group(1) @binding(6) var<storage, read_write> eviction_counter: array<atomic<u32>>;

@compute @workgroup_size(8, 8, 4)
fn cs(@builtin(global_invocation_id) id: vec3<u32>) {
#if SHADOWS_ENABLED
  let vtpr = u32(vsm_settings.virtual_tiles_per_row);
  let max_lods = u32(vsm_settings.max_lods);

  if (id.x >= vtpr || id.y >= vtpr) {
    return;
  }

  let light_idx = id.z / max_lods;
  let clipmap_idx = id.z % max_lods;

  let light_count = dense_lights_buffer.header.light_count;
  if (light_idx >= light_count) {
    return;
  }

  let shadow_index = light_shadow_idx_buffer[light_idx];
  if (shadow_index == 0xffffffffu) {
    return;
  }

  let tile_coords = vec2<u32>(id.xy);
  let word_and_mask = vsm_get_virtual_tile_word_and_mask(
    tile_coords,
    clipmap_idx,
    shadow_index,
    vsm_settings,
  );
  let word = word_and_mask.x;
  let mask = word_and_mask.y;
  let not_visible = (bitmask[word] & mask) == 0u;

  let slice = shadow_index * max_lods + clipmap_idx;
  let entry = textureLoad(page_table, tile_coords, slice).r;
  let current_physical_id = vsm_pte_to_physical_id(entry, vsm_settings);

  if (vsm_pte_is_valid(entry) && not_visible) {
    atomicAnd(&lru[1u + current_physical_id], ~lru_pinned_flag);
    textureStore(page_table, tile_coords, slice, vec4<u32>(0u));
  }
#endif
}

fn atomic_decrement_if_not_zero(counter: ptr<storage, atomic<u32>, read_write>) -> u32 {
    var old_value = atomicLoad(counter);
    loop {
        if (old_value == 0u) {
            // Already zero, do not decrement
            return 0u;
        }
        let new_value = old_value - 1u;
        let result = atomicCompareExchangeWeak(counter, old_value, new_value);
        if (result.exchanged) {
            // Successfully decremented
            return old_value;
        }
        // Try again with the new observed value
        old_value = result.old_value;
    }
}