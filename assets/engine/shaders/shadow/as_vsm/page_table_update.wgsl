// AS-VSM Stage C: Update Page Table
// Updates the page table with new (lod, physicalID) for each requested tile.
#include "common.wgsl"
#include "lighting_common.wgsl"
#include "shadow/shadows_common.wgsl"

@group(1) @binding(0) var<storage, read_write> lru: array<atomic<u32>>;
@group(1) @binding(1) var page_table: texture_storage_2d_array<r32uint, read_write>;
@group(1) @binding(2) var<storage, read> light_shadow_idx_buffer: array<u32>;
@group(1) @binding(3) var<uniform> vsm_settings: ASVSMSettings;
@group(1) @binding(4) var<storage, read_write> bitmask: array<u32>;
@group(1) @binding(5) var<storage, read> light_count_buffer: array<u32>;

@compute @workgroup_size(8, 8, 4)
fn cs(@builtin(global_invocation_id) id: vec3<u32>) {
#if SHADOWS_ENABLED
    // ------------------------------------------------------------------
    // Calculate per-light stride inside the bitmask buffer
    // ------------------------------------------------------------------
    let vtpr = u32(vsm_settings.virtual_tiles_per_row);
    let stride_words = ((vtpr * vtpr * u32(vsm_settings.max_lods) + 31u) >> 5u);

    // Compute linear index *within* the bitmask for this light
    let index_in_stride = id.y * 8u + id.x; // 8×8 x/y work-group → 64 indices
    if (index_in_stride >= stride_words) {
        return;
    }

    let light_count = light_count_buffer[0u];
    if (id.z >= light_count) {
        return;
    }

    // Fetch the shadow index for this light
    let shadow_index = light_shadow_idx_buffer[id.z];
    if (shadow_index == 0xffffffffu) {
        return;
    }

    // Global word index into the shared buffer
    let global_index = shadow_index * stride_words + index_in_stride;

    // Fetch mask of virtual tiles for *this* light
    var bits = bitmask[global_index];

    while(bits != 0u) {
      let shift = countTrailingZeros(bits);
      bits = bits & (bits - 1u);

      let tile_id = index_in_stride * 32u + shift;

      let new_pte_coords = vsm_pte_get_tile_coords(tile_id, vsm_settings);
      let page_table_index = shadow_index * u32(vsm_settings.max_lods) + new_pte_coords.z;

      let current_pte_val_at_new_coords = textureLoad(page_table, new_pte_coords.xy, page_table_index).r;
      let current_pte_is_valid = vsm_pte_is_resident(current_pte_val_at_new_coords);
      if (current_pte_is_valid) {
        continue; // Already mapped by a concurrent thread or previous pass
      }

      let ptpr = u32(vsm_settings.physical_tiles_per_row);
      let total_lru_entries = ptpr * ptpr * u32(vsm_settings.max_physical_pools);

      let lru_head = atomicAdd(&lru[0u], 1u);
      let lru_slot_index = 1u + lru_head % total_lru_entries;
      let physical_id = atomicLoad(&lru[lru_slot_index]);

      // Update New PTE – build entry with new format
      let pool_id = physical_id / (ptpr * ptpr);
      let local_physical_id = physical_id - pool_id * (ptpr * ptpr);
      let phys_x = local_physical_id % ptpr;
      let phys_y = local_physical_id / ptpr;

      let new_pte_value =
        ((phys_x  << pte_phys_x_shift) & pte_phys_x_mask)    |
        ((phys_y  << pte_phys_y_shift) & pte_phys_y_mask)    |
        ((pool_id << pte_pool_id_shift) & pte_pool_id_mask)   |
        (1u      << pte_residency_shift) | // resident
        (1u      << pte_dirty_shift)     | // dirty, needs clearing
        (0u      << pte_frame_age_shift);

      textureStore(page_table, new_pte_coords.xy, page_table_index, vec4<u32>(new_pte_value));
    }    
#endif
} 