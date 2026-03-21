// AS-VSM Stage C: Update Page Table
// Updates the page table with new (lod, physicalID) for each requested tile.
#include "common.wgsl"
#include "lighting_common.wgsl"
#include "shadow/shadows_common.wgsl"

@group(1) @binding(0) var<storage, read_write> lru: array<atomic<u32>>;
@group(1) @binding(1) var page_table: texture_storage_2d_array<r32uint, read_write>;
@group(1) @binding(2) var<storage, read> light_shadow_idx_buffer: array<u32>;
@group(1) @binding(3) var<storage, read> light_view_idx_buffer: array<u32>;
@group(1) @binding(4) var<uniform> vsm_settings: ASVSMSettings;
@group(1) @binding(5) var<storage, read_write> bitmask: array<u32>;
@group(1) @binding(6) var<storage, read> dense_lights_buffer: DenseLightsBuffer;
@group(1) @binding(7) var<storage, read_write> eviction_counter: array<atomic<u32>>;
@group(1) @binding(8) var page_offset: texture_storage_2d_array<rgba32float, write>;

// ------------------------------------------------------------------
// LRU helpers – the MSB of each physical_id entry is treated as a
// "pinned" flag.  A pinned page is considered in-use and therefore
// ineligible for eviction.
// ------------------------------------------------------------------

// Returns an unpinned physical page id, pinning it atomically in the process.
fn lru_acquire_free_page(total_lru_entries: u32) -> u32 {
  var attempt: u32 = 0u;
  loop {
    // Atomically fetch and increment the head pointer.
    let lru_head        = atomicAdd(&lru[0u], 1u);
    let slot_index      = 1u + (lru_head % total_lru_entries);

    // Load the entry.  MSB == pinned flag, lower 31 bits == physical_id.
    let raw_entry       = atomicLoad(&lru[slot_index]);
    let is_pinned       = (raw_entry & lru_pinned_flag) != 0u;

    if (!is_pinned) {
      // Attempt to pin the entry.  If another thread pins it first, we'll retry.
      let prev = atomicOr(&lru[slot_index], lru_pinned_flag);
      if ((prev & lru_pinned_flag) == 0u) {
        // Successfully pinned – return the physical id (mask off the flag).
        return raw_entry & ~lru_pinned_flag;
      }
    }

    attempt = attempt + 1u;
    if (attempt >= total_lru_entries) {
      // No free pages – signal failure by returning 0xffffffffu.
      return 0xffffffffu;
    }
  }
}

@compute @workgroup_size(256)
fn cs(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
#if SHADOWS_ENABLED
    // ------------------------------------------------------------------
    // Calculate per-light stride inside the bitmask buffer
    // ------------------------------------------------------------------
    let vtpr         = u32(vsm_settings.virtual_tiles_per_row);
    let max_lods     = u32(vsm_settings.max_lods);
    let stride_words = ((vtpr * vtpr * max_lods + 31u) >> 5u);
    // Global word index into the shared buffer (per-light stride offset)
    let global_index = id.x;

    // Compute linear index *within* the bitmask for this light.
    // Each 8×8 work-group covers 64 consecutive 32-bit words.
    // Combine the work-group offset (group_id.x) with the local thread offset
    // to obtain a unique word index for the entire dispatch.
    let light_count = dense_lights_buffer.header.light_count;
    if (id.z >= light_count) {
        return;
    }

    // Fetch the shadow index for this light
    let light_index = global_index / stride_words;
    let shadow_index = light_shadow_idx_buffer[light_index];
    if (shadow_index == 0xffffffffu) {
        return;
    }

    let light_view_idx = light_view_idx_buffer[light_index];
    if (light_view_idx == 0xffffffffu) {
        return;
    }

    // Fetch mask of virtual tiles for *this* light
    var bits = bitmask[global_index];
    var light_offset = view_buffer[light_view_idx].view_matrix[3];

    // Go over all virtual tiles that are currently visible for this light
    while(bits != 0u) {
      let shift = countTrailingZeros(bits);
      bits = bits & (bits - 1u);

      let tile_id = global_index * 32u + shift;

      let pte_coords = vsm_pte_get_tile_coords(tile_id, vsm_settings);
      let page_table_index = shadow_index * max_lods + pte_coords.z;

      let pte = textureLoad(page_table, pte_coords.xy, page_table_index).r;
      let pte_is_valid = vsm_pte_is_valid(pte);

      if (pte_is_valid) {
        let current_physical_id = vsm_pte_to_physical_id(pte, vsm_settings);
        let lru_slot = 1u + current_physical_id;
        atomicOr(&lru[lru_slot], lru_pinned_flag);

        // Increment the PTE frame age field by 1 (cap at max value)
        let current_frame_age = vsm_pte_get_frame_age(pte);
        let new_frame_age = clamp(current_frame_age + 1u, 0u, 0x00000080u);
        let new_pte_value = (pte & ~pte_frame_age_mask) | ((new_frame_age << pte_frame_age_shift) & pte_frame_age_mask);
        textureStore(page_table, pte_coords.xy, page_table_index, vec4<u32>(new_pte_value));

        continue;
      }

      let ptpr = u32(vsm_settings.physical_tiles_per_row);
      let total_lru_entries = ptpr * ptpr * u32(vsm_settings.max_physical_pools);

      // Acquire a free (unpinned) physical page from the LRU ring.
      let physical_id = lru_acquire_free_page(total_lru_entries);
      if (physical_id == 0xffffffffu) {
        atomicAdd(&eviction_counter[0], 1u);
        continue;
      }

      // Update New PTE – build entry with new format
      let physical_id_xy_pool = vsm_physical_id_to_xy_pool(physical_id, vsm_settings);

      let new_pte_value =
        ((physical_id_xy_pool.x << pte_phys_x_shift)  & pte_phys_x_mask)    |
        ((physical_id_xy_pool.y << pte_phys_y_shift)  & pte_phys_y_mask)    |
        ((physical_id_xy_pool.z << pte_pool_id_shift) & pte_pool_id_mask)   |
        (1u      << pte_residency_shift);

      textureStore(page_table, pte_coords.xy, page_table_index, vec4<u32>(vsm_pte_mark_dirty(new_pte_value)));
      // Store the light's VP matrix translation (as a placeholder for the offset)
      // In a real implementation, you may want to store a more precise offset or index.
      textureStore(page_offset, pte_coords.xy, page_table_index, light_offset);
    }    
#endif
} 