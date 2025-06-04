// AS-VSM Stage E: Update Page Table
// Updates the page table with new (lod, physicalID) for each requested tile.
#include "common.wgsl"
#include "lighting_common.wgsl"

@group(1) @binding(0) var<storage, read> requested_tiles: array<u32>;
@group(1) @binding(1) var<storage, read_write> lru: array<atomic<u32>>;
@group(1) @binding(2) var page_table: texture_storage_2d_array<r32uint, read_write>;
@group(1) @binding(3) var<storage, read> dense_shadow_casting_lights_buffer: array<u32>;
@group(1) @binding(4) var<storage, read> settings: ASVSMSettings;
@group(1) @binding(5) var<storage, read_write> physical_to_virtual_map: array<u32>;

// Constant for an invalid packed virtual coordinate, assuming 0,0 is valid.
// Max virtual coord is (tile_count-1, tile_count-1). If tile_count is e.g. 4096 (2^12),
// then max packed value is ((2^12-1)<<16) | (2^12-1) which is < 2^28. So 0xFFFFFFFF is safe.
const INVALID_PACKED_VIRT_COORD = 0xFFFFFFFFu;

@compute @workgroup_size(64, 4)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let count = requested_tiles[0u]; // Total number of requests
    if (idx >= count) {
        return;
    }

    let base_idx = 1u + idx * 3u;
    let tile_id_new = requested_tiles[base_idx];
    let light_idx = requested_tiles[base_idx + 1u];

    let dense_shadow_casting_light_idx = dense_shadow_casting_lights_buffer[light_idx];

    let new_pte_coords = vsm_pte_get_tile_coords(tile_id_new, settings);

    let current_pte_val_at_new_coords = textureLoad(page_table, new_pte_coords.xy, dense_shadow_casting_light_idx).r;
    let current_pte_is_valid = vsm_pte_is_valid(current_pte_val_at_new_coords);
    if (current_pte_is_valid) {
        return; // Already mapped by a concurrent thread or previous pass
    }

    let lru_per_light = u32(settings.physical_tiles_per_row
      * settings.physical_tiles_per_row
      * settings.max_lods);
    let lru_offset = dense_shadow_casting_light_idx * (lru_per_light + 1u);

    let lru_head = atomicAdd(&lru[lru_offset], 1u);
    let lru_slot_index = lru_offset + (lru_head % lru_per_light);
    let physical_id_to_reuse = atomicLoad(&lru[lru_slot_index]);

    // Evict Old PTE using the reverse map
    let packed_old_vx_vy = physical_to_virtual_map[physical_id_to_reuse];

    if (packed_old_vx_vy != INVALID_PACKED_VIRT_COORD) { // Check if the physical tile was actually mapped
        let old_vx = packed_old_vx_vy & 0xFFFFu;
        let old_vy = (packed_old_vx_vy >> 16u) & 0xFFFFu;
        let old_pte_coords = vec2<i32>(i32(old_vx), i32(old_vy));

        // Sanity check: ensure the old PTE indeed points to the physical_id_to_reuse
        let old_pte_val_check = textureLoad(page_table, old_pte_coords, dense_shadow_casting_light_idx).r;
        let old_pte_phys_id_check = old_pte_val_check & 0x07FFFFFFu; // Assuming 27 bits for physical ID
        let old_pte_valid_check = vsm_pte_is_valid(old_pte_val_check);

        if (old_pte_valid_check && old_pte_phys_id_check == physical_id_to_reuse) {
            textureStore(page_table, old_pte_coords, dense_shadow_casting_light_idx, vec4<u32>(0u)); // Invalidate old PTE
        }
    }

    // Update New PTE
    let new_pte_value = 0x80000000u | (new_pte_coords.z << 27u) | (physical_id_to_reuse & 0x07FFFFFFu);
    textureStore(page_table, new_pte_coords.xy, dense_shadow_casting_light_idx, vec4<u32>(new_pte_value));

    // Update Reverse Map
    let new_packed_vx_vy = (new_pte_coords.y << 16u) | new_pte_coords.x;
    physical_to_virtual_map[physical_id_to_reuse] = new_packed_vx_vy;
} 