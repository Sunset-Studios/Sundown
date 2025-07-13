#include "common.wgsl"
#include "shadow/shadows_common.wgsl"

@group(1) @binding(0) var page_table: texture_storage_2d_array<r32uint, read_write>;
@group(1) @binding(1) var<storage, read_write> shadow_atlas_depth: array<atomic<u32>>;
@group(1) @binding(2) var<uniform> vsm_settings: ASVSMSettings;

// Each workgroup clears one tile if it's dirty.
// Workgroup size is chosen to be a reasonable divisor of common tile sizes.
@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    #if SHADOWS_ENABLED
    // workgroup_id (wg_id) corresponds to a virtual tile's coordinates and slice.
    let pte_coords = wg_id.xy;
    let slice_idx = wg_id.z;

    let pte = textureLoad(page_table, pte_coords, slice_idx);

    // Only perform the memory write if the tile is dirty.
    if (vsm_pte_is_dirty(pte.r)) {
        // This is a dirty tile, so the workgroup will clear it.
        let phys_page_xy = vsm_pte_get_phys_xy(pte.r);
        let pool_idx     = vsm_pte_get_memory_pool_index(pte.r);
        
        let tile_size    = u32(vsm_settings.tile_size);
        let phys_dim     = u32(vsm_settings.physical_dim);

        // Each invocation is responsible for clearing multiple pixels in the tile.
        let invocation_index = local_id.y * 8u + local_id.x; // A value from 0-63
        let pixels_per_invocation = (tile_size * tile_size) / 64u;

        for (var i = 0u; i < pixels_per_invocation; i = i + 1u) {
            let pixel_index_in_tile = invocation_index * pixels_per_invocation + i;
            let pixel_in_tile_x = pixel_index_in_tile % tile_size;
            let pixel_in_tile_y = pixel_index_in_tile / tile_size;
            let target_pixel = phys_page_xy * tile_size + vec2<u32>(pixel_in_tile_x, pixel_in_tile_y);

            let slice_offset = pool_idx * phys_dim * phys_dim;
            let linear_index = slice_offset + target_pixel.y * phys_dim + target_pixel.x;
            atomicStore(&shadow_atlas_depth[linear_index], 0u);
        }
    }
    #endif
} 