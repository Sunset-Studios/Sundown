#include "common.wgsl"
#include "shadow/shadows_common.wgsl"

struct DispatchArgs {
    workgroup_count_x: atomic<u32>,
    workgroup_count_y: u32,
    workgroup_count_z: u32,
};

@group(1) @binding(0) var page_table: texture_storage_2d_array<r32uint, read>;
@group(1) @binding(1) var<uniform> vsm_settings: ASVSMSettings;
@group(1) @binding(2) var<storage, read_write> dirty_tiles: array<vec2<u32>>;
@group(1) @binding(3) var<storage, read_write> dispatch_args: array<DispatchArgs>;

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
#if SHADOWS_ENABLED
    let page_table_dims = textureDimensions(page_table);
    if (gid.x >= page_table_dims.x || gid.y >= page_table_dims.y) {
        return;
    }

    let pte = textureLoad(page_table, gid.xy, gid.z).r;
    if (!vsm_pte_is_resident(pte) || !vsm_pte_is_dirty(pte)) {
        return;
    }

    let tiles_per_row = u32(vsm_settings.virtual_tiles_per_row);
    let tiles_per_slice = tiles_per_row * tiles_per_row;
    let tile_offset = atomicAdd(&dispatch_args[gid.z].workgroup_count_x, 1u);
    if (tile_offset >= tiles_per_slice) {
        return;
    }

    let dst = gid.z * tiles_per_slice + tile_offset;
    dirty_tiles[dst] = vec2<u32>(gid.y * tiles_per_row + gid.x, pte);
#endif
}
