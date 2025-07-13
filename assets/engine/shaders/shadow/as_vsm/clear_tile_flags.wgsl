#include "common.wgsl"
#include "shadow/shadows_common.wgsl"

@group(1) @binding(0) var page_table: texture_storage_2d_array<r32uint, read_write>;

@compute @workgroup_size(16, 16, 1)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    #if SHADOWS_ENABLED
    let pte_coords = global_id.xy;
    let slice_idx = global_id.z;
    let pte = textureLoad(page_table, pte_coords, slice_idx).r;
    let entry = pte & ~pte_dirty_mask;

    textureStore(page_table, pte_coords, slice_idx, vec4<u32>(entry));
    #endif
} 