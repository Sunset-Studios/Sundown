#include "common.wgsl"
#include "shadow/shadows_common.wgsl"

@group(1) @binding(0) var page_table: texture_storage_2d_array<r32uint, read_write>;

@compute @workgroup_size(16, 16, 1)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    #if SHADOWS_ENABLED
    let pte_coords = global_id.xy;
    let slice_idx = global_id.z;
    var pte = textureLoad(page_table, pte_coords, slice_idx).r;

    let linger = vsm_pte_get_dirty_linger(pte);
    if (linger > 0u) {
        pte = vsm_pte_set_dirty_linger(pte, linger - 1u);
    } else {
        pte = vsm_pte_set_dirty_linger(pte & ~pte_dirty_mask, 0u);
    }

    textureStore(page_table, pte_coords, slice_idx, vec4<u32>(pte));
    #endif
} 