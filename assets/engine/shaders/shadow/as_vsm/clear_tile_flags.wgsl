#include "common.wgsl"
#include "shadow/shadows_common.wgsl"

@group(1) @binding(0) var page_table: texture_storage_2d_array<r32uint, read_write>;

@compute @workgroup_size(16, 16, 1)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    #if SHADOWS_ENABLED
    let pte_coords = global_id.xy;
    let slice_idx = global_id.z;
    var pte = textureLoad(page_table, pte_coords, slice_idx).r;

    if (vsm_pte_is_dirty(pte)) {
        let frame_age = vsm_pte_get_frame_age(pte);
        let clear_immediately = !vsm_pte_should_linger_dirty(pte);
        let clear_after_linger = frame_age >= as_vsm_dirty_clear_min_frame_age;
        if (clear_immediately || clear_after_linger) {
            pte = vsm_pte_clear_dirty(pte);
        }
    }

    textureStore(page_table, pte_coords, slice_idx, vec4<u32>(pte));
    #endif
} 