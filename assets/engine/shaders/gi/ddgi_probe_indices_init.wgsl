// =============================================================================
// DDGI Probe Indices Init
// - Fills probe_update_indices with an identity mapping (i -> i)
// - Kept as a standalone pass so we can later replace it with a GPU reordering pass
//   and update only a prefix of indices each frame.
// =============================================================================
#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read_write> probe_update_indices: array<u32>;

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let probe_count = u32(ddgi_params.probe_counts.x);
    if (gid.x >= probe_count) {
        return;
    }
    probe_update_indices[gid.x] = gid.x;
}


