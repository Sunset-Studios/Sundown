// =============================================================================
// DDGI Probe Indirection Reset
// - Initializes probe indirection to INVALID and seeds the free list.
// =============================================================================
#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read_write> probe_states: array<ProbeStateData>;
@group(1) @binding(2) var<storage, read_write> probe_free_list: DDGIProbeIndirectionFreeList;

@compute @workgroup_size(256, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let probe_count = u32(ddgi_params.probe_counts.x);
    let capacity = u32(ddgi_params.probe_storage_capacity);

    if (gid.x < probe_count) {
        probe_states[gid.x].sparse_index = INVALID_IDX;
    }

    if (gid.x < capacity) {
        probe_free_list.entries[gid.x] = gid.x;
    }

    if (gid.x == 0u) {
        atomicStore(&probe_free_list.counter, capacity);
    }
}
