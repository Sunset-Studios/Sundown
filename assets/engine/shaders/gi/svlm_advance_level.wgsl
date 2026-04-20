#include "gi/svlm_common.wgsl"

// Frontier advance pass.
//
// Classification appends children into next_nodes with atomics. The JS render
// graph swaps the current/next buffer handles between passes; this tiny pass
// only publishes the next frontier count and resets the append counter.

@group(1) @binding(0) var<storage, read_write> svlm_counters: SVLMCounters;

@compute @workgroup_size(1, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }

    let next_count = atomicLoad(&svlm_counters.next_count);
    atomicStore(&svlm_counters.curr_count, next_count);
    atomicStore(&svlm_counters.next_count, 0u);
    atomicAdd(&svlm_counters.current_level, 1u);
}
