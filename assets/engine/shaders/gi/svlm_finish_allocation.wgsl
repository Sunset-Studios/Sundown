#include "gi/svlm_common.wgsl"

// Publishes that the hierarchy and leaf allocation passes have finished.
// The CPU waits for this bit through the existing counter readback before it
// creates the persistent irradiance buffer at the realized probe count.

@group(1) @binding(0) var<storage, read_write> svlm_counters: SVLMCounters;

const SVLM_IRRADIANCE_STATUS_ALLOCATION_COMPLETE = 1u << 1u;

@compute @workgroup_size(1, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }

    atomicOr(
        &svlm_counters.irradiance_status,
        SVLM_IRRADIANCE_STATUS_ALLOCATION_COMPLETE
    );
}
