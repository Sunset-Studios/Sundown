// =============================================================================
// DDGI Probe Surface Feedback Clear
// Clears the transient current-frame surface-active flags.
// =============================================================================

#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read_write> probe_surface_flags: array<u32>;

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let probe_count = u32(ddgi_params.probe_counts.x);
    let probe_index = gid.x;
    if (probe_index >= probe_count) {
        return;
    }

    probe_surface_flags[probe_index] = 0u;
}
