// =============================================================================
// DDGI Probe Ray Trace - Init Pass
// - Publishes the active ray count after probe scheduling
// - Per-ray initialization and direction generation happen in the hit pass so
//   the transient buffer only stores data that survives that pass
// =============================================================================
#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read_write> probe_ray_data: DDGIProbeRayDataBuffer;
@group(1) @binding(2) var<storage, read> gi_counters: GICountersReadOnly;

// =============================================================================
// Main
// =============================================================================
@compute @workgroup_size(1, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let max_probes_per_frame = u32(ddgi_params.probe_counts.z);
    let active_probe_count = min(gi_counters.probe_update_count, max_probes_per_frame);
    let rays_per_probe = ddgi_max_rays_per_probe(&ddgi_params);

    if (gid.x == 0u) {
        atomicStore(&probe_ray_data.header.active_ray_count, active_probe_count * rays_per_probe);
    }
}
