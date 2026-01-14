// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                    DDGI PROBE INDICES INITIALIZATION                      ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Builds the list of probes to update this frame based on probe states.   ║
// ║  Only probes that should be traced (not OFF or SLEEPING) are included.   ║
// ║                                                                           ║
// ║  The output probe_update_indices array contains:                          ║
// ║  - Indices of all probes that need tracing                                ║
// ║  - Compacted via atomic counter for efficient dispatch                    ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read_write> probe_update_indices: array<u32>;
@group(1) @binding(2) var<storage, read> probe_states: array<u32>;
@group(1) @binding(3) var<storage, read_write> gi_counters: GICounters;

// =============================================================================
// MAIN COMPUTE SHADER
// =============================================================================

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let probe_count = u32(ddgi_params.probe_counts.x);
    let probes_per_frame = u32(ddgi_params.probe_counts.z);
    let candidate_count = min(probes_per_frame, probe_count);

    if (gid.x >= candidate_count) {
        return;
    }

    // -------------------------------------------------------------------------
    // Round-robin probe selection
    // -------------------------------------------------------------------------
    let frame_index_u32 = u32(ddgi_params.frame_index);
    let base_probe_index = frame_index_u32 * probes_per_frame;
    let probe_index = (base_probe_index + gid.x) % max(probe_count, 1u);

    // -------------------------------------------------------------------------
    // Read probe state and include only probes that should be traced
    // -------------------------------------------------------------------------
    let state_data = probe_state_read(&probe_states, probe_index);
    let state = probe_state_get_state(state_data.packed_state);

    if (probe_state_is_active(state)) {
        // Atomically allocate a slot in the update indices array.
        // Max writes per frame are bounded by `candidate_count <= probes_per_frame`.
        let slot = atomicAdd(&gi_counters.probe_update_count, 1u);
        probe_update_indices[slot] = probe_index;
    }

}


