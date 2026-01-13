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
    
    if (gid.x >= probe_count) {
        return;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Read probe state and check if it should be traced
    // ─────────────────────────────────────────────────────────────────────────
    let state_data = probe_state_read(&probe_states, gid.x);
    let state = probe_state_get_state(state_data.packed_state);
    
    // Only include probes that should trace this frame
    if (probe_state_should_trace(state)) {
        // Atomically allocate a slot in the update indices array
        let slot = atomicAdd(&gi_counters.probe_update_count, 1u);
        probe_update_indices[slot] = gid.x;
    }

}


