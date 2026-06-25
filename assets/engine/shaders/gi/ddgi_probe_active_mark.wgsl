// =============================================================================
// DDGI Active Probe Mark
// Builds one active flag per permuted probe slot.
// The active set is the current-frame depth-derived surface flag buffer.
// =============================================================================

#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read_write> probe_states: array<ProbeStateData>;
@group(1) @binding(2) var<storage, read_write> probe_surface_flags: array<u32>;
@group(1) @binding(3) var<storage, read_write> active_flags: array<u32>;

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let probe_count = u32(ddgi_params.probe_counts.x);
    let slot = gid.x;
    if (slot >= probe_count) {
        return;
    }

    let probe_index = ddgi_probe_index_from_permuted_slot(
        slot,
        probe_count,
        u32(ddgi_params.frame_index),
        u32(ddgi_params.permutation_stride),
        u32(ddgi_params.permutation_base_offset),
        u32(ddgi_params.permutation_frame_stride)
    );

    let packed = probe_states[probe_index].packed_state;
    let state = probe_state_get_state(packed);
    let flags = probe_state_get_flags(packed);
    let is_surface_active = probe_surface_flags[probe_index] != 0u;
    if (!is_surface_active || !ddgi_probe_in_cascade(&ddgi_params, probe_index)) {
        active_flags[slot] = 0u;
    } else {
        active_flags[slot] = 1u;
        if (state == PROBE_STATE_SLEEPING || state == PROBE_STATE_OFF) {
            probe_states[probe_index].packed_state = probe_state_pack(PROBE_STATE_UNINITIALIZED, 0u, 0u, flags);
            probe_states[probe_index].sample_count = 0u;
        }
    }
}
