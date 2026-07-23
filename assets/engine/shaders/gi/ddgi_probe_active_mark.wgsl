// =============================================================================
// DDGI Active Probe Mark
// Assigns scheduler priority to the current-frame depth-visible candidate set.
// Non-visible probes are not candidates, regardless of their priority.
// =============================================================================

#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read_write> probe_states: array<ProbeStateData>;
@group(1) @binding(2) var<storage, read_write> probe_surface_flags: array<u32>;
@group(1) @binding(3) var<storage, read_write> candidate_priorities: array<u32>;
@group(1) @binding(4) var<storage, read> probe_depth_slots: array<u32>;
@group(1) @binding(5) var<storage, read_write> depth_slot_last_used: array<u32>;

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
    let convergence_frames = probe_state_get_convergence_frames(packed);
    let flags = probe_state_get_flags(packed);
    let is_surface_active = probe_surface_flags[probe_index] != 0u;
    if (!is_surface_active || !ddgi_probe_in_cascade_shell(&ddgi_params, probe_index)) {
        candidate_priorities[slot] = DDGI_PROBE_SCHEDULE_PRIORITY_NONE;
    } else {
        let encoded_depth_slot = probe_depth_slots[probe_index];
        if (encoded_depth_slot != 0u) {
            depth_slot_last_used[encoded_depth_slot - 1u] = u32(ddgi_params.frame_index);
        }

        var schedule_state = state;
        var schedule_convergence_frames = convergence_frames;
        if (state == PROBE_STATE_SLEEPING || state == PROBE_STATE_OFF) {
            schedule_state = PROBE_STATE_UNINITIALIZED;
            schedule_convergence_frames = 0u;
            probe_states[probe_index].packed_state = probe_state_pack(schedule_state, 0u, 0u, flags);
            probe_states[probe_index].sample_count = 0u;
        }
        candidate_priorities[slot] = select(
            DDGI_PROBE_SCHEDULE_PRIORITY_FRESH,
            ddgi_probe_schedule_priority_for_state(schedule_state, schedule_convergence_frames),
            encoded_depth_slot != 0u
        );
    }
}
