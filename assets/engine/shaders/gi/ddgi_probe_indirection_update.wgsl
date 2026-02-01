// =============================================================================
// DDGI Probe Indirection Update
// - Maintains a bounded sparse mapping for active probes.
// - Allocates new slots for newly active probes and releases inactive slots.
// =============================================================================
#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read_write> probe_states: array<ProbeStateData>;
@group(1) @binding(2) var<storage, read_write> probe_free_list: DDGIProbeIndirectionFreeList;
@group(1) @binding(3) var<storage, read_write> sh_probes: array<u32>;
@group(1) @binding(4) var<storage, read_write> probe_depth_moments: array<vec4<f32>>;

fn ddgi_probe_free_list_pop() -> u32 {
    loop {
        var old_count = atomicLoad(&probe_free_list.counter);
        if (old_count == 0u) {
            return INVALID_IDX;
        }

        let new_count = old_count - 1u;
        let result = atomicCompareExchangeWeak(&probe_free_list.counter, old_count, new_count);
        if (result.exchanged) {
            return probe_free_list.entries[new_count];
        }

        old_count = result.old_value;
    }
}

fn ddgi_probe_clear_sparse_slot(slot: u32) {
    let sh_base = slot * DDGI_SH_PROBE_SIZE_U32;
    for (var i = 0u; i < DDGI_SH_PROBE_SIZE_U32; i = i + 1u) {
        sh_probes[sh_base + i] = 0u;
    }

    let depth_base = slot * DDGI_DEPTH_TEXEL_COUNT;
    for (var texel = 0u; texel < DDGI_DEPTH_TEXEL_COUNT; texel = texel + 1u) {
        probe_depth_moments[depth_base + texel] = vec4<f32>(0.0);
    }
}

@compute @workgroup_size(256, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let probe_count = u32(ddgi_params.probe_counts.x);
    if (gid.x >= probe_count) {
        return;
    }

    let state = probe_state_get_state(probe_states[gid.x].packed_state);
    let is_state_active = probe_state_is_active(state);
    let is_in_cascade = ddgi_probe_in_cascade(&ddgi_params, gid.x);
    let is_active = is_state_active && is_in_cascade;
    let current_slot = probe_states[gid.x].sparse_index;

    // If the probe is not active and has an indirection slot, free it
    if (!is_active && current_slot != INVALID_IDX) {
        probe_states[gid.x].sparse_index = INVALID_IDX;
        let index = atomicAdd(&probe_free_list.counter, 1u);
        if (index < u32(ddgi_params.probe_storage_capacity)) {
            probe_free_list.entries[index] = current_slot;
        }
    // Otherwise if the probe is active and has no indirection slot, allocate a new one
    } else if (is_active && current_slot == INVALID_IDX) {
        let slot = ddgi_probe_free_list_pop();
        probe_states[gid.x].sparse_index = slot;
        ddgi_probe_clear_sparse_slot(slot);
    }
}
