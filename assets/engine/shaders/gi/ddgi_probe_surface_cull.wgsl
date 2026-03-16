// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                    DDGI PROBE SURFACE CULLING                             ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Marks probes as SLEEPING/OFF when their cell does not overlap geometry   ║
// ║                                                                           ║
// ║  This shader runs after probe tracing and marks probes as SLEEPING/OFF    ║
// ║  based on the probe surface visibility flag from DDGI sampling in the     ║
// ║  previous frame.                                                          ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read_write> probe_states: array<ProbeStateData>;
@group(1) @binding(2) var<storage, read> probe_cull_flags: array<u32>;

// =============================================================================
// MAIN COMPUTE SHADER
// =============================================================================

@compute @workgroup_size(256, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let probe_count = u32(ddgi_params.probe_counts.x);
    let frame_index_u32 = u32(ddgi_params.frame_index);

    let probe_index = gid.x;
    if (probe_index >= probe_count) {
        return;
    }

    let cull_word = probe_cull_flags[probe_index / 32u];
    let is_cull_visible = ((cull_word >> (probe_index % 32u)) & 1u) != 0u;
    if (!is_cull_visible) {
        return;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Read current probe state (using read_write version for modify pass)
    // ─────────────────────────────────────────────────────────────────────────
    var current_state = probe_state_get_state(probe_states[probe_index].packed_state);
    var init_frames = probe_state_get_init_frames(probe_states[probe_index].packed_state);
    var convergence_frames = probe_state_get_convergence_frames(probe_states[probe_index].packed_state);
    var flags = probe_state_get_flags(probe_states[probe_index].packed_state);

    // Immediately mark non-surface visible probes as sleeping
    if ((flags & PROBE_STATE_FLAG_SURFACE_VISIBLE) == 0u) {
        current_state = PROBE_STATE_SLEEPING;
        init_frames = 0u;
        convergence_frames = 0u;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Write updated state
    // ─────────────────────────────────────────────────────────────────────────
    probe_states[probe_index].packed_state = probe_state_pack(current_state, init_frames, convergence_frames, flags);
}

