// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                       DDGI ACTIVE PROBE MARK                              ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Pass 1 of active-only probe cycling with frustum culling priority:       ║
// ║  - Builds one "active flags" array over a deterministic permutation of    ║
// ║    probe indices: one u32 per slot with bit 0 = non-culled active,        ║
// ║    bit 1 = culled active.                                                  ║
// ║  - The permutation is frame-shifted so we cycle through the active set    ║
// ║    temporally without structured artifacts.                               ║
// ║                                                                           ║
// ║  Output:                                                                  ║
// ║  - active_flags[slot]: bit 0 set when permuted probe is active AND        ║
// ║    visible in frustum; bit 1 set when active AND culled (not in frustum)   ║
// ║                                                                           ║
// ║  This enables the scheduling system to prioritize visible probes while    ║
// ║  still updating culled probes stochastically.                            ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read> probe_states: array<ProbeStateData>;
@group(1) @binding(2) var<storage, read> probe_cull_flags: array<u32>;
@group(1) @binding(3) var<storage, read_write> active_flags: array<u32>;

// =============================================================================
// MAIN
// =============================================================================

@compute @workgroup_size(256, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let probe_count = u32(ddgi_params.probe_counts.x);
    let frame_index_u32 = u32(ddgi_params.frame_index);

    let slot = gid.x;
    if (slot >= probe_count) {
        return;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Get the probe index from the permuted slot
    // ─────────────────────────────────────────────────────────────────────────
    let probe_index = ddgi_probe_index_from_permuted_slot(
        slot,
        probe_count,
        frame_index_u32,
        u32(ddgi_params.permutation_stride),
        u32(ddgi_params.permutation_base_offset),
        u32(ddgi_params.permutation_frame_stride)
    );

    // ─────────────────────────────────────────────────────────────────────────
    // Check if probe is active (based on probe state)
    // ─────────────────────────────────────────────────────────────────────────
    let state = probe_state_get_state(probe_states[probe_index].packed_state);
    let is_state_active = probe_state_is_active(state);
    
    // ─────────────────────────────────────────────────────────────────────────
    // Clipmap selection: only update probes in their cascade's "shell"
    // For cascade N > 0, skip probes that fall within cascade N-1's bounds
    // ─────────────────────────────────────────────────────────────────────────
    let is_in_cascade = ddgi_probe_in_cascade(&ddgi_params, probe_index);
    let is_active = is_state_active && is_in_cascade;

    // ─────────────────────────────────────────────────────────────────────────
    // Check if probe is in frustum and visible (from packed cull buffer)
    // ─────────────────────────────────────────────────────────────────────────
    let cull_word = probe_cull_flags[probe_index / 32u];
    let cull_bit = (probe_index % 32u);
    let is_culled = ((cull_word >> cull_bit) & 1u) == 0u;

    // ─────────────────────────────────────────────────────────────────────────
    // Pack both flags into a single u32: bit 0 = non-culled active, bit 1 = culled active
    // ─────────────────────────────────────────────────────────────────────────
    let nonculled_bit = select(0u, 1u, is_active && !is_culled);
    let culled_bit = select(0u, 2u, is_active && is_culled);
    active_flags[slot] = nonculled_bit | culled_bit;
}
