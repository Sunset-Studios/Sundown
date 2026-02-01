// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                       DDGI ACTIVE PROBE MARK                              ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Pass 1 of active-only probe cycling with frustum culling priority:       ║
// ║  - Builds two "active flag" arrays over a deterministic permutation of    ║
// ║    probe indices: one for non-culled active, one for culled active.       ║
// ║  - The permutation is frame-shifted so we cycle through the active set    ║
// ║    temporally without structured artifacts.                               ║
// ║                                                                           ║
// ║  Output:                                                                  ║
// ║  - active_flags_nonculled[slot] = 1 when permuted probe is active AND     ║
// ║    visible in frustum                                                     ║
// ║  - active_flags_culled[slot] = 1 when permuted probe is active AND        ║
// ║    culled (not in frustum)                                                ║
// ║                                                                           ║
// ║  This dual-output enables the scheduling system to prioritize visible     ║
// ║  probes while still updating culled probes stochastically.                ║
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
@group(1) @binding(2) var<storage, read_write> active_flags_nonculled: array<u32>;
@group(1) @binding(3) var<storage, read_write> active_flags_culled: array<u32>;
@group(1) @binding(4) var<storage, read_write> active_flags_by_index: array<u32>;

// =============================================================================
// STOCHASTIC (BUT DETERMINISTIC) PROBE CYCLING
// =============================================================================
// We want a selection pattern that:
// - Looks "random" to avoid structured artifacts (better temporal distribution)
// - Is deterministic (given probe_count and frame_index)
// - Does not miss probes: every probe index must be visited eventually
//
// Approach:
// - Treat the per-frame probe picks as a walk over Z_n (n = probe_count).
// - Use an affine map:   idx(k) = (offset + k * stride) mod n
// - If gcd(stride, n) = 1, this is a permutation: k=0..n-1 visits every probe once.
// - We set k = frame_index * probes_per_frame + local_id so the walk advances by
//   probes_per_frame each frame without gaps.
//
// OPTIMIZATION: The stride, base_offset, and frame_stride values are UNIFORM
// across all shader invocations (they only depend on probe_count). These are
// precomputed on the CPU and passed via DDGIParams, eliminating expensive
// GCD computation loops that previously ran per-thread.
//
// The permutation formula is:
//   probe_index = (base_offset + (frame_stride * frame_index + slot) * stride) % probe_count

fn ddgi_probe_index_from_permuted_slot(
    slot: u32,
    probe_count: u32,
    frame_index_u32: u32,
    stride: u32,
    base_offset: u32,
    frame_stride: u32
) -> u32 {
    let safe_probe_count = max(probe_count, 1u);
    let frame_shift = frame_index_u32 * frame_stride;
    let k = frame_shift + slot;
    return (base_offset + k * stride) % safe_probe_count;
}

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
    // Check if probe is in frustum and visible (from flags byte in packed_state)
    // ─────────────────────────────────────────────────────────────────────────
    let is_culled = !ddgi_probe_state_get_cull_visible(probe_states[probe_index].packed_state);

    // ─────────────────────────────────────────────────────────────────────────
    // Output to appropriate flag array based on culling status
    // - Non-culled active probes get priority in scheduling
    // - Culled active probes are scheduled stochastically to fill remaining budget
    // ─────────────────────────────────────────────────────────────────────────
    // Store culling-aware flags in permuted order (slot-space) for scheduling
    active_flags_nonculled[slot] = select(0u, 1u, is_active && !is_culled);
    active_flags_culled[slot] = select(0u, 1u, is_active && is_culled);
    active_flags_by_index[probe_index] = select(0u, 1u, is_active);
}
