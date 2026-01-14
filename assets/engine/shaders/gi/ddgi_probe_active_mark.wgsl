// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                       DDGI ACTIVE PROBE MARK                              ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Pass 1 of active-only probe cycling:                                     ║
// ║  - Builds an "active flag" array (0/1) over a deterministic permutation   ║
// ║    of probe indices.                                                     ║
// ║  - The permutation is frame-shifted so we cycle through the active set    ║
// ║    temporally without structured artifacts.                               ║
// ║                                                                           ║
// ║  Output: active_flags_permuted[i] == 1 when the permuted probe is active  ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read> probe_states: array<u32>;
@group(1) @binding(2) var<storage, read_write> active_flags: array<u32>;

// =============================================================================
// STOCHASTIC (BUT DETERMINISTIC) PERMUTATION HELPERS
// =============================================================================

fn ddgi_gcd_u32(a: u32, b: u32) -> u32 {
    var x = a;
    var y = b;
    for (var iter = 0u; iter < 32u; iter = iter + 1u) {
        if (y == 0u) { break; }
        let t = x % y;
        x = y;
        y = t;
    }
    return x;
}

fn ddgi_coprime_stride_from_seed(sequence_seed: u32, modulus: u32) -> u32 {
    if (modulus <= 1u) {
        return 1u;
    }

    let range = modulus - 1u;
    var stride = (hash(sequence_seed) % range) + 1u;
    stride = stride | 1u;
    stride = ((stride - 1u) % range) + 1u;

    for (var iter = 0u; iter < 32u; iter = iter + 1u) {
        if (ddgi_gcd_u32(stride, modulus) == 1u) {
            break;
        }
        stride = stride + 2u;
        stride = select(stride, stride % modulus, stride >= modulus);
        stride = select(stride, 1u, stride == 0u);
    }

    return select(stride, 1u, ddgi_gcd_u32(stride, modulus) != 1u);
}

fn ddgi_probe_index_from_permuted_slot(slot: u32, probe_count: u32, frame_index_u32: u32) -> u32 {
    let safe_probe_count = max(probe_count, 1u);

    // Stable sequence for a given grid configuration (good temporal stability).
    let sequence_seed = hash(probe_count ^ 0xA3C59AC3u);
    let stride = ddgi_coprime_stride_from_seed(sequence_seed, safe_probe_count);
    let base_offset = hash(sequence_seed ^ 0x85ebca6bu) % safe_probe_count;

    // Frame shift so we cycle through the permutation across frames.
    // Use a coprime "frame stride" so the start position visits all residues
    // regardless of probe_count and probes_per_frame relationships.
    let frame_stride = ddgi_coprime_stride_from_seed(sequence_seed ^ 0xC2B2AE35u, safe_probe_count);
    let frame_shift = frame_index_u32 * frame_stride;

    let k = frame_shift + slot;
    return (base_offset + k * stride) % safe_probe_count;
}

// =============================================================================
// MAIN
// =============================================================================

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let probe_count = u32(ddgi_params.probe_counts.x);
    let frame_index_u32 = u32(ddgi_params.frame_index);

    let slot = gid.x;
    if (slot >= probe_count) {
        return;
    }

    let probe_index = ddgi_probe_index_from_permuted_slot(slot, probe_count, frame_index_u32);

    let state_data = probe_state_read(&probe_states, probe_index);
    let state = probe_state_get_state(state_data.packed_state);

    // Store flag in permuted order (slot-space).
    active_flags[slot] = select(0u, 1u, probe_state_is_active(state));
}

