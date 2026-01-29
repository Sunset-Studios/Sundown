// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                    DDGI PROBE INDICES INITIALIZATION                      ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Builds the list of probes to update this frame with frustum culling      ║
// ║  balance. The per-frame budget is split between non-culled and culled      ║
// ║  probes using a configurable ratio, then backfilled if either bucket      ║
// ║  runs out of active probes.                                               ║
// ║                                                                           ║
// ║  Budget Strategy:                                                         ║
// ║  ┌─────────────────────────────────────────────────────────────────────┐  ║
// ║  │ 1) Allocate culled vs non-culled using probe_update_culled_ratio    │  ║
// ║  │ 2) Clamp each allocation to active counts                           │  ║
// ║  │ 3) Backfill remaining slots from the other bucket                   │  ║
// ║  └─────────────────────────────────────────────────────────────────────┘  ║
// ║                                                                           ║
// ║  The stochastic selection uses a frame-shifted permutation to ensure      ║
// ║  temporal coverage of all probes while avoiding structured artifacts.     ║
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

// Non-culled probe data (visible in frustum)
@group(1) @binding(2) var<storage, read> active_flags_nonculled: array<u32>;
@group(1) @binding(3) var<storage, read> prefix_sum_nonculled: array<u32>;
@group(1) @binding(4) var<storage, read> block_prefixes_nonculled: array<u32>;

// Culled probe data (outside frustum)
@group(1) @binding(5) var<storage, read> active_flags_culled: array<u32>;
@group(1) @binding(6) var<storage, read> prefix_sum_culled: array<u32>;
@group(1) @binding(7) var<storage, read> block_prefixes_culled: array<u32>;

// Counters containing total nonculled/culled counts
@group(1) @binding(8) var<storage, read_write> gi_counters: GICounters;

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
// Notes:
// - `offset` and `stride` are derived from a hash of `probe_count` to make the
//   permutation "stochastic" but still deterministic for a given configuration.
// - `stride` is forced to be coprime with n using a small bounded search.

fn ddgi_gcd_u32(a: u32, b: u32) -> u32 {
    var x = a;
    var y = b;

    // Bounded Euclid to keep shader compilers happy.
    for (var iter = 0u; iter < 32u; iter = iter + 1u) {
        if (y == 0u) {
            break;
        }
        let t = x % y;
        x = y;
        y = t;
    }

    return x;
}

fn ddgi_coprime_stride_from_seed(sequence_seed: u32, modulus: u32) -> u32 {
    // Degenerate cases: 0 or 1 probe -> any stride works; keep it simple.
    if (modulus <= 1u) {
        return 1u;
    }

    // Start with a hashed candidate in [1, modulus-1], bias to odd.
    // Odd isn't strictly required, but helps avoid easy common factors with powers-of-two.
    let range = modulus - 1u;
    var stride = (hash(sequence_seed) % range) + 1u;
    stride = stride | 1u;
    stride = ((stride - 1u) % range) + 1u;

    // Bounded search for a coprime stride.
    for (var iter = 0u; iter < 32u; iter = iter + 1u) {
        if (ddgi_gcd_u32(stride, modulus) == 1u) {
            break;
        }

        // Try the next odd. Wrap to stay in [1, modulus-1].
        stride = stride + 2u;
        stride = select(stride, stride % modulus, stride >= modulus);
        stride = select(stride, 1u, stride == 0u);
    }

    // Last-resort fallback: stride=1 always coprime and still cycles fully.
    return select(stride, 1u, ddgi_gcd_u32(stride, modulus) != 1u);
}

fn ddgi_probe_index_from_permuted_slot(slot: u32, probe_count: u32, frame_index_u32: u32) -> u32 {
    let safe_probe_count = max(probe_count, 1u);

    let sequence_seed = hash(probe_count ^ 0xA3C59AC3u);
    let stride = ddgi_coprime_stride_from_seed(sequence_seed, safe_probe_count);
    let base_offset = hash(sequence_seed ^ 0x85ebca6bu) % safe_probe_count;

    let frame_stride = ddgi_coprime_stride_from_seed(sequence_seed ^ 0xC2B2AE35u, safe_probe_count);
    let frame_shift = frame_index_u32 * frame_stride;
    let k = frame_shift + slot;
    return (base_offset + k * stride) % safe_probe_count;
}

// =============================================================================
// MAIN COMPUTE SHADER
// =============================================================================

@compute @workgroup_size(128, 1, 1)
fn cs(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let probe_count = u32(ddgi_params.probe_counts.x);
    let probes_per_frame = u32(ddgi_params.probe_counts.z);
    let frame_index_u32 = u32(ddgi_params.frame_index);

    let slot = gid.x;
    if (slot >= probe_count) {
        return;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Read total counts from GI counters
    // (ray_queue_shadow_head stores nonculled count, ray_queue_primary_head stores culled count)
    // ─────────────────────────────────────────────────────────────────────────
    let total_nonculled = atomicLoad(&gi_counters.ray_queue_shadow_head);
    let total_culled = atomicLoad(&gi_counters.ray_queue_primary_head);

    // ─────────────────────────────────────────────────────────────────────────
    // Get the probe index from the permuted slot
    // ─────────────────────────────────────────────────────────────────────────
    let probe_index = ddgi_probe_index_from_permuted_slot(slot, probe_count, frame_index_u32);

    // ─────────────────────────────────────────────────────────────────────────
    // Weighted Scheduling Logic
    // - probe_update_culled_ratio balances culled vs nonculled selection.
    // - Any unused budget is backfilled from the other group.
    // ─────────────────────────────────────────────────────────────────────────
    let culled_ratio = clamp(ddgi_params.probe_update_culled_ratio, 0.0, 1.0);
    let desired_culled_budget_f32 = f32(probes_per_frame) * culled_ratio;
    let desired_culled_budget = min(probes_per_frame, u32(desired_culled_budget_f32 + 0.5));

    var culled_budget = min(total_culled, desired_culled_budget);
    var remaining_budget = probes_per_frame - culled_budget;
    var nonculled_budget = min(total_nonculled, remaining_budget);
    remaining_budget = remaining_budget - nonculled_budget;
    culled_budget = min(total_culled, culled_budget + remaining_budget);

    let global_prefix_nonculled = prefix_sum_nonculled[slot] + block_prefixes_nonculled[wid.x];
    let global_prefix_culled = prefix_sum_culled[slot] + block_prefixes_culled[wid.x];

    // ─────────────────────────────────────────────────────────────────────
    // Scatter nonculled probes into [0, nonculled_budget)
    // ─────────────────────────────────────────────────────────────────────
    if (active_flags_nonculled[slot] != 0u && global_prefix_nonculled < nonculled_budget) {
        probe_update_indices[global_prefix_nonculled] = probe_index;
    }

    // ─────────────────────────────────────────────────────────────────────
    // Scatter culled probes into [nonculled_budget, nonculled_budget + culled_budget)
    // ─────────────────────────────────────────────────────────────────────
    if (active_flags_culled[slot] != 0u && global_prefix_culled < culled_budget) {
        let output_slot = nonculled_budget + global_prefix_culled;
        probe_update_indices[output_slot] = probe_index;
    }
}
