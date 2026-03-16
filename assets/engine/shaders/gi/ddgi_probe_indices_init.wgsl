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

// Packed active flags: bit 0 = non-culled, bit 1 = culled
@group(1) @binding(2) var<storage, read> active_flags: array<u32>;

// Non-culled probe data (visible in frustum)
@group(1) @binding(3) var<storage, read> prefix_sum_nonculled: array<u32>;
@group(1) @binding(4) var<storage, read> block_prefixes_nonculled: array<u32>;

// Culled probe data (outside frustum)
@group(1) @binding(5) var<storage, read> prefix_sum_culled: array<u32>;
@group(1) @binding(6) var<storage, read> block_prefixes_culled: array<u32>;

// Counters containing total nonculled/culled counts
@group(1) @binding(7) var<storage, read> gi_counters: GICountersReadOnly;


// =============================================================================
// MAIN COMPUTE SHADER
// =============================================================================

@compute @workgroup_size(256, 1, 1)
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
    let total_nonculled = gi_counters.ray_queue_shadow_head;
    let total_culled = gi_counters.ray_queue_primary_head;

    // ─────────────────────────────────────────────────────────────────────────
    // Get the probe index from the permuted slot using CPU-precomputed params
    // (Eliminates expensive GCD computation that was previously done per-thread)
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

    let packed_flags = active_flags[slot];
    let is_nonculled = (packed_flags & 1u) != 0u;
    let is_culled = (packed_flags & 2u) != 0u;

    // ─────────────────────────────────────────────────────────────────────
    // Scatter nonculled probes into [0, nonculled_budget)
    // ─────────────────────────────────────────────────────────────────────
    if (is_nonculled && global_prefix_nonculled < nonculled_budget) {
        probe_update_indices[global_prefix_nonculled] = probe_index;
    }

    // ─────────────────────────────────────────────────────────────────────
    // Scatter culled probes into [nonculled_budget, nonculled_budget + culled_budget)
    // ─────────────────────────────────────────────────────────────────────
    if (is_culled && global_prefix_culled < culled_budget) {
        let output_slot = nonculled_budget + global_prefix_culled;
        probe_update_indices[output_slot] = probe_index;
    }
}
