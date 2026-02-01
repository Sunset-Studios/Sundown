// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                 DDGI Active Probe Block Prefix Scan                       ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Pass 3 of active-only probe cycling with frustum culling priority:       ║
// ║  - Computes exclusive prefix scan over block_sums for BOTH categories     ║
// ║    (nonculled and culled) in a single pass.                               ║
// ║  - Publishes total counts for scheduling logic.                           ║
// ║                                                                           ║
// ║  Output:                                                                  ║
// ║  - block_prefixes_nonculled[b] = sum of nonculled active in blocks [0..b) ║
// ║  - block_prefixes_culled[b] = sum of culled active in blocks [0..b)       ║
// ║  - gi_counters.probe_update_count = min(total_active, probes_per_frame)   ║
// ║  - gi_counters.nonculled_probe_count = total nonculled active probes      ║
// ║  - gi_counters.culled_probe_count = total culled active probes            ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

@group(1) @binding(0) var<storage, read> block_sums_nonculled_in: array<u32>;
@group(1) @binding(1) var<storage, read> block_sums_culled_in: array<u32>;
@group(1) @binding(2) var<storage, read_write> block_prefixes_nonculled_out: array<u32>;
@group(1) @binding(3) var<storage, read_write> block_prefixes_culled_out: array<u32>;
@group(1) @binding(4) var<storage, read_write> gi_counters: GICounters;
@group(1) @binding(5) var<uniform> ddgi_params: DDGIParams;

const WORKGROUP_SIZE = 256u;

var<workgroup> shared_subgroup_sums_nonculled: array<u32, 32>;
var<workgroup> shared_subgroup_sums_culled: array<u32, 32>;
var<workgroup> shared_running_offset_nonculled: u32;
var<workgroup> shared_running_offset_culled: u32;
var<workgroup> shared_chunk_total_nonculled: u32;
var<workgroup> shared_chunk_total_culled: u32;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn cs(
    @builtin(local_invocation_id) lid: vec3<u32>,
#if HAS_SUBGROUPS
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_size) sg_size: u32
#endif
) {
    let local_idx = lid.x;
    let num_blocks = arrayLength(&block_sums_nonculled_in);

    // Initialize running offsets
    if (local_idx == 0u) {
        shared_running_offset_nonculled = 0u;
        shared_running_offset_culled = 0u;
    }
    workgroupBarrier();

    // Process all blocks in chunks of WORKGROUP_SIZE
    for (var chunk_base = 0u; chunk_base < num_blocks; chunk_base = chunk_base + WORKGROUP_SIZE) {
        let chunk_idx = chunk_base + local_idx;
        let value_nonculled = select(0u, block_sums_nonculled_in[chunk_idx], chunk_idx < num_blocks);
        let value_culled = select(0u, block_sums_culled_in[chunk_idx], chunk_idx < num_blocks);

#if HAS_SUBGROUPS
        let warp_ctx = make_warp_ctx(local_idx, sg_lane, sg_size);

        // ─────────────────────────────────────────────────────────────────────
        // Process nonculled
        // ─────────────────────────────────────────────────────────────────────
        let subgroup_exclusive_nonculled = warp_scan_exclusive_add_u32(warp_ctx, value_nonculled);
        let subgroup_total_nonculled = warp_reduce_add_u32(warp_ctx, value_nonculled);

        if (is_warp_leader(warp_ctx)) {
            shared_subgroup_sums_nonculled[warp_ctx.warp_id] = subgroup_total_nonculled;
        }
        workgroupBarrier();

        var prefix_from_prev_sg_nonculled = 0u;
        for (var s = 0u; s < warp_ctx.warp_id; s = s + 1u) {
            prefix_from_prev_sg_nonculled = prefix_from_prev_sg_nonculled + shared_subgroup_sums_nonculled[s];
        }

        let chunk_exclusive_nonculled = subgroup_exclusive_nonculled + prefix_from_prev_sg_nonculled;

        // ─────────────────────────────────────────────────────────────────────
        // Process culled
        // ─────────────────────────────────────────────────────────────────────
        let subgroup_exclusive_culled = warp_scan_exclusive_add_u32(warp_ctx, value_culled);
        let subgroup_total_culled = warp_reduce_add_u32(warp_ctx, value_culled);

        if (is_warp_leader(warp_ctx)) {
            shared_subgroup_sums_culled[warp_ctx.warp_id] = subgroup_total_culled;
        }
        workgroupBarrier();

        var prefix_from_prev_sg_culled = 0u;
        for (var s = 0u; s < warp_ctx.warp_id; s = s + 1u) {
            prefix_from_prev_sg_culled = prefix_from_prev_sg_culled + shared_subgroup_sums_culled[s];
        }

        let chunk_exclusive_culled = subgroup_exclusive_culled + prefix_from_prev_sg_culled;

        // Compute chunk totals
        if (local_idx == 0u) {
            var chunk_total_nonculled = 0u;
            var chunk_total_culled = 0u;
            let subgroup_count = (WORKGROUP_SIZE + sg_size - 1u) / sg_size;
            for (var ss = 0u; ss < subgroup_count; ss = ss + 1u) {
                chunk_total_nonculled = chunk_total_nonculled + shared_subgroup_sums_nonculled[ss];
                chunk_total_culled = chunk_total_culled + shared_subgroup_sums_culled[ss];
            }
            shared_chunk_total_nonculled = chunk_total_nonculled;
            shared_chunk_total_culled = chunk_total_culled;
        }
        workgroupBarrier();

#else
        let lane = lane_id(local_idx, LOGICAL_WARP_SIZE);
        let warp_id_local = warp_id(local_idx, LOGICAL_WARP_SIZE);
        let warp_ctx = make_warp_ctx(local_idx, lane, LOGICAL_WARP_SIZE);

        // ─────────────────────────────────────────────────────────────────────
        // Process nonculled
        // ─────────────────────────────────────────────────────────────────────
        let warp_exclusive_nonculled = warp_scan_exclusive_add_u32(warp_ctx, value_nonculled);
        let warp_total_nonculled = warp_reduce_add_u32(warp_ctx, value_nonculled);

        if (is_warp_leader(warp_ctx)) {
            shared_subgroup_sums_nonculled[warp_id_local] = warp_total_nonculled;
        }
        workgroupBarrier();

        var prefix_from_prev_warps_nonculled = 0u;
        for (var w = 0u; w < warp_id_local; w = w + 1u) {
            prefix_from_prev_warps_nonculled = prefix_from_prev_warps_nonculled + shared_subgroup_sums_nonculled[w];
        }

        let chunk_exclusive_nonculled = warp_exclusive_nonculled + prefix_from_prev_warps_nonculled;

        // ─────────────────────────────────────────────────────────────────────
        // Process culled
        // ─────────────────────────────────────────────────────────────────────
        let warp_exclusive_culled = warp_scan_exclusive_add_u32(warp_ctx, value_culled);
        let warp_total_culled = warp_reduce_add_u32(warp_ctx, value_culled);

        if (is_warp_leader(warp_ctx)) {
            shared_subgroup_sums_culled[warp_id_local] = warp_total_culled;
        }
        workgroupBarrier();

        var prefix_from_prev_warps_culled = 0u;
        for (var w = 0u; w < warp_id_local; w = w + 1u) {
            prefix_from_prev_warps_culled = prefix_from_prev_warps_culled + shared_subgroup_sums_culled[w];
        }

        let chunk_exclusive_culled = warp_exclusive_culled + prefix_from_prev_warps_culled;

        // Compute chunk totals
        if (local_idx == 0u) {
            var chunk_total_nonculled = 0u;
            var chunk_total_culled = 0u;
            let warp_count = (WORKGROUP_SIZE + LOGICAL_WARP_SIZE - 1u) / LOGICAL_WARP_SIZE;
            for (var ww = 0u; ww < warp_count; ww = ww + 1u) {
                chunk_total_nonculled = chunk_total_nonculled + shared_subgroup_sums_nonculled[ww];
                chunk_total_culled = chunk_total_culled + shared_subgroup_sums_culled[ww];
            }
            shared_chunk_total_nonculled = chunk_total_nonculled;
            shared_chunk_total_culled = chunk_total_culled;
        }
        workgroupBarrier();
#endif

        // ─────────────────────────────────────────────────────────────────────
        // Write block prefixes with running offset
        // ─────────────────────────────────────────────────────────────────────
        let running_offset_nonculled = shared_running_offset_nonculled;
        let running_offset_culled = shared_running_offset_culled;

        if (chunk_idx < num_blocks) {
            block_prefixes_nonculled_out[chunk_idx] = running_offset_nonculled + chunk_exclusive_nonculled;
            block_prefixes_culled_out[chunk_idx] = running_offset_culled + chunk_exclusive_culled;
        }
        workgroupBarrier();

        // Update running offsets for next chunk
        if (local_idx == 0u) {
            shared_running_offset_nonculled = running_offset_nonculled + shared_chunk_total_nonculled;
            shared_running_offset_culled = running_offset_culled + shared_chunk_total_culled;
        }
        workgroupBarrier();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Write final counts to GI counters (only thread 0)
    // ─────────────────────────────────────────────────────────────────────────
    if (local_idx == 0u) {
        let total_nonculled_probes = shared_running_offset_nonculled;
        let total_culled_probes = shared_running_offset_culled;
        let total_active_probes = total_nonculled_probes + total_culled_probes;
        let probes_per_frame = u32(ddgi_params.probe_counts.z);

        // Store the total count for scheduling logic
        let update_count = min(total_active_probes, probes_per_frame);
        atomicStore(&gi_counters.probe_update_count, update_count);

        // Store separate counts for nonculled and culled (repurpose existing atomic slots)
        // We use ray_queue_shadow_head for nonculled count and ray_queue_primary_head for culled
        // These are repurposed during DDGI passes when they're not used for ray queues
        atomicStore(&gi_counters.ray_queue_shadow_head, total_nonculled_probes);
        atomicStore(&gi_counters.ray_queue_primary_head, total_culled_probes);
    }
}
