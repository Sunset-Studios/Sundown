// =============================================================================
// GI-1.0 World Cache Block Prefix Scan
// - Computes an exclusive prefix scan over `block_sums` (one u32 per workgroup)
// - Produces `block_prefixes`, where block_prefixes[b] = sum(block_sums[0..b))
// - Writes total active count and indirect dispatch parameters for later passes
//
// NOTE:
// - This pass is intentionally "tiny": `block_sums` length is ~ total_cells / 128.
// - Doing this once avoids the O(num_blocks^2) recomputation that happens if each
//   scatter workgroup sums all previous block sums on its own.
// =============================================================================

#include "common.wgsl"
#include "gi/gi_common.wgsl"

@group(1) @binding(0) var<storage, read> block_sums_in: array<u32>;
@group(1) @binding(1) var<storage, read_write> block_prefixes_out: array<u32>;
@group(1) @binding(2) var<storage, read_write> dispatch_params: array<u32>; // [x, y, z, total_count]
@group(1) @binding(3) var<storage, read_write> gi_counters: GICounters;

const WORKGROUP_SIZE = 128u;

// Shared staging for cross-subgroup accumulation within the scan workgroup.
// Worst-case subgroup_size == 4 => 128 / 4 = 32 subgroups.
var<workgroup> shared_subgroup_sums: array<u32, 32>;
var<workgroup> shared_running_offset: u32;
var<workgroup> shared_chunk_total: u32;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn cs(
    @builtin(local_invocation_id) lid: vec3<u32>,
#if HAS_SUBGROUPS
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_size) sg_size: u32
#endif
) {
    let local_idx = lid.x;
    let num_blocks = arrayLength(&block_sums_in);

    if (local_idx == 0u) {
        shared_running_offset = 0u;
    }
    workgroupBarrier();

    // Chunked exclusive scan: processes WORKGROUP_SIZE elements per iteration.
    for (var chunk_base = 0u; chunk_base < num_blocks; chunk_base = chunk_base + WORKGROUP_SIZE) {
        let chunk_idx = chunk_base + local_idx;
        let value = select(0u, block_sums_in[chunk_idx], chunk_idx < num_blocks);

#if HAS_SUBGROUPS
        let warp_ctx = make_warp_ctx(local_idx, sg_lane, sg_size);
        let subgroup_exclusive = warp_scan_exclusive_add_u32(warp_ctx, value);
        let subgroup_total = warp_reduce_add_u32(warp_ctx, value);

        if (is_warp_leader(warp_ctx)) {
            shared_subgroup_sums[warp_ctx.warp_id] = subgroup_total;
        }
        workgroupBarrier();

        var prefix_from_previous_subgroups = 0u;
        for (var s = 0u; s < warp_ctx.warp_id; s = s + 1u) {
            prefix_from_previous_subgroups = prefix_from_previous_subgroups + shared_subgroup_sums[s];
        }

        let chunk_exclusive = subgroup_exclusive + prefix_from_previous_subgroups;

        if (local_idx == 0u) {
            var chunk_total = 0u;
            let subgroup_count = (WORKGROUP_SIZE + sg_size - 1u) / sg_size;
            for (var ss = 0u; ss < subgroup_count; ss = ss + 1u) {
                chunk_total = chunk_total + shared_subgroup_sums[ss];
            }
            shared_chunk_total = chunk_total;
        }
        workgroupBarrier();
#else
        // Fallback: logical warp scan/reduction
        let lane = lane_id(local_idx, LOGICAL_WARP_SIZE);
        let warp_id_local = warp_id(local_idx, LOGICAL_WARP_SIZE);
        let warp_ctx = make_warp_ctx(local_idx, lane, LOGICAL_WARP_SIZE);

        let warp_exclusive = warp_scan_exclusive_add_u32(warp_ctx, value);
        let warp_total = warp_reduce_add_u32(warp_ctx, value);

        if (is_warp_leader(warp_ctx)) {
            shared_subgroup_sums[warp_id_local] = warp_total;
        }
        workgroupBarrier();

        var prefix_from_previous_warps = 0u;
        for (var w = 0u; w < warp_id_local; w = w + 1u) {
            prefix_from_previous_warps = prefix_from_previous_warps + shared_subgroup_sums[w];
        }

        let chunk_exclusive = warp_exclusive + prefix_from_previous_warps;

        if (local_idx == 0u) {
            var chunk_total = 0u;
            let warp_count = (WORKGROUP_SIZE + LOGICAL_WARP_SIZE - 1u) / LOGICAL_WARP_SIZE;
            for (var ww = 0u; ww < warp_count; ww = ww + 1u) {
                chunk_total = chunk_total + shared_subgroup_sums[ww];
            }
            shared_chunk_total = chunk_total;
        }
        workgroupBarrier();
#endif

        let running_offset = shared_running_offset;

        if (chunk_idx < num_blocks) {
            // Exclusive prefix for this block (global, across all chunks)
            block_prefixes_out[chunk_idx] = running_offset + chunk_exclusive;
        }
        workgroupBarrier();

        if (local_idx == 0u) {
            shared_running_offset = running_offset + shared_chunk_total;
        }
        workgroupBarrier();
    }

    // Publish totals + dispatch params once
    if (local_idx == 0u) {
        let total_count = shared_running_offset;

        atomicStore(&gi_counters.active_cache_cell_count, total_count);

        // Write dispatch parameters:
        // - X is number of workgroups for later ray passes (2x: shadow + primary)
        // - YZ are 1
        dispatch_params[0] = (2u * total_count + (WORKGROUP_SIZE - 1u)) / WORKGROUP_SIZE; // ceil_div
        dispatch_params[1] = 1u;
        dispatch_params[2] = 1u;
        dispatch_params[3] = total_count;
    }
}


