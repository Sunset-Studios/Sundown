// =============================================================================
// DDGI Active Probe Block Prefix Scan
// - Pass 3 of active-only probe cycling:
//   Computes an exclusive prefix scan over block_sums (one u32 per workgroup)
//   and publishes the total active probe count.
// - Also writes `gi_counters.probe_update_count` as:
//     min(total_active_probes, probes_per_frame)
// =============================================================================

#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

@group(1) @binding(0) var<storage, read> block_sums_in: array<u32>;
@group(1) @binding(1) var<storage, read_write> block_prefixes_out: array<u32>;
@group(1) @binding(2) var<storage, read_write> gi_counters: GICounters;
@group(1) @binding(3) var<uniform> ddgi_params: DDGIParams;

const WORKGROUP_SIZE = 128u;

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
            block_prefixes_out[chunk_idx] = running_offset + chunk_exclusive;
        }
        workgroupBarrier();

        if (local_idx == 0u) {
            shared_running_offset = running_offset + shared_chunk_total;
        }
        workgroupBarrier();
    }

    if (local_idx == 0u) {
        let total_active_probes = shared_running_offset;
        let probes_per_frame = u32(ddgi_params.probe_counts.z);
        let update_count = min(total_active_probes, probes_per_frame);
        atomicStore(&gi_counters.probe_update_count, update_count);
    }
}

