// =============================================================================
// DDGI Active Probe Prefix Sum
// Computes exclusive prefix sums over depth-visible scheduler candidates.
// =============================================================================

#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

@group(1) @binding(0) var<storage, read> candidate_priorities_in: array<u32>;
@group(1) @binding(1) var<storage, read_write> prefix_sum: array<u32>;
@group(1) @binding(2) var<storage, read_write> block_sums: array<u32>;

const WORKGROUP_SIZE = 128u;

var<workgroup> fresh_subgroup_sums: array<u32, 4u>;
var<workgroup> normal_subgroup_sums: array<u32, 4u>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn cs(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
#if HAS_SUBGROUPS
    @builtin(subgroup_invocation_id) sid: u32,
    @builtin(subgroup_size) ss: u32
#endif
) {
    let global_idx = gid.x;
    let local_idx = lid.x;
    let probe_count = arrayLength(&candidate_priorities_in);
    let block_count = arrayLength(&block_sums) / DDGI_PROBE_SCHEDULE_PRIORITY_COUNT;
    var active_priority = DDGI_PROBE_SCHEDULE_PRIORITY_NONE;
    if (global_idx < probe_count) {
        active_priority = candidate_priorities_in[global_idx];
    }
    let fresh_value = select(0u, 1u, active_priority == DDGI_PROBE_SCHEDULE_PRIORITY_FRESH);
    let normal_value = select(0u, 1u, active_priority == DDGI_PROBE_SCHEDULE_PRIORITY_NORMAL);

#if HAS_SUBGROUPS
    let warp_ctx = make_warp_ctx(local_idx, sid, ss);
    let fresh_subgroup_exclusive = warp_scan_exclusive_add_u32(warp_ctx, fresh_value);
    let fresh_subgroup_total = warp_reduce_add_u32(warp_ctx, fresh_value);
    let normal_subgroup_exclusive = warp_scan_exclusive_add_u32(warp_ctx, normal_value);
    let normal_subgroup_total = warp_reduce_add_u32(warp_ctx, normal_value);

    if (is_warp_leader(warp_ctx)) {
        fresh_subgroup_sums[sid] = fresh_subgroup_total;
        normal_subgroup_sums[sid] = normal_subgroup_total;
    }
    workgroupBarrier();

    var fresh_prefix_from_prev_sg = 0u;
    var normal_prefix_from_prev_sg = 0u;
    for (var i = 0u; i < sid; i = i + 1u) {
        fresh_prefix_from_prev_sg = fresh_prefix_from_prev_sg + fresh_subgroup_sums[i];
        normal_prefix_from_prev_sg = normal_prefix_from_prev_sg + normal_subgroup_sums[i];
    }

    let fresh_final_exclusive = fresh_subgroup_exclusive + fresh_prefix_from_prev_sg;
    let normal_final_exclusive = normal_subgroup_exclusive + normal_prefix_from_prev_sg;
#else
    let lane = lane_id(local_idx, LOGICAL_WARP_SIZE);
    let warp_id_local = warp_id(local_idx, LOGICAL_WARP_SIZE);
    let warp_ctx = make_warp_ctx(local_idx, lane, LOGICAL_WARP_SIZE);
    let fresh_warp_exclusive = warp_scan_exclusive_add_u32(warp_ctx, fresh_value);
    let fresh_warp_total = warp_reduce_add_u32(warp_ctx, fresh_value);
    let normal_warp_exclusive = warp_scan_exclusive_add_u32(warp_ctx, normal_value);
    let normal_warp_total = warp_reduce_add_u32(warp_ctx, normal_value);

    if (is_warp_leader(warp_ctx)) {
        fresh_subgroup_sums[warp_id_local] = fresh_warp_total;
        normal_subgroup_sums[warp_id_local] = normal_warp_total;
    }
    workgroupBarrier();

    var fresh_prefix_from_prev_warps = 0u;
    var normal_prefix_from_prev_warps = 0u;
    for (var i = 0u; i < warp_id_local; i = i + 1u) {
        fresh_prefix_from_prev_warps = fresh_prefix_from_prev_warps + fresh_subgroup_sums[i];
        normal_prefix_from_prev_warps = normal_prefix_from_prev_warps + normal_subgroup_sums[i];
    }

    let fresh_final_exclusive = fresh_warp_exclusive + fresh_prefix_from_prev_warps;
    let normal_final_exclusive = normal_warp_exclusive + normal_prefix_from_prev_warps;
#endif

    if (global_idx < probe_count) {
        prefix_sum[DDGI_PROBE_SCHEDULE_PRIORITY_BUCKET_FRESH * probe_count + global_idx] = fresh_final_exclusive;
        prefix_sum[DDGI_PROBE_SCHEDULE_PRIORITY_BUCKET_NORMAL * probe_count + global_idx] = normal_final_exclusive;
    }

    if (local_idx == WORKGROUP_SIZE - 1u) {
        block_sums[DDGI_PROBE_SCHEDULE_PRIORITY_BUCKET_FRESH * block_count + wid.x] = fresh_final_exclusive + fresh_value;
        block_sums[DDGI_PROBE_SCHEDULE_PRIORITY_BUCKET_NORMAL * block_count + wid.x] = normal_final_exclusive + normal_value;
    }
}
