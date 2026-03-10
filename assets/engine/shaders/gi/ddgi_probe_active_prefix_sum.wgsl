// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                    DDGI Active Probe Prefix Sum                           ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Pass 2 of active-only probe cycling with frustum culling priority:       ║
// ║  - Reads packed active flags (bit 0 = nonculled, bit 1 = culled) and      ║
// ║    computes exclusive prefix sums for both in a single pass.               ║
// ║                                                                           ║
// ║  Output:                                                                  ║
// ║  - prefix_sum_nonculled[i] = number of nonculled active flags in [0..i)   ║
// ║  - prefix_sum_culled[i] = number of culled active flags in [0..i)         ║
// ║  - block_sums_nonculled[b] = total nonculled active flags in workgroup b  ║
// ║  - block_sums_culled[b] = total culled active flags in workgroup b        ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

#include "common.wgsl"

@group(1) @binding(0) var<storage, read> active_flags_in: array<u32>;
@group(1) @binding(1) var<storage, read_write> prefix_sum_nonculled: array<u32>;
@group(1) @binding(2) var<storage, read_write> prefix_sum_culled: array<u32>;
@group(1) @binding(3) var<storage, read_write> block_sums_nonculled: array<u32>;
@group(1) @binding(4) var<storage, read_write> block_sums_culled: array<u32>;

const WORKGROUP_SIZE = 256u;

// Workgroup shared memory for both categories
var<workgroup> subgroup_sums_nonculled: array<u32, 4u>;
var<workgroup> subgroup_sums_culled: array<u32, 4u>;

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
    let workgroup_idx = wid.x;
    let array_len = arrayLength(&active_flags_in);

    // ─────────────────────────────────────────────────────────────────────────
    // Unpack flags: bit 0 = nonculled, bit 1 = culled
    // ─────────────────────────────────────────────────────────────────────────
    let packed = select(0u, active_flags_in[global_idx], global_idx < array_len);
    let value_nonculled = packed & 1u;
    let value_culled = (packed >> 1u) & 1u;

#if HAS_SUBGROUPS
    // ─────────────────────────────────────────────────────────────────────────
    // Subgroup path: process nonculled
    // ─────────────────────────────────────────────────────────────────────────
    let warp_ctx = make_warp_ctx(local_idx, sid, ss);
    
    let subgroup_exclusive_nonculled = warp_scan_exclusive_add_u32(warp_ctx, value_nonculled);
    let subgroup_total_nonculled = warp_reduce_add_u32(warp_ctx, value_nonculled);

    if (is_warp_leader(warp_ctx)) {
        subgroup_sums_nonculled[sid] = subgroup_total_nonculled;
    }
    workgroupBarrier();

    var prefix_from_prev_sg_nonculled = 0u;
    for (var i = 0u; i < sid; i = i + 1u) {
        prefix_from_prev_sg_nonculled = prefix_from_prev_sg_nonculled + subgroup_sums_nonculled[i];
    }

    let final_exclusive_nonculled = subgroup_exclusive_nonculled + prefix_from_prev_sg_nonculled;

    // ─────────────────────────────────────────────────────────────────────────
    // Subgroup path: process culled
    // ─────────────────────────────────────────────────────────────────────────
    let subgroup_exclusive_culled = warp_scan_exclusive_add_u32(warp_ctx, value_culled);
    let subgroup_total_culled = warp_reduce_add_u32(warp_ctx, value_culled);

    if (is_warp_leader(warp_ctx)) {
        subgroup_sums_culled[sid] = subgroup_total_culled;
    }
    workgroupBarrier();

    var prefix_from_prev_sg_culled = 0u;
    for (var i = 0u; i < sid; i = i + 1u) {
        prefix_from_prev_sg_culled = prefix_from_prev_sg_culled + subgroup_sums_culled[i];
    }

    let final_exclusive_culled = subgroup_exclusive_culled + prefix_from_prev_sg_culled;

#else
    // ─────────────────────────────────────────────────────────────────────────
    // Logical warp path: process nonculled
    // ─────────────────────────────────────────────────────────────────────────
    let lane = lane_id(local_idx, LOGICAL_WARP_SIZE);
    let warp_id_local = warp_id(local_idx, LOGICAL_WARP_SIZE);
    let warp_ctx = make_warp_ctx(local_idx, lane, LOGICAL_WARP_SIZE);

    let warp_exclusive_nonculled = warp_scan_exclusive_add_u32(warp_ctx, value_nonculled);
    let warp_total_nonculled = warp_reduce_add_u32(warp_ctx, value_nonculled);

    if (is_warp_leader(warp_ctx)) {
        subgroup_sums_nonculled[warp_id_local] = warp_total_nonculled;
    }
    workgroupBarrier();

    var prefix_from_prev_warps_nonculled = 0u;
    for (var i = 0u; i < warp_id_local; i = i + 1u) {
        prefix_from_prev_warps_nonculled = prefix_from_prev_warps_nonculled + subgroup_sums_nonculled[i];
    }

    let final_exclusive_nonculled = warp_exclusive_nonculled + prefix_from_prev_warps_nonculled;

    // ─────────────────────────────────────────────────────────────────────────
    // Logical warp path: process culled
    // ─────────────────────────────────────────────────────────────────────────
    let warp_exclusive_culled = warp_scan_exclusive_add_u32(warp_ctx, value_culled);
    let warp_total_culled = warp_reduce_add_u32(warp_ctx, value_culled);

    if (is_warp_leader(warp_ctx)) {
        subgroup_sums_culled[warp_id_local] = warp_total_culled;
    }
    workgroupBarrier();

    var prefix_from_prev_warps_culled = 0u;
    for (var i = 0u; i < warp_id_local; i = i + 1u) {
        prefix_from_prev_warps_culled = prefix_from_prev_warps_culled + subgroup_sums_culled[i];
    }

    let final_exclusive_culled = warp_exclusive_culled + prefix_from_prev_warps_culled;
#endif

    // ─────────────────────────────────────────────────────────────────────────
    // Write results for both categories
    // ─────────────────────────────────────────────────────────────────────────
    if (global_idx < arrayLength(&prefix_sum_nonculled)) {
        prefix_sum_nonculled[global_idx] = final_exclusive_nonculled;
        prefix_sum_culled[global_idx] = final_exclusive_culled;
    }

    // Write block sums (last thread in workgroup)
    if (local_idx == WORKGROUP_SIZE - 1u) {
        block_sums_nonculled[workgroup_idx] = final_exclusive_nonculled + value_nonculled;
        block_sums_culled[workgroup_idx] = final_exclusive_culled + value_culled;
    }
}
