// =============================================================================
// GI-1.0 World Cache Compact
// - Compacts active entries in the world cache using parallel prefix-sum
// - Produces dense array of indices of active entries
// - Generates indirect dispatch parameters for subsequent passes
// =============================================================================
#include "common.wgsl"

// =============================================================================
// Pass 2: Prefix Sum (Optimized with Native Subgroups)
// =============================================================================
@group(1) @binding(0) var<storage, read> active_flags_in: array<u32>;
@group(1) @binding(1) var<storage, read_write> prefix_sum: array<u32>;
@group(1) @binding(2) var<storage, read_write> block_sums: array<u32>;

const WORKGROUP_SIZE = 128u;

var<workgroup> subgroup_sums: array<u32, 4u>;

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
    
    // Load input value
    let value = select(0u, active_flags_in[global_idx], global_idx < arrayLength(&active_flags_in));
    
#if HAS_SUBGROUPS
    // Native subgroup-level exclusive scan
    let warp_ctx = make_warp_ctx(local_idx, sid, ss);
    let subgroup_exclusive = warp_scan_exclusive_add_u32(warp_ctx, value);
    let subgroup_total = warp_reduce_add_u32(warp_ctx, value);
    
    // Store subgroup totals for cross-subgroup accumulation
    //let subgroup_id = local_idx / ss;
    if (is_warp_leader(warp_ctx)) {
        subgroup_sums[sid] = subgroup_total;
    }
    workgroupBarrier();
    
    // Accumulate prefix from previous subgroups
    var prefix_from_previous_subgroups = 0u;
    for (var i = 0u; i < sid; i = i + 1u) {
        prefix_from_previous_subgroups = prefix_from_previous_subgroups + subgroup_sums[i];
    }
    
    let final_exclusive = subgroup_exclusive + prefix_from_previous_subgroups;
#else
    // Fallback to logical warp implementation
    let lane = lane_id(local_idx, LOGICAL_WARP_SIZE);
    let warp_id_local = warp_id(local_idx, LOGICAL_WARP_SIZE);
    
    var warp_ctx = make_warp_ctx(local_idx, lane, LOGICAL_WARP_SIZE);
    let warp_exclusive = warp_scan_exclusive_add_u32(warp_ctx, value);
    let warp_total = warp_reduce_add_u32(warp_ctx, value);
    
    // Store warp totals
    if (is_warp_leader(warp_ctx)) {
        subgroup_sums[warp_id_local] = warp_total;
    }
    workgroupBarrier();
    
    // Accumulate prefix from previous warps
    var prefix_from_previous_warps = 0u;
    for (var i = 0u; i < warp_id_local; i = i + 1u) {
        prefix_from_previous_warps = prefix_from_previous_warps + subgroup_sums[i];
    }
    
    let final_exclusive = warp_exclusive + prefix_from_previous_warps;
#endif
    
    // Write workgroup-level exclusive prefix sum
    if (global_idx < arrayLength(&prefix_sum)) {
        prefix_sum[global_idx] = final_exclusive;
    }
    
    // Last thread stores the total sum for this workgroup
    if (local_idx == WORKGROUP_SIZE - 1u) {
        block_sums[workgroup_idx] = final_exclusive + value;
    }
}
