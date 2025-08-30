#include "common.wgsl"
#include "acceleration_common.wgsl"

// H-PLOC Step 2.5: Initialize leaf clusters.

// -----------------------------------------------------------------------------
// Data Structures
// -----------------------------------------------------------------------------
struct BVHData {
    leaf_count: atomic<u32>,
    bvh2_count: atomic<u32>,
    root_index: u32,
    prim_count: u32,
};

struct IndexPair {
    hi: u32,
    lo: u32,
};

//------------------------------------------------------------------------------
// Bindings & Uniforms
//------------------------------------------------------------------------------
@group(1) @binding(0) var<storage, read> bounds: array<AABB>;
@group(1) @binding(1) var<storage, read_write> cluster_idx: array<u32>;
@group(1) @binding(2) var<storage, read_write> counters: BVHData;
@group(1) @binding(3) var<storage, read> morton_codes: array<u32>;
@group(1) @binding(4) var<storage, read_write> parent_idx: array<u32>;
@group(1) @binding(5) var<storage, read_write> index_pairs: array<IndexPair>;

//------------------------------------------------------------------------------
// HPLOC Kernels 
//------------------------------------------------------------------------------

@compute @workgroup_size(64)
fn initialize_leaf_clusters(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
#if HAS_SUBGROUPS
    @builtin(subgroup_invocation_id)  subgroup_id: u32,
    @builtin(subgroup_size) subgroup_size: u32
#endif
) {
    let prim_idx = gid.x;

#if HAS_SUBGROUPS
    let lane = subgroup_id;
    let warp_ctx = make_warp_ctx(local_id.x, lane, subgroup_size);
#else
    let lane = lane_id(local_id.x, LOGICAL_WARP_SIZE);
    let warp_ctx = make_warp_ctx(local_id.x, lane, LOGICAL_WARP_SIZE);
#endif

    // 1) Warp-aggregate the increment amount
    let is_valid_leaf = select(0u, 1u, is_leaf(bounds[prim_idx]));
    let warp_sum = warp_reduce_add_u32(warp_ctx, is_valid_leaf);
    // 2) One atomicAdd per warp
    if (is_warp_leader(warp_ctx)) {
      atomicAdd(&counters.leaf_count, warp_sum);
    }

    // parent_idx is set to INVALID_IDX because this leaf is not attached to a parent yet
    parent_idx[prim_idx] = INVALID_IDX;
    index_pairs[prim_idx].hi = INVALID_IDX;
    index_pairs[prim_idx].lo = INVALID_IDX;

    workgroupBarrier();

    if (local_id.x == 0u) {
        let leaf_count = atomicLoad(&counters.leaf_count);
        atomicStore(&counters.bvh2_count, leaf_count);
    }
}