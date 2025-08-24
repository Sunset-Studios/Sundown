#include "common.wgsl"
#include "acceleration_common.wgsl"

// H-PLOC Step 2.5: Initialize leaf clusters.

// -----------------------------------------------------------------------------
// Data Structures
// -----------------------------------------------------------------------------
struct Cluster {
    aabb_min_and_node_idx: vec4<f32>,
    aabb_max_and_is_active: vec4<f32>,
};

struct Counters {
    bvh2_count: atomic<u32>,
    bvh4_count: atomic<u32>,
};

//------------------------------------------------------------------------------
// Bindings & Uniforms
//------------------------------------------------------------------------------
@group(1) @binding(0) var<storage, read> bounds: array<AABB>;
@group(1) @binding(1) var<storage, read_write> cluster_idx: array<u32>;
@group(1) @binding(2) var<storage, read_write> bvh2_nodes: array<BVH2Node>;
@group(1) @binding(3) var<storage, read_write> counters: Counters;
@group(1) @binding(4) var<storage, read_write> clusters: array<Cluster>;
@group(1) @binding(5) var<storage, read> morton_codes: array<u32>;
@group(1) @binding(6) var<storage, read_write> parent_idx: array<u32>;

//------------------------------------------------------------------------------
// HPLOC Kernels 
//------------------------------------------------------------------------------

@compute @workgroup_size(256)
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
    let warp_sum = warp_reduce_add_u32(warp_ctx, 1u);

    // 2) One atomicAdd per warp
    var base = 0u;
    if (is_warp_leader(warp_ctx)) {
      base = atomicAdd(&counters.bvh2_count, warp_sum);
    }
    base = warp_broadcast_u32(warp_ctx, base, 0u);

    // 3) Per-lane exclusive prefix to get unique index
    let offset = warp_scan_exclusive_add_u32(warp_ctx, 1u);
    let leaf_node_idx = base + offset;

    let sorted_prim_idx = cluster_idx[prim_idx];
    let bound = bounds[sorted_prim_idx];

    // min and max will be filled later when this leaf is attached to a parent
    bvh2_nodes[leaf_node_idx].min_and_left_child.w = f32(0x80000000u | sorted_prim_idx);
    bvh2_nodes[leaf_node_idx].max_and_right_child.w = f32(0xffffffffu);

    clusters[leaf_node_idx].aabb_min_and_node_idx = bound.min;
    clusters[leaf_node_idx].aabb_max_and_is_active = bound.max;

    // Seed cluster_idx with the leaf mapping in sorted order
    parent_idx[prim_idx] = INVALID_IDX;
    cluster_idx[prim_idx] = leaf_node_idx;
}
