diagnostic(off,subgroup_uniformity);

#include "common.wgsl"
#include "acceleration_common.wgsl"

// Based on "Efficient BVH8 construction for GPU ray tracing" by Vinkler et al.
// Adapted from BVH8 to BVH4 while maintaining the exact algorithm structure

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
const INVALID_ASSIGNMENT: u32 = 0xFu;
const NQ: u32 = 8u; 
const THETA: f32 = 8.0;
const INV_THETA: f32 = 0.125; // 1.0 / 8.0;
const MAX_COST: f32 = 10.0;
const SPIN_THRESHOLD: u32 = 65536u; // 1u << 16u; // tune as needed
const WATCHDOG_ABORT: u32 = 1u; // set to 1u to force-deactivate lanes when tripped
const invalid_bounds: AABB = AABB(
    vec4<f32>(0.0, 0.0, 0.0, -1.0), 
    vec4<f32>(0.0, 0.0, 0.0, -1.0)
);

// -----------------------------------------------------------------------------
// Data Structures
// -----------------------------------------------------------------------------
struct BuildState {
    work_counter: atomic<u32>,
    node_counter: atomic<u32>,
    leaf_counter: atomic<u32>, 
    work_alloc_counter: atomic<u32>,
    prim_count: u32,
};

struct BVHData {
    leaf_count: u32,
    bvh2_count: u32,
    root_index: u32,
    prim_count: u32,
};

struct IndexPair {
    hi: atomic<u32>,
    lo: atomic<u32>,
};

//------------------------------------------------------------------------------
// Bindings & Uniforms
//------------------------------------------------------------------------------
@group(1) @binding(0) var<storage, read_write> bounds: array<AABB>;
@group(1) @binding(1) var<storage, read_write> bvh4_nodes: array<BVH4Node>;
@group(1) @binding(2) var<storage, read_write> build_state: BuildState;
@group(1) @binding(3) var<storage, read_write> index_pairs: array<IndexPair>;
@group(1) @binding(4) var<storage, read_write> prim_indices: array<u32>;
@group(1) @binding(5) var<storage, read_write> bvh_data: BVHData;
// Debug watchdog: [0]=stuck_event_count, [1]=last_stuck_work_id, [2]=max_spin_count
@group(1) @binding(6) var<storage, read_write> debug_watchdog: array<atomic<u32>>;

var<workgroup> wg_sync_count : atomic<u32>;
var<workgroup> wg_break_uniform : u32;
//------------------------------------------------------------------------------
// Utility Functions
//------------------------------------------------------------------------------

// Get nibble (4-bit value) from packed assignments
fn get_nibble(assignments: u32, slot: u32) -> u32 {
    return (assignments >> (slot * 4u)) & 0xFu;
}

// Set nibble (4-bit value) in packed assignments
fn set_nibble(assignments: ptr<function, u32>, slot: u32, value: u32) {
    let shift = slot * 4u;
    let clear_mask = ~(0xFu << shift);
    *assignments = (*assignments & clear_mask) | (value << shift);
}

// Count bits below position in a mask
fn count_bits_below(mask: u32, pos: u32) -> u32 {
    let shifted_mask = mask & ((1u << pos) - 1u);
    return countOneBits(shifted_mask);
}

// Ceiling of log2
fn ceil_log2(x: f32) -> u32 {
    let bits = bitcast<u32>(x);
    let exponent = (bits >> 23u) & 0xFFu;
    let is_pow_2 = (bits & ((1u << 23u) - 1u)) == 0u;
    return exponent + u32(!is_pow_2);
}

// Inverse power of 2
fn inv_pow2(exp: u32) -> f32 {
    return bitcast<f32>((254u - exp) << 23u);
}

// Cost function for placing child at slot s with given offset
fn get_cost(slot: u32, offset: vec3<f32>) -> f32 {
    let sx = select(1.0, -1.0, ((slot >> 2u) & 1u) != 0u);
    let sy = select(1.0, -1.0, ((slot >> 1u) & 1u) != 0u);
    let sz = select(1.0, -1.0, (slot & 1u) != 0u);
    return sx * offset.x + sy * offset.y + sz * offset.z;
}

// Cost of placing child c in slot s
fn get_cost_for_child(child_idx: u32, slot: u32, offsets: array<vec3<f32>, 4>) -> f32 {
    return get_cost(slot, offsets[child_idx]);
}

// Load index pair from array
fn load_index_pair(idx: u32) -> vec2<u32> {
    // Load hi last; writer stores hi last as the "ready" flag.
    let hi = atomicLoad(&index_pairs[idx].hi);
    let lo = atomicLoad(&index_pairs[idx].lo);
    return vec2<u32>(hi, lo);
}
// Store index pair to array
fn store_index_pair(idx: u32, hi: u32, lo: u32) {
    // Publish lo first, then hi as the ready flag.
    atomicStore(&index_pairs[idx].lo, lo);
    atomicStore(&index_pairs[idx].hi, hi);
}

fn syncthreads_count(warp_ctx: WarpCtx, pred: bool) -> u32 {
    let m = warp_ballot_u32(warp_ctx, pred);
    let subgroup_true = mask_popcount(m);

    if (warp_ctx.lane_id == 0u) {
      atomicStore(&wg_sync_count, 0u);
    }
    workgroupBarrier();

    if (warp_ctx.lane_id == 0u) {
      atomicAdd(&wg_sync_count, subgroup_true);
    }
    workgroupBarrier();

    return workgroupUniformLoad(&wg_sync_count);
}

// Greedy assignment. Assigns each child to the
// best available slot independently, using the cost function and skipping
// already-assigned slots.
fn greedy_assignment(offsets: array<vec3<f32>, 4>, n: u32) -> u32 {
    var assignments: u32 = INVALID_IDX;

    for (var c = 0u; c < n; c = c + 1u) {
        var max_cost = -3.402823e+38;
        var best_slot = INVALID_ASSIGNMENT;
        let offset = offsets[c];

        for (var s = 0u; s < 4u; s = s + 1u) {
            // If slot already assigned, skip
            if (get_nibble(assignments, s) != INVALID_ASSIGNMENT) {
                continue;
            }

            let cost = get_cost(s, offset);
            if (cost > max_cost) {
                max_cost = cost;
                best_slot = s;
            }
        }

        set_nibble(&assignments, best_slot, c);
    }

    return assignments;
}

// BVH4 Node Creation
fn create_bvh4_node(
    bounds_arg: AABB,
    child_nodes: array<u32, 4>,
    child_base_idx: u32,
    prim_base_idx: u32,
    assignments: u32,
    inner_mask: u32,
    leaf_mask: u32
) -> BVH4Node {
    var node: BVH4Node;

    // Write world-space bounds directly
    node.min = bounds_arg.min;
    node.max = bounds_arg.max;

    // Encode per-slot children directly into BVH4Node.children
    // Convention:
    // - If slot i is inner: children[i] = f32(child_base_idx + rank among inner slots)
    // - If slot i is leaf:  children[i] = f32(0x80000000 | (prim_base_idx + rank among leaf slots))
    node.children = vec4<f32>(-1.0, -1.0, -1.0, -1.0);

    for (var i = 0u; i < 4u; i = i + 1u) {
        if (get_nibble(assignments, i) == INVALID_ASSIGNMENT) {
            continue;
        }

        let is_inner = (inner_mask & (1u << i)) != 0u;
        let inner_rank = count_bits_below(inner_mask, i);
        let leaf_rank = count_bits_below(leaf_mask, i);

        let encoded = select(
            // leaf
            f32(0x80000000u | (prim_base_idx + leaf_rank)),
            // inner
            f32(child_base_idx + inner_rank),
            is_inner
        );

        node.children[i] = encoded;
    }

    return node;
}

//------------------------------------------------------------------------------
// Single Leaf Handling
//------------------------------------------------------------------------------

fn create_bvh4_single_leaf(work_id: u32) {
    if (work_id == 0u) {
        var child_nodes: array<u32, 4>;
        child_nodes[0] = 0u;
        let assignments = 0xffffff0u;
        let bvh2_node = bounds[0];
        atomicAdd(&build_state.leaf_counter, 1u);
        prim_indices[0] = 0u;

        bvh4_nodes[0] = create_bvh4_node(bvh2_node, child_nodes, 0u, 0u, assignments, 0x0u, 0x1u);
    }
}

//------------------------------------------------------------------------------
// Main Kernel
//------------------------------------------------------------------------------
@compute @workgroup_size(32)
fn convert_bvh2_to_bvh4(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
#if HAS_SUBGROUPS
    @builtin(subgroup_invocation_id)  subgroup_id: u32,
    @builtin(subgroup_size) subgroup_size: u32
#endif
) {
    let leaf_count = bvh_data.leaf_count;
    let bvh2_count = bvh_data.bvh2_count;

    if (group_id.x == 0u && local_id.x == 0u) {
        store_index_pair(0u, bvh2_count - 1u, 0u);
        bvh_data.root_index = bvh2_count - 1u;
    }
    workgroupBarrier();

#if HAS_SUBGROUPS
    let lane = subgroup_id;
    let warp_ctx = make_warp_ctx(local_id.x, lane, subgroup_size);
#else
    let lane = lane_id(local_id.x, LOGICAL_WARP_SIZE);
    let warp_ctx = make_warp_ctx(local_id.x, lane, LOGICAL_WARP_SIZE);
#endif

    // Atomic work allocation
    var work_id = 0u;
    if (is_warp_leader(warp_ctx)) {
        work_id = atomicAdd(&build_state.work_counter, warp_ctx.warp_size);
    }
    work_id = warp_shuffle_u32(warp_ctx, work_id, 0u) + warp_ctx.lane_id;

    var lane_active = work_id < leaf_count;
    var spin_count: u32 = 0u;

    loop {
        let count = syncthreads_count(warp_ctx, lane_active);
        if (local_id.x == 0u) {
            wg_break_uniform = select(0u, 1u, count == 0u);
        }
        let should_break = workgroupUniformLoad(&wg_break_uniform) != 0u;
        if (should_break) {
            break;
        }

        var produced = 0u;
        if (is_warp_leader(warp_ctx)) {
            produced = atomicLoad(&build_state.work_alloc_counter);
        }
        produced = warp_broadcast_u32(warp_ctx, produced, 0u);

        let do_work = lane_active && (work_id < produced);
        if (do_work) {
            // Load index pair
            let index_pair = load_index_pair(work_id);
            let bvh2_node_idx = index_pair.x;
            let bvh4_node_idx = index_pair.y;

            // If no work assigned to this slot yet, skip (keep lane active to poll until assigned)
            var has_work = (bvh2_node_idx != INVALID_IDX);

            var bvh2_node = invalid_bounds;
            if (has_work) {
                bvh2_node = bounds[bvh2_node_idx];
            }

            if (is_leaf(bvh2_node)) {
                prim_indices[bvh4_node_idx] = u32(bvh2_node.min.w);
                lane_active = false;
                has_work = false;
                spin_count = 0u;
            }

            if (has_work) {
                // Gather children using top-down traversal
                var inner_mask = 0u;
                var child_count = 0u;
                var child_nodes: array<u32, 4>;

                var child_bounds_local: array<AABB, 2>;
                var child_bounds_cached: array<AABB, 4>;
                var left_child = u32(bvh2_node.min.w);
                var right_child = u32(bvh2_node.max.w);
                var msb = 0;

                // Top-down traversal to collect up to 4 nodes
                loop {
                    child_bounds_local[0] = invalid_bounds;
                    child_bounds_local[1] = invalid_bounds;

                    child_bounds_local[0].min = select(invalid_bounds.min, bounds[left_child].min, left_child != INVALID_IDX);
                    child_bounds_local[1].min = select(invalid_bounds.min, bounds[right_child].min, right_child != INVALID_IDX);
                    child_bounds_local[0].max = select(invalid_bounds.max, bounds[left_child].max, left_child != INVALID_IDX);
                    child_bounds_local[1].max = select(invalid_bounds.max, bounds[right_child].max, right_child != INVALID_IDX);

                    // Choose the child with smaller area first
                    let smaller = calculate_aabb_surface_area(child_bounds_local[0].min.xyz, child_bounds_local[0].max.xyz) <
                                  calculate_aabb_surface_area(child_bounds_local[1].min.xyz, child_bounds_local[1].max.xyz);
                    var first = select(1u, 0u, smaller);

                    // Push both children; the first goes at index msb (or 0 initially), second at child_count
                    for (var i = 0u; i < 2u; i = i + 1u) {
                        let idx = select(child_count, u32(msb), i == 0u);
                        let chosen = select(right_child, left_child, first == 0u);
                        let chosen_bounds = child_bounds_local[first];

                        if (chosen_bounds.max.w != -1.0) {
                            inner_mask = inner_mask | (1u << idx);
                        }

                        child_nodes[idx] = chosen;
                        child_bounds_cached[idx] = chosen_bounds;

                        child_count = child_count + 1u;

                        // Toggle to the other child for the second push
                        first = 1u - first;
                    }

                    // Pop the last inner node from the stack
                    msb = 31 - i32(countLeadingZeros(inner_mask));

                    if (msb < 0 || child_count == 4u) {
                       break;
                    }

                    // Clear msb and overwrite that slot next iteration
                    inner_mask = inner_mask & ~(1u << u32(msb));
                    child_count = child_count - 1u;
 
                    let expanded_node = child_bounds_cached[u32(msb)];
                    left_child = u32(expanded_node.min.w);
                    right_child = u32(expanded_node.max.w);
                }

                let parent_centroid = bvh2_node.min.xyz + bvh2_node.max.xyz;

                // Reorder the child nodes (greedy by default, optional auction)
                var offsets: array<vec3<f32>, 4>;
                for (var c = 0u; c < child_count; c = c + 1u) {
                    let centroid = child_bounds_cached[c].min.xyz + child_bounds_cached[c].max.xyz;
                    offsets[c] = parent_centroid - centroid;
                }
                var assignments = greedy_assignment(offsets, child_count);

                // Compute new masks after reordering
                var new_inner_mask: u32 = 0u;
                var leaf_mask: u32 = 0u;
                for (var i = 0u; i < 4u; i = i + 1u) {
                    if (get_nibble(assignments, i) == INVALID_ASSIGNMENT) {
                        continue;
                    }

                    let bit = (inner_mask >> get_nibble(assignments, i)) & 1u;
                    new_inner_mask = new_inner_mask | (bit << i);
                    leaf_mask = leaf_mask | ((1u - bit) << i);
                }
                inner_mask = new_inner_mask;

                let inner_count = countOneBits(inner_mask);
                let leaf_children = child_count - inner_count;

                // Allocate new inner nodes, leaf nodes and work items
                let child_base_idx = atomicAdd(&build_state.node_counter, inner_count);
                let work_base_idx = atomicAdd(&build_state.work_alloc_counter, child_count - 1u);

                var prim_base_idx: u32 = 0u;
                if (leaf_children > 0u) {
                    prim_base_idx = atomicAdd(&build_state.leaf_counter, leaf_children);
                }

                // Add new work in the index pair list
                for (var i = 0u; i < 4u; i = i + 1u) {
                    if (get_nibble(assignments, i) == INVALID_ASSIGNMENT) {
                        continue;
                    }

                    var pair_hi = child_nodes[get_nibble(assignments, i)];
                    var pair_lo = select(
                        prim_base_idx + count_bits_below(leaf_mask, i),
                        child_base_idx + count_bits_below(inner_mask, i),
                        (inner_mask & (1u << i)) != 0u
                    );

                    let c = count_bits_below(inner_mask | leaf_mask, i);
                    let idx = select(work_base_idx + c - 1u, work_id, c == 0u);
                    store_index_pair(idx, pair_hi, pair_lo);
                }

                // Create and store the new BVH4 node
                bvh4_nodes[bvh4_node_idx] = create_bvh4_node(
                    bvh2_node, child_nodes, child_base_idx, prim_base_idx, 
                    assignments, inner_mask, leaf_mask
                );
            }
        }

        // Watchdog: if we are continually selected for work but never complete a leaf,
        // and also did not have valid work this iteration, count it toward spin.
        if (lane_active) {
            spin_count += 1u;
            if (spin_count > SPIN_THRESHOLD) {
                atomicAdd(&debug_watchdog[0], 1u);
                if (WATCHDOG_ABORT == 1u) {
                    lane_active = false;
                } else {
                    // Reset counter so we do not overflow and can count multiple events
                    spin_count = 0u;
                }
            }
        }
    }
}

