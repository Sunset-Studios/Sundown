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
const INV_THETA: f32 = 1.0 / 8.0;
const MAX_COST: f32 = 10.0;
const QUANT_STEP: f32 = 1.0 / 255.0; // 1.0 / ((1 << NQ) - 1)
const WARP_SIZE: u32 = 32u;

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
    *assignments = (*assignments & clear_mask) | ((value & 0xFu) << shift);
}

// Count bits below position in a mask
fn count_bits_below(mask: u32, pos: u32) -> u32 {
    let shifted_mask = mask & ((1u << pos) - 1u);
    return countOneBits(shifted_mask);
}

// Ceiling of log2
fn ceil_log2(x: f32) -> u32 {
    if (x <= 1.0) { return 0u; }
    let bits = bitcast<u32>(x);
    let exponent = (bits >> 23u) & 0xFFu;
    let mantissa = bits & 0x7FFFFFu;
    let result = exponent - 127u;
    return select(result, result + 1u, mantissa != 0u);
}

// Inverse power of 2
fn inv_pow2(exp: u32) -> f32 {
    if (exp == 0u) { return 1.0; }
    return 1.0 / pow(2.0, f32(exp));
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
    let hi = atomicLoad(&index_pairs[idx].hi);
    let lo = atomicLoad(&index_pairs[idx].lo);
    return vec2<u32>(hi, lo);
}

// Store index pair to array
fn store_index_pair(idx: u32, hi: u32, lo: u32) {
    atomicStore(&index_pairs[idx].hi, hi);
    atomicStore(&index_pairs[idx].lo, lo);
}

// Auction algorithm for optimal assignment
fn auction_assignment(offsets: array<vec3<f32>, 4>, max_cost: f32, n: u32) -> u32 {
    var prices: array<f32, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) { 
        prices[i] = 0.0; 
    }

    // Initialize epsilon (perfect port of CUDA logic)
    let threshold = 1.0 / f32(n);
    var epsilon = max(max_cost, threshold);

    var assignments: u32 = INVALID_IDX;
    while (epsilon >= threshold) {
        // Reset assignments
        assignments = INVALID_IDX;

        // Reset bidders with slots ranging from 0 to 3 (0x3210 = packed nibbles)
        var bidders: u32 = 0x3210u;
        var bidder_count = n;

        while (bidder_count > 0u) {
            bidder_count = bidder_count - 1u;
            let c = get_nibble(bidders, bidder_count);

            var winning_reward = -3.402823e+38; // -FLT_MAX
            var second_winning_reward = -3.402823e+38;
            var winning_slot: u32 = INVALID_ASSIGNMENT;

            for (var s = 0u; s < 4u; s = s + 1u) {
                let reward = get_cost(s, offsets[c]) - prices[s];
                if (reward > winning_reward) {
                    second_winning_reward = winning_reward;
                    winning_reward = reward;
                    winning_slot = s;
                } else if (reward > second_winning_reward) {
                    second_winning_reward = reward;
                }
            }

            prices[winning_slot] = prices[winning_slot] + (winning_reward - second_winning_reward) + epsilon;

            let previous_assignment = get_nibble(assignments, winning_slot);
            set_nibble(&assignments, winning_slot, c);

            if (previous_assignment != INVALID_ASSIGNMENT) {
                set_nibble(&bidders, bidder_count, previous_assignment);
                bidder_count = bidder_count + 1u;
            }
        }

        // Epsilon scaling
        epsilon *= INV_THETA;
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
// Single Leaf Handling (perfect CUDA port)
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
// Main Kernel (perfect CUDA port with BVH4 adaptations)
//------------------------------------------------------------------------------

@compute @workgroup_size(WARP_SIZE)
fn convert_bvh2_to_bvh4(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
#if HAS_SUBGROUPS
    @builtin(subgroup_invocation_id)  subgroup_id: u32,
    @builtin(subgroup_size) subgroup_size: u32
#endif
) {
    let size = bvh_data.bvh2_count;

    // Limit to first warp per workgroup for deterministic behavior
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

    var lane_active = work_id < build_state.prim_count;

    // Handle single leaf case
    if (build_state.prim_count == 1u) {
        create_bvh4_single_leaf(work_id);
        return;
    }

    // Main processing loop
    while (warp_any(warp_ctx, lane_active)) {
        if (!lane_active) {
            continue;
        }

        // Load index pair
        let index_pair = load_index_pair(work_id);
        let bvh2_node_idx = index_pair.x;
        let bvh4_node_idx = index_pair.y;

        // If no work assigned, skip
        if (bvh2_node_idx == INVALID_IDX || bvh2_node_idx >= size) {
            lane_active = false;
            continue;
        }

        let bvh2_node = bounds[bvh2_node_idx];
        // If leaf node, create BVH4 leaf
        if (is_leaf(bvh2_node)) {
            prim_indices[bvh4_node_idx] = u32(bvh2_node.min.w);
            lane_active = false;
            continue;
        }

        // Gather children using top-down traversal
        var inner_mask: u32 = 0u;
        var child_count: u32 = 0u;
        var child_nodes: array<u32, 4>;

        var child_bounds: array<AABB, 2>;
        var left_child = u32(bvh2_node.min.w);
        var right_child = u32(bvh2_node.max.w);
        var msb: i32 = 0;

        // Top-down traversal to collect up to 4 nodes
        loop {
            child_bounds[0] = bounds[left_child];
            child_bounds[1] = bounds[right_child];

            // Push both children
            let area0 = calculate_aabb_surface_area(child_bounds[0].min.xyz, child_bounds[0].max.xyz);
            let area1 = calculate_aabb_surface_area(child_bounds[1].min.xyz, child_bounds[1].max.xyz);
            let first = !(area0 < area1);

            // Order children by 'first'
            let ordered_0 = select(right_child, left_child, first);
            let ordered_1 = select(left_child, right_child, first);

            // Push ordered_0 at next slot
            let idx0 = child_count;
            if (!is_leaf(bounds[ordered_0])) {
                inner_mask = inner_mask | (1u << idx0);
            }
            child_nodes[idx0] = ordered_0;
            child_count = child_count + 1u;

            // Push ordered_1 at next slot
            let idx1 = child_count;
            if (!is_leaf(bounds[ordered_1])) {
                inner_mask = inner_mask | (1u << idx1);
            }
            child_nodes[idx1] = ordered_1;
            child_count = child_count + 1u;

            // Pop the last inner node from the stack
            msb = 31 - i32(countLeadingZeros(inner_mask));

            if (msb < 0 || child_count == 4u) {
                break;
            }

            // Set the msb to 0
            inner_mask = inner_mask & ~(1u << u32(msb));
            child_count -= 1u;

            let new_idx = child_nodes[u32(msb)];
            let expanded_node = bounds[new_idx];
            left_child = u32(expanded_node.min.w);
            right_child = u32(expanded_node.max.w);
        }

        let parent_centroid = bvh2_node.min.xyz + bvh2_node.max.xyz;

        // Reorder children using Auction assignment
        let diagonal = bvh2_node.max.xyz - bvh2_node.min.xyz;
        let max_dim = max(max(diagonal.x, diagonal.y), diagonal.z);
        var cost_scale = select(MAX_COST / max_dim, 0.0, max_dim == 0.0);
        cost_scale *= 0.5;

        var offsets: array<vec3<f32>, 4>;
        var max_cost = -3.402823e+38;

        for (var c = 0u; c < child_count; c = c + 1u) {
            let child_bounds_local = bounds[child_nodes[c]];
            let centroid = child_bounds_local.min.xyz + child_bounds_local.max.xyz;
            offsets[c] = (parent_centroid - centroid) * cost_scale;

            let cost_mag = abs(offsets[c].x) + abs(offsets[c].y) + abs(offsets[c].z);
            if (cost_mag > max_cost) {
                max_cost = cost_mag;
            }
        }

        let assignments = auction_assignment(offsets, max_cost, child_count);

        // Compute new masks after reordering (perfect CUDA port)
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
        let leaf_count = child_count - inner_count;

        // Allocate new inner nodes, leaf nodes and work items
        let child_base_idx = atomicAdd(&build_state.node_counter, inner_count);
        let work_base_idx = atomicAdd(&build_state.work_alloc_counter, child_count - 1u);

        var prim_base_idx: u32 = 0u;
        if (leaf_count > 0u) {
            prim_base_idx = atomicAdd(&build_state.leaf_counter, leaf_count);
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