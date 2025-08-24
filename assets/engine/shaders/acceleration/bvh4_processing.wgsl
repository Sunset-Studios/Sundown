#include "common.wgsl"
#include "acceleration_common.wgsl"

// H-PLOC Step 2: GPU-based acceleration structure construction.
// This kernel converts a binary BVH (BVH2) into a wide-branching BVH (BVH4).
// High-level:
// - Traverse the BVH2 top-down using a small local stack.
// - Build each BVH4 node by gathering up to four BVH2 children via a tiny frontier that expands
//   inner BVH2 nodes while capacity remains.
// - Order gathered children into four slots with an auction-based assignment heuristic
//   to improve spatial coherence for traversal.
// - Emit leaves as encoded handles; allocate BVH4 inner nodes for internal children and push them.

// -----------------------------------------------------------------------------
// Data Structures
// -----------------------------------------------------------------------------
// Node allocation counters used during conversion. The BVH4 index space grows monotonically via
// atomic increments to avoid write hazards while maintaining deterministic indexing.
struct Counters {
    bvh2_count: atomic<u32>,
    bvh4_count: atomic<u32>,
};

//------------------------------------------------------------------------------
// Bindings & Uniforms
//------------------------------------------------------------------------------
// bvh2_nodes: input linear array of BVH2 nodes (built by previous construction stage).
@group(1) @binding(0) var<storage, read_write> bvh2_nodes: array<BVH2Node>;
// bvh4_nodes: output linear array of BVH4 nodes produced by this conversion.
@group(1) @binding(1) var<storage, read_write> bvh4_nodes: array<BVH4Node>;
// counters: atomic counters used to allocate BVH4 node indices deterministically.
@group(1) @binding(2) var<storage, read_write> counters: Counters;
// scene_aabb: world-space bounds for initializing the root stack state.
@group(1) @binding(3) var<uniform> scene_aabb: AABB;

//------------------------------------------------------------------------------
// Kernel 1: BVH2 -> BVH4 Conversion
//------------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Helpers adapted from CUDA BVH8 converter (ported for BVH4)
// -----------------------------------------------------------------------------
// Assignment packing and cost/epsilon schedule parameters for child-slot ordering.

const INVALID_ASSIGNMENT: u32 = 0xFu;
const THETA: f32 = 8.0;
const INV_THETA: f32 = 1.0 / THETA;
const MAX_COST: f32 = 10.0;

// Packed 4-bit-per-slot utilities to encode child index per slot [0..3].
fn get_nibble(assignments: u32, slot: u32) -> u32 {
    return (assignments >> (slot * 4u)) & 0xFu;
}

fn set_nibble(assignments: ptr<function, u32>, slot: u32, value: u32) {
    let shift = slot * 4u;
    let clear_mask = ~(0xFu << shift);
    *assignments = (*assignments & clear_mask) | ((value & 0xFu) << shift);
}

// Cost of placing a child at slot s (s in [0,3]) using 3-bit sign encoding
// The fixed sign pattern approximates consistent octants, encouraging coherent child placement.
fn get_cost(slot: u32, offset: vec3<f32>) -> f32 {
    // Map 4 slots to 4 distinct sign combinations across xyz:
    // s=0: (+,+,+), s=1: (-,+,-), s=2: (+,-,-), s=3: (-,-,+)
    let sx = select(1.0, -1.0, (slot & 2u) != 0u);
    let sy = select(1.0, -1.0, (slot & 1u) != 0u);
    let parity = ((slot & 1u) ^ ((slot >> 1u) & 1u)) != 0u;
    let sz = select(1.0, -1.0, parity);
    return sx * offset.x + sy * offset.y + sz * offset.z;
}

// AABB centroid helpers used for computing offsets relative to the parent.
fn aabb_centroid2x(aabb: AABB) -> vec3<f32> {
    return aabb.min.xyz + aabb.max.xyz;
}

fn aabb_centroid(aabb: AABB) -> vec3<f32> {
    return 0.5 * (aabb.min.xyz + aabb.max.xyz);
}

// Auction assignment (maximization) for up to 4 slots/children
// Maximizes sum over children of (cost(slot, child) - price[slot]) by iteratively raising prices
// to resolve conflicts. Returns a nibble-packed map: slot -> child_index.
fn auction_assignment(offsets: array<vec3<f32>, 4>, max_cost: f32, n: u32) -> u32 {
    var prices: array<f32, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) { prices[i] = 0.0; }

    var assignments: u32 = 0xffffffffu; // per-slot nibble → child index

    // Epsilon initialization per literature (prevents stalling for small n and sets scale).
    let threshold = 1.0 / f32(max(n, 1u));
    var epsilon = max(max_cost, threshold);

    while (epsilon >= threshold) {
        // Reset assignments and bidders (encode bidders as nibble stream 0..n-1)
        assignments = 0xffffffffu;
        var bidders: u32 = 0xffffffffu;
        for (var i = 0u; i < n; i = i + 1u) { set_nibble(&bidders, i, i); }
        var bidder_count = n;

        // main loop
        while (bidder_count > 0u) {
            bidder_count = bidder_count - 1u;
            let c = get_nibble(bidders, bidder_count);

            var winning_reward = -3.4e38;
            var second_reward = -3.4e38;
            var winning_slot: u32 = INVALID_ASSIGNMENT;

            for (var s = 0u; s < 4u; s = s + 1u) {
                let reward = get_cost(s, offsets[c]) - prices[s];
                if (reward > winning_reward) {
                    second_reward = winning_reward;
                    winning_reward = reward;
                    winning_slot = s;
                } else if (reward > second_reward) {
                    second_reward = reward;
                }
            }

            // Raise price so the winner becomes only epsilon better than its second-best option.
            prices[winning_slot] = prices[winning_slot] + (winning_reward - second_reward) + epsilon;

            let prev = get_nibble(assignments, winning_slot);
            set_nibble(&assignments, winning_slot, c);
            if (prev != INVALID_ASSIGNMENT) {
                set_nibble(&bidders, bidder_count, prev);
                bidder_count = bidder_count + 1u;
            }
        }

        epsilon = epsilon * INV_THETA; // shrink epsilon to tighten optimality
    }

    return assignments;
}

// One-pass, no extra global buffers: each thread picks a top-level BVH2 root and converts its subtree using a local stack
// Strategy:
// - Deterministic single-thread conversion to avoid races and ensure stable node ordering.
// - Local stack stores (bvh2_index, bvh4_index, parent ws AABB) tuples for depth-first processing.
// - Per-node, gather up to 4 children from a frontier that expands BVH2 inners while capacity
//   remains, preferring smaller surface-area first for better packing.
@compute @workgroup_size(256)
fn convert_bvh2_to_bvh4(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    // Single-thread conversion for determinism and simplicity
    if (idx != 0u) { return; }

    let scene_min = scene_aabb.min.xyz;
    let scene_ext = scene_aabb.max.xyz - scene_min;
    // Note: world-space coordinates are used directly to preserve BVH2 frame and bounds.

    // Local stacks (keep modest to avoid excessive private memory per thread)
    const MAX_STACK: u32 = 256u;
    // Stack of BVH2 node indices to be processed.
    var stack_bvh2: array<u32, MAX_STACK>;
    // Parallel stack of corresponding BVH4 node indices where results are written.
    var stack_bvh4: array<u32, MAX_STACK>;
    // Parent AABB min and extent in world-space to carry context if needed by heuristics.
    var stack_min: array<vec3<f32>, MAX_STACK>;
    var stack_ext: array<vec3<f32>, MAX_STACK>;
    var sp = 0u;

    // Allocate BVH4 node for this root
    let root4 = atomicAdd(&counters.bvh4_count, 1u);
    // Push root onto stack
    stack_bvh2[sp] = idx;
    stack_bvh4[sp] = root4;
    stack_min[sp] = scene_min;
    stack_ext[sp] = scene_ext;
    sp = sp + 1u;

    while (sp > 0u) {
        sp = sp - 1u;

        let node2_idx = stack_bvh2[sp];
        let node4_idx = stack_bvh4[sp];
        let parent_min = stack_min[sp];
        let parent_extent = stack_ext[sp];

        let node2 = bvh2_nodes[node2_idx];
        let decoded_min = node2.min_and_left_child.xyz;
        let decoded_max = node2.max_and_right_child.xyz;

        // Gather up to four children using a small frontier (top-down), adapted for BVH4
        // Expand an inner node only if there is capacity to add both of its children; otherwise,
        // take it as a whole child to respect the BVH4 fan-out limit.
        var child_indices: array<u32, 4>;
        var child_aabbs: array<AABB, 4>;
        var child_count = 0u;

        var frontier: array<u32, 8>;
        var frontier_size = 0u;

        let c0 = u32(node2.min_and_left_child.w);
        let c1 = u32(node2.max_and_right_child.w);
        if (c0 != 0xffffffffu) { frontier[frontier_size] = c0; frontier_size = frontier_size + 1u; }
        if (c1 != 0xffffffffu) { frontier[frontier_size] = c1; frontier_size = frontier_size + 1u; }

        while (child_count < 4u && frontier_size > 0u) {
            frontier_size = frontier_size - 1u;
            let ci = frontier[frontier_size];
            let cn = bvh2_nodes[ci];
            let decoded_child = AABB(vec4<f32>(cn.min_and_left_child.xyz, 0.0), vec4<f32>(cn.max_and_right_child.xyz, 0.0));

            if (!is_leaf(cn) && (child_count + frontier_size + 1u) < 4u) {
                // Expand this inner node if we still have room to add both children
                let l = u32(cn.min_and_left_child.w);
                let r = u32(cn.max_and_right_child.w);
                // Prefer expanding the child with smaller area first (like the original)
                let l_node = bvh2_nodes[l];
                let r_node = bvh2_nodes[r];
                let l_dec = AABB(vec4<f32>(l_node.min_and_left_child.xyz, 0.0), vec4<f32>(l_node.max_and_right_child.xyz, 0.0));
                let r_dec = AABB(vec4<f32>(r_node.min_and_left_child.xyz, 0.0), vec4<f32>(r_node.max_and_right_child.xyz, 0.0));
                let l_size = max(vec3<f32>(0.0), l_dec.max.xyz - l_dec.min.xyz);
                let r_size = max(vec3<f32>(0.0), r_dec.max.xyz - r_dec.min.xyz);
                let l_area = 2.0 * (l_size.x * l_size.y + l_size.x * l_size.z + l_size.y * l_size.z);
                let r_area = 2.0 * (r_size.x * r_size.y + r_size.x * r_size.z + r_size.y * r_size.z);
                let expand_right_first = r_area < l_area;
                // Push larger area last so smaller expands first
                frontier[frontier_size] = select(l, r, expand_right_first); frontier_size = frontier_size + 1u;
                frontier[frontier_size] = select(r, l, expand_right_first); frontier_size = frontier_size + 1u;
            } else {
                child_indices[child_count] = ci;
                child_aabbs[child_count] = decoded_child;
                child_count = child_count + 1u;
            }
        }

        // If nothing was gathered (defensive), skip this node
        if (child_count == 0u) {
            continue;
        }

        // Use the BVH2 node's world AABB for the BVH4 node
        let node_min_ws = decoded_min;
        let node_max_ws = decoded_max;

        var out_node: BVH4Node;
        out_node.min = vec4<f32>(node_min_ws, 0.0);
        out_node.max = vec4<f32>(node_max_ws, 0.0);
        // Children indices/handles are initialized to 0xffffffff to mark unused slots.
        out_node.children = vec4<f32>(-1.0, -1.0, -1.0, -1.0);

        // Reorder children via auction assignment and emit in slot order [0..3]
        // Resolves slot conflicts globally by raising prices to maximize total reward.
        var assignments: u32 = 0xffffffffu;
        let parent_centroid = 0.5 * (decoded_min + decoded_max);

        var offsets: array<vec3<f32>, 4>;
        var max_cost = -3.4e38;
        let max_dim = max(max(node_max_ws.x - node_min_ws.x, node_max_ws.y - node_min_ws.y), node_max_ws.z - node_min_ws.z);
        var cost_scale = select(MAX_COST / max_dim, 0.0, max_dim == 0.0);
        cost_scale = cost_scale * 0.5;
        for (var c = 0u; c < child_count; c = c + 1u) {
            let centroid = aabb_centroid(child_aabbs[c]);
            offsets[c] = (parent_centroid - centroid) * cost_scale;
            let abssum = abs(offsets[c].x) + abs(offsets[c].y) + abs(offsets[c].z);
            max_cost = select(max_cost, abssum, abssum > max_cost);
        }
        assignments = auction_assignment(offsets, max_cost, child_count);

        // Emit in slot order and push internal children
        for (var s = 0u; s < 4u; s = s + 1u) {
            let asg = get_nibble(assignments, s);
            if (asg == INVALID_ASSIGNMENT) { continue; }

            let ci = child_indices[asg];
            let cn = bvh2_nodes[ci];
            let child_min_ws = child_aabbs[asg].min.xyz;
            let child_max_ws = child_aabbs[asg].max.xyz;

            if (is_leaf(cn)) {
                // Encode leaf: high bit set | primitive index from BVH2 leaf
                // Leaves are distinguishable from internal node indices at traversal time.
                let leaf_word = u32(cn.min_and_left_child.w);
                let prim_idx = leaf_word & 0x7fffffffu;
                let leaf_handle = 0x80000000u | prim_idx;
                out_node.children[s] = f32(leaf_handle);
            } else {
                let child4 = atomicAdd(&counters.bvh4_count, 1u);
                out_node.children[s] = f32(child4);
                if (sp + 1u < MAX_STACK) {
                    stack_bvh2[sp] = ci;
                    stack_bvh4[sp] = child4;
                    stack_min[sp] = child_min_ws;
                    stack_ext[sp] = child_max_ws - child_min_ws;
                    sp = sp + 1u;
                }
            }
        }

        // Store the completed BVH4 node into the output array at the reserved index.
        bvh4_nodes[node4_idx] = out_node;
    }
}
