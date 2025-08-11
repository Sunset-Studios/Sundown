#include "common.wgsl"
#include "acceleration_common.wgsl"

// H-PLOC Step 2: GPU-based acceleration structure construction.

// -----------------------------------------------------------------------------
// Data Structures
// -----------------------------------------------------------------------------
struct Cluster {
    aabb_min_and_node_idx: vec4<f32>,
    aabb_max_and_is_active: vec4<f32>,
};

struct HPLOCUniforms {
    primitive_count: u32,
    pass_num: u32,
};

struct Counters {
    bvh2_count: atomic<u32>,
    bvh4_count: atomic<u32>,
};

//------------------------------------------------------------------------------
// Utility Functions
//------------------------------------------------------------------------------
fn is_leaf(node: BVH2Node) -> bool {
    return (node.children[0] & 0x80000000u) != 0u;
}

//------------------------------------------------------------------------------
// Bindings & Uniforms
//------------------------------------------------------------------------------
@group(1) @binding(0) var<storage, read> bounds: array<AABB>;
@group(1) @binding(1) var<storage, read_write> sorted_indices: array<u32>;
@group(1) @binding(2) var<storage, read_write> bvh2_nodes: array<BVH2Node>;
@group(1) @binding(3) var<storage, read_write> bvh4_nodes: array<BVH4Node>;
@group(1) @binding(4) var<storage, read_write> counters: Counters;
@group(1) @binding(5) var<uniform> hploc_uniforms: HPLOCUniforms;
@group(1) @binding(6) var<storage, read_write> clusters_in: array<Cluster>;
@group(1) @binding(7) var<storage, read_write> clusters_out: array<Cluster>;
@group(1) @binding(8) var<uniform> scene_aabb: AABB;

//------------------------------------------------------------------------------
// Kernel 1: BVH2 Construction using H-PLOC
//------------------------------------------------------------------------------

var<workgroup> wg_clusters: array<Cluster, WAVE_SIZE>;
var<workgroup> wg_best_costs: array<f32, WAVE_SIZE>;
var<workgroup> wg_best_pairs: array<vec2<i32>, WAVE_SIZE>;
var<workgroup> wg_selected_pair: vec2<i32>;

@compute @workgroup_size(1)
fn initialize_leaf_clusters(@builtin(global_invocation_id) gid: vec3<u32>) {
    let prim_idx = gid.x;
    if (prim_idx >= hploc_uniforms.primitive_count) { return; }

    let sorted_prim_idx = sorted_indices[prim_idx];
    let bound = bounds[sorted_prim_idx];

    let leaf_node_idx = atomicAdd(&counters.bvh2_count, 1u);

    let child1 = 0x80000000u | sorted_prim_idx;
    let child2 = 0xffffffffu;

    // q_min_max will be filled later when this leaf is attached to a parent
    bvh2_nodes[leaf_node_idx].q_min_max = vec2<u32>(0u, 0u);
    bvh2_nodes[leaf_node_idx].children = vec2<u32>(child1, child2);

    clusters_in[prim_idx].aabb_min_and_node_idx = vec4<f32>(bound.min.xyz, f32(leaf_node_idx));
    clusters_in[prim_idx].aabb_max_and_is_active = vec4<f32>(bound.max.xyz, 1.0);
}


@compute @workgroup_size(WAVE_SIZE)
fn build_bvh2_hploc(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let wave_idx = group_id.x;
    let thread_idx = local_id.x;
    let num_clusters_in = hploc_uniforms.primitive_count;
    let quant_scale = f32(QUANT_MAX);

    // 1. Load wave's clusters into shared memory
    let cluster_global_idx = wave_idx * WAVE_SIZE + thread_idx;
    if (cluster_global_idx < num_clusters_in) {
        wg_clusters[thread_idx] = clusters_in[cluster_global_idx];
    } else {
        wg_clusters[thread_idx].aabb_max_and_is_active.w = 0.0;
    }
    workgroupBarrier();

    // 2. Merge clusters within the wave using parallel reduction
    var active_count = min(WAVE_SIZE, num_clusters_in - wave_idx * WAVE_SIZE);
    for (var i = 0u; i < WAVE_SIZE - 1u && active_count > 1u; i++) {
        var local_best_cost: f32 = 1e38;
        var local_best_pair: vec2<i32> = vec2<i32>(-1, -1);

        // Each active thread evaluates pairs between its cluster and all other active clusters
        if (wg_clusters[thread_idx].aabb_max_and_is_active.w != 0.0) {
            for (var k = 0u; k < WAVE_SIZE; k = k + 1u) {
                if (k == thread_idx) { continue; }
                if (wg_clusters[k].aabb_max_and_is_active.w == 0.0) { continue; }

                let merged = merge_aabbs(
                    wg_clusters[thread_idx].aabb_min_and_node_idx.xyz,
                    wg_clusters[thread_idx].aabb_max_and_is_active.xyz,
                    wg_clusters[k].aabb_min_and_node_idx.xyz,
                    wg_clusters[k].aabb_max_and_is_active.xyz
                );
                let cost = calculate_aabb_surface_area(merged.min.xyz, merged.max.xyz);

                if (cost < local_best_cost) {
                    local_best_cost = cost;
                    local_best_pair = vec2<i32>(i32(thread_idx), i32(k));
                }
            }
        }

        // Write to shared memory
        wg_best_costs[thread_idx] = local_best_cost;
        wg_best_pairs[thread_idx] = local_best_pair; 

        workgroupBarrier();

        // Reduction: stride halves each iteration
        var stride = WAVE_SIZE / 2u;
        while (stride > 0u) {
            if (thread_idx < stride) {
                if (wg_best_costs[thread_idx + stride] < wg_best_costs[thread_idx]) {
                    wg_best_costs[thread_idx] = wg_best_costs[thread_idx + stride];
                    wg_best_pairs[thread_idx] = wg_best_pairs[thread_idx + stride];
                }
            }
            workgroupBarrier();
            stride = stride / 2u;
        }

        // Thread 0 publishes the selected pair
        if (thread_idx == 0u) {
            wg_selected_pair = wg_best_pairs[0];
        }
        workgroupBarrier();

        var merge_idx1 = wg_selected_pair.x;
        var merge_idx2 = wg_selected_pair.y;

        let merged = merge_aabbs(
            wg_clusters[u32(merge_idx1)].aabb_min_and_node_idx.xyz,
            wg_clusters[u32(merge_idx1)].aabb_max_and_is_active.xyz,
            wg_clusters[u32(merge_idx2)].aabb_min_and_node_idx.xyz,
            wg_clusters[u32(merge_idx2)].aabb_max_and_is_active.xyz
        );
        let child1 = u32(wg_clusters[u32(merge_idx1)].aabb_min_and_node_idx.w);
        let child2 = u32(wg_clusters[u32(merge_idx2)].aabb_min_and_node_idx.w);
          
        let parent_node_idx = atomicAdd(&counters.bvh2_count, 1u);

        // Write the merged AABB to the parent node
        let parent_min = merged.min.xyz;
        let parent_max = merged.max.xyz;
        let parent_extent = parent_max - parent_min;

        bvh2_nodes[parent_node_idx].children = vec2<u32>(child1, child2);

        // Encode child 0 quantisation
        let c0_min = wg_clusters[u32(merge_idx1)].aabb_min_and_node_idx.xyz;
        let c0_max = wg_clusters[u32(merge_idx1)].aabb_max_and_is_active.xyz;
        let rel0_min = (c0_min - parent_min) / parent_extent;
        let rel0_max = (c0_max - parent_min) / parent_extent;
        let q0_min = pack_quant3(
            u32(round(rel0_min.x * quant_scale)),
            u32(round(rel0_min.y * quant_scale)),
            u32(round(rel0_min.z * quant_scale))
        );
        let q0_max = pack_quant3(
            u32(round(rel0_max.x * quant_scale)),
            u32(round(rel0_max.y * quant_scale)),
            u32(round(rel0_max.z * quant_scale))
        );

        // Encode child 1 quantisation
        let c1_min = wg_clusters[u32(merge_idx2)].aabb_min_and_node_idx.xyz;
        let c1_max = wg_clusters[u32(merge_idx2)].aabb_max_and_is_active.xyz;
        let rel1_min = (c1_min - parent_min) / parent_extent;
        let rel1_max = (c1_max - parent_min) / parent_extent;
        let q1_min = pack_quant3(
            u32(round(rel1_min.x * quant_scale)),
            u32(round(rel1_min.y * quant_scale)),
            u32(round(rel1_min.z * quant_scale))
        );
        let q1_max = pack_quant3(
            u32(round(rel1_max.x * quant_scale)),
            u32(round(rel1_max.y * quant_scale)),
            u32(round(rel1_max.z * quant_scale))
        );

        // Write the quantised AABBs (relative to parent) to the children
        bvh2_nodes[child1].q_min_max = vec2<u32>(q0_min, q0_max);
        bvh2_nodes[child2].q_min_max = vec2<u32>(q1_min, q1_max);

        // Update the surviving cluster
        if (thread_idx == u32(merge_idx1)) {
            wg_clusters[u32(merge_idx1)].aabb_min_and_node_idx = vec4<f32>(merged.min.xyz, f32(parent_node_idx));
            wg_clusters[u32(merge_idx1)].aabb_max_and_is_active = vec4<f32>(merged.max.xyz, 1.0);
        }
        // Mark the merged cluster as inactive
        if (thread_idx == u32(merge_idx2)) {
            wg_clusters[u32(merge_idx2)].aabb_max_and_is_active.w = 0.0;
        }

        // Update the active count
        active_count--;

        // Wait for all threads to complete the merge
        workgroupBarrier();
    }

    // 3. Write surviving cluster back to global memory
    if (wg_clusters[thread_idx].aabb_max_and_is_active.w > 0.0) {
        clusters_out[wave_idx * WAVE_SIZE + thread_idx] = wg_clusters[thread_idx]; // Each wave produces one cluster
    }
}

//------------------------------------------------------------------------------
// Kernel 2: BVH2 -> BVH4 Conversion
//------------------------------------------------------------------------------

// One-pass, no extra global buffers: each thread picks a top-level BVH2 root and converts its subtree using a local stack
@compute @workgroup_size(256)
fn convert_bvh2_to_bvh4(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total_nodes = atomicLoad(&counters.bvh2_count);
    if (idx >= total_nodes) { return; }

    let root2 = bvh2_nodes[idx];
    if (is_leaf(root2)) { return; }
    // Only start at top-level BVH2 roots (nodes with no parent quant set)
    if (!(root2.q_min_max.x == 0u && root2.q_min_max.y == 0u)) { return; }

    let scene_min = scene_aabb.min.xyz;
    let scene_ext = scene_aabb.max.xyz - scene_min;

    // Local stacks
    const MAX_STACK: u32 = 2048u;
    var stack_bvh2: array<u32, MAX_STACK>;
    var stack_bvh4: array<u32, MAX_STACK>;
    var stack_min: array<vec3<f32>, MAX_STACK>;
    var stack_ext: array<vec3<f32>, MAX_STACK>;
    var sp = 0u;

    // Allocate BVH4 node for this root
    let root4 = atomicAdd(&counters.bvh4_count, 1u);
    stack_bvh2[sp] = idx;
    stack_bvh4[sp] = root4;
    stack_min[sp] = scene_min;
    stack_ext[sp] = scene_ext;
    sp = sp + 1u;

    loop {
        if (sp == 0u) { break; }
        sp = sp - 1u;
        let node2_idx = stack_bvh2[sp];
        let node4_idx = stack_bvh4[sp];
        let parent_min = stack_min[sp];
        let parent_extent = stack_ext[sp];

        let node2 = bvh2_nodes[node2_idx];

        // Gather up to four children
        var child_indices: array<u32, 4>;
        var child_aabbs: array<AABB, 4>;
        var child_count = 0u;

        var gather: array<u32, 4>;
        var gather_size = 0u;
        let c0 = node2.children[0];
        let c1 = node2.children[1];
        if (c0 != 0xffffffffu) { gather[gather_size] = c0; gather_size = gather_size + 1u; }
        if (c1 != 0xffffffffu) { gather[gather_size] = c1; gather_size = gather_size + 1u; }

        for (var i = 0u; i < gather_size; i = i + 1u) {
            let ci = gather[i];
            let cn = bvh2_nodes[ci];
            let decoded = decode_quant_aabb(parent_min, parent_extent, cn.q_min_max.x, cn.q_min_max.y);
            if (is_leaf(cn) || child_count >= 3u) {
                child_indices[child_count] = ci;
                child_aabbs[child_count] = decoded;
                child_count = child_count + 1u;
            } else {
                if (gather_size + 1u < 4u) {
                    gather[gather_size] = cn.children[0]; gather_size = gather_size + 1u;
                    gather[gather_size] = cn.children[1]; gather_size = gather_size + 1u;
                } else {
                    child_indices[child_count] = ci;
                    child_aabbs[child_count] = decoded;
                    child_count = child_count + 1u;
                }
            }
        }

        // Compute this BVH4 node's world AABB
        var node_min_ws = child_aabbs[0].min.xyz;
        var node_max_ws = child_aabbs[0].max.xyz;
        for (var j = 1u; j < child_count; j = j + 1u) {
            node_min_ws = vec3<f32>(
                min(node_min_ws.x, child_aabbs[j].min.x),
                min(node_min_ws.y, child_aabbs[j].min.y),
                min(node_min_ws.z, child_aabbs[j].min.z)
            );
            node_max_ws = vec3<f32>(
                max(node_max_ws.x, child_aabbs[j].max.x),
                max(node_max_ws.y, child_aabbs[j].max.y),
                max(node_max_ws.z, child_aabbs[j].max.z)
            );
        }

        // Encode relative to parent
        let self_rel_min = (node_min_ws - parent_min) / parent_extent;
        let self_rel_max = (node_max_ws - parent_min) / parent_extent;
        let self_qmin = pack_quant3(
            u32(round(self_rel_min.x * quant_scale)),
            u32(round(self_rel_min.y * quant_scale)),
            u32(round(self_rel_min.z * quant_scale))
        );
        let self_qmax = pack_quant3(
            u32(round(self_rel_max.x * quant_scale)),
            u32(round(self_rel_max.y * quant_scale)),
            u32(round(self_rel_max.z * quant_scale))
        );

        var out_node: BVH4Node;
        out_node.q_min_max = vec2<u32>(self_qmin, self_qmax);
        out_node.children = vec4<u32>(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu);

        // Emit children; push internal onto local stack
        for (var k = 0u; k < child_count; k = k + 1u) {
            let ci = child_indices[k];
            let cn = bvh2_nodes[ci];
            let child_min_ws = child_aabbs[k].min.xyz;
            let child_max_ws = child_aabbs[k].max.xyz;
            if (is_leaf(cn)) {
                out_node.children[k] = ci;
            } else {
                let child4 = atomicAdd(&counters.bvh4_count, 1u);
                out_node.children[k] = child4;
                if (sp + 1u < MAX_STACK) {
                    stack_bvh2[sp] = ci;
                    stack_bvh4[sp] = child4;
                    stack_min[sp] = child_min_ws;
                    stack_ext[sp] = child_max_ws - child_min_ws;
                    sp = sp + 1u;
                }
            }
        }

        bvh4_nodes[node4_idx] = out_node;
    }
}
