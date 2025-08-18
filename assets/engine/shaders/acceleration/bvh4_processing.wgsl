#include "common.wgsl"
#include "acceleration_common.wgsl"

// H-PLOC Step 2: GPU-based acceleration structure construction.

// -----------------------------------------------------------------------------
// Data Structures
// -----------------------------------------------------------------------------
struct Counters {
    bvh2_count: atomic<u32>,
    bvh4_count: atomic<u32>,
};

//------------------------------------------------------------------------------
// Bindings & Uniforms
//------------------------------------------------------------------------------
@group(1) @binding(0) var<storage, read_write> bvh2_nodes: array<BVH2Node>;
@group(1) @binding(1) var<storage, read_write> bvh4_nodes: array<BVH4Node>;
@group(1) @binding(2) var<storage, read_write> bvh4_parents: array<u32>;
@group(1) @binding(3) var<storage, read_write> counters: Counters;
@group(1) @binding(4) var<uniform> scene_aabb: AABB;

//------------------------------------------------------------------------------
// Kernel 1: BVH2 -> BVH4 Conversion
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
    let quant_scale = f32(QUANT_MAX);

    // Local stacks (keep modest to avoid excessive private memory per thread)
    const MAX_STACK: u32 = 256u;
    var stack_bvh2: array<u32, MAX_STACK>;
    var stack_bvh4: array<u32, MAX_STACK>;
    var stack_min: array<vec3<f32>, MAX_STACK>;
    var stack_ext: array<vec3<f32>, MAX_STACK>;
    var sp = 0u;

    // Allocate BVH4 node for this root
    let root4 = atomicAdd(&counters.bvh4_count, 1u);
    // root has no parent
    bvh4_parents[root4] = 0xffffffffu;
    // Push root onto stack
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
        let decoded = decode_quant_aabb(
            parent_min,
            parent_extent,
            node2.q_min_max.x,
            node2.q_min_max.y
        );
        let decoded_min = select(decoded.min.xyz, scene_aabb.min.xyz, node2_idx == gid.x);
        let decoded_max = select(decoded.max.xyz, scene_aabb.max.xyz, node2_idx == gid.x);

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
            let decoded_child = decode_quant_aabb(
                decoded_min,
                decoded_max - decoded_min,
                cn.q_min_max.x,
                cn.q_min_max.y
            );
            if (is_leaf(cn) || child_count >= 3u) {
                child_indices[child_count] = ci;
                child_aabbs[child_count] = decoded_child;
                child_count = child_count + 1u;
            } else {
                if (gather_size + 1u < 4u) {
                    gather[gather_size] = cn.children[0]; gather_size = gather_size + 1u;
                    gather[gather_size] = cn.children[1]; gather_size = gather_size + 1u;
                } else {
                    child_indices[child_count] = ci;
                    child_aabbs[child_count] = decoded_child;
                    child_count = child_count + 1u;
                }
            }
        }

        // If nothing was gathered (defensive), skip this node
        if (child_count == 0u) {
            continue;
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
        let parent_extent_safe = max(parent_extent, vec3<f32>(1e-6, 1e-6, 1e-6));
        let self_rel_min = clamp((node_min_ws - parent_min) / parent_extent_safe, vec3<f32>(0.0), vec3<f32>(1.0));
        let self_rel_max = clamp((node_max_ws - parent_min) / parent_extent_safe, vec3<f32>(0.0), vec3<f32>(1.0));
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
                // record parent pointer for BVH4 internal node
                bvh4_parents[child4] = node4_idx;
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
