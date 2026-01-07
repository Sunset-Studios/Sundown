#include "common.wgsl"
#include "acceleration_common.wgsl"

// Bindings for BVH traversal
@group(1) @binding(0) var<storage, read> bvh8_nodes: array<BVH8Node>;
@group(1) @binding(1) var<storage, read> bvh8_prim_indices: array<u32>;
@group(1) @binding(2) var<uniform> scene_bounds: array<vec4<f32>, 2>;
@group(1) @binding(3) var<storage, read> rays: array<Ray>; // The mesh's vertex buffer
@group(1) @binding(4) var<storage, read_write> hits: array<RayHit>;
@group(1) @binding(5) var<storage, read> bvh2_bounds: array<AABB>;

@compute @workgroup_size(256)
fn traverse_tlas_bvh(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ray_count = arrayLength(&rays);
    if (global_id.x >= ray_count) { return; }

    let ray = rays[global_id.x];

    var hit: RayHit;
    hit.position_and_t = vec4<f32>(ray.origin_and_tmin.xyz, ray.direction_and_tmax.w);
    hit.normal_and_user_data = vec4<f32>(0.0, 0.0, 0.0, -1.0);

    var node_stack: array<u32, 32>;
    node_stack[0]   = 0u;
    var stack_size  = 1u;

    loop {
        if (stack_size == 0u) { break; }
        stack_size = stack_size - 1u;

        var node_idx = node_stack[stack_size];
        if (node_idx == INVALID_IDX) { continue; }

        var node = bvh8_nodes[node_idx];
        let t_aabb = intersect_aabb(&ray, node.min.xyz, node.max.xyz);

        if (t_aabb.y >= t_aabb.x && t_aabb.x >= ray.origin_and_tmin.w && t_aabb.x < hit.position_and_t.w) {
            if (stack_size < 32u) {
                let leaf_mask = bitcast<u32>(node.min.w);

                for (var i = 0u; i < 8u; i = i + 1u) {
                    let child_raw = bvh8_child(&node, i);
                    if (child_raw < 0.0) { continue; }

                    let is_leaf_child = ((leaf_mask >> i) & 1u) != 0u;
                    let child_idx = u32(child_raw);

                    if (is_leaf_child) {
                        let leaf_bounds = bvh2_bounds[child_idx];
                        let t_leaf = intersect_aabb(&ray, leaf_bounds.min.xyz, leaf_bounds.max.xyz);
                        if (t_leaf.y >= t_leaf.x && t_leaf.x >= ray.origin_and_tmin.w && t_leaf.x < hit.position_and_t.w) {
                            let prim = u32(leaf_bounds.min.w);
                            hit.position_and_t = vec4<f32>(
                                ray.origin_and_tmin.xyz + ray.direction_and_tmax.xyz * t_leaf.x,
                                t_leaf.x
                            );
                            hit.normal_and_user_data = vec4<f32>(0.0, 0.0, 0.0, f32(prim));
                        }
                    } else {
                        let child_node = bvh8_nodes[child_idx];
                        let t_aabb_child = intersect_aabb(&ray, child_node.min.xyz, child_node.max.xyz);

                        if (t_aabb_child.y >= t_aabb_child.x && t_aabb_child.x >= ray.origin_and_tmin.w && t_aabb_child.x < hit.position_and_t.w) {
                            node_stack[stack_size] = child_idx;
                            stack_size = stack_size + 1u;
                        }
                    }
                }
            }
        }
    }

    hits[global_id.x] = hit;
}
