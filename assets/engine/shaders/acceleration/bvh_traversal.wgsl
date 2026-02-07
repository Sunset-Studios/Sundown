#include "common.wgsl"
#include "acceleration_common.wgsl"

// Bindings for BVH traversal
@group(1) @binding(0) var<uniform> bvh_info: BVHInfo;
@group(1) @binding(1) var<storage, read> rays: array<Ray>; // The mesh's vertex buffer
@group(1) @binding(2) var<storage, read_write> hits: array<RayHit>;
@group(1) @binding(3) var<storage, read> bvh2_bounds: array<AABB>;

@compute @workgroup_size(256)
fn traverse_tlas_bvh(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ray_count = arrayLength(&rays);
    if (global_id.x >= ray_count) { return; }

    let ray = rays[global_id.x];

    var hit: RayHit;
    hit.position_and_t = vec4<f32>(ray.origin_and_tmin.xyz, ray.direction_and_tmax.w);
    hit.normal_and_user_data = vec4<f32>(0.0, 0.0, 0.0, -1.0);

    if (bvh_info.bvh2_count == 0u) {
        hits[global_id.x] = hit;
        return;
    }

    var node_stack: array<u32, 32>;
    node_stack[0] = bvh_info.bvh2_count - 1u;
    var stack_size  = 1u;

    loop {
        if (stack_size == 0u) { break; }
        stack_size = stack_size - 1u;

        var node_idx = node_stack[stack_size];
        if (node_idx == INVALID_IDX) { continue; }

        let node = bvh2_bounds[node_idx];
        let t_aabb = intersect_aabb(&ray, node.min.xyz, node.max.xyz);

        if (t_aabb.y >= t_aabb.x && t_aabb.x >= ray.origin_and_tmin.w && t_aabb.x < hit.position_and_t.w) {
            if (stack_size < 32u) {
                if (is_leaf(node)) {
                    let t_leaf = intersect_aabb(&ray, node.min.xyz, node.max.xyz);
                    if (t_leaf.y >= t_leaf.x && t_leaf.x >= ray.origin_and_tmin.w && t_leaf.x < hit.position_and_t.w) {
                        let prim = u32(-node.max.w - 1.0);
                        hit.position_and_t = vec4<f32>(
                            ray.origin_and_tmin.xyz + ray.direction_and_tmax.xyz * t_leaf.x,
                            t_leaf.x
                        );
                        hit.normal_and_user_data = vec4<f32>(0.0, 0.0, 0.0, f32(prim));
                    }
                } else {
                    let left_idx = u32(node.min.w);
                    let left_node = bvh2_bounds[left_idx];
                    let t_aabb_left = intersect_aabb(&ray, left_node.min.xyz, left_node.max.xyz);
                    if (t_aabb_left.y >= t_aabb_left.x && t_aabb_left.x >= ray.origin_and_tmin.w && t_aabb_left.x < hit.position_and_t.w) {
                        node_stack[stack_size] = left_idx;
                        stack_size = stack_size + 1u;
                    }

                    let right_idx = u32(node.max.w);
                    let right_node = bvh2_bounds[right_idx];
                    let t_aabb_right = intersect_aabb(&ray, right_node.min.xyz, right_node.max.xyz);
                    if (t_aabb_right.y >= t_aabb_right.x && t_aabb_right.x >= ray.origin_and_tmin.w && t_aabb_right.x < hit.position_and_t.w) {
                        node_stack[stack_size] = right_idx;
                        stack_size = stack_size + 1u;
                    }
                }
            }
        }
    }

    hits[global_id.x] = hit;
}
