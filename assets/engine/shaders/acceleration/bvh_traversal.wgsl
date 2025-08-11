#include "common.wgsl"
#include "acceleration_common.wgsl"

// Bindings for BVH traversal
@group(1) @binding(0) var<storage, read> bvh4_nodes: array<BVH4Node>;
@group(1) @binding(1) var<uniform> scene_bounds: array<vec4<f32>, 2>;
@group(1) @binding(2) var<storage, read> rays: array<Ray>; // The mesh's vertex buffer
@group(1) @binding(3) var<storage, read_write> hits: array<RayHit>;

@compute @workgroup_size(256)
fn traverse_tlas_bvh(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ray_count = arrayLength(&rays);
    if (global_id.x >= ray_count) { return; }

    let ray = rays[global_id.x];

    var hit: RayHit;
    hit.position_and_t = vec4<f32>(ray.origin_and_tmin.xyz, ray.direction_and_tmax.w);
    hit.normal_and_user_data = vec4<f32>(0.0, 0.0, 0.0, -1.0);

    // Traversal stacks
    var node_stack: array<u32, 32>;
    var min_stack:  array<vec3<f32>, 32>;
    var extent_stack: array<vec3<f32>, 32>;

    // Scene-root bounding box
    let scene_min   = scene_bounds[0].xyz;
    let scene_max   = scene_bounds[1].xyz;
    let scene_extent = scene_max - scene_min;

    // Seed the stack with the root node (index 0)
    node_stack[0]   = 0u;
    min_stack[0]    = scene_min;
    extent_stack[0] = scene_extent;
    var stack_size  = 1u;

    loop {
        if (stack_size == 0u) { break; }
        stack_size = stack_size - 1u;

        let node_idx      = node_stack[stack_size];
        if (node_idx == 0xffffffffu) { continue; }

        let parent_min    = min_stack[stack_size];
        let parent_extent = extent_stack[stack_size];
        let node = bvh4_nodes[node_idx];

        let decoded    = decode_quant_aabb(parent_min, parent_extent,
                                              node.q_min_max.x,
                                              node.q_min_max.y);
        let t_aabb    = intersect_aabb(ray, decoded.min.xyz, decoded.max.xyz);

        if (t_aabb > 0.0 && t_aabb < hit.position_and_t.w) {
            if ((node_idx & 0x80000000u) != 0u) { // leaf
                hit.position_and_t = vec4<f32>(
                    ray.origin_and_tmin.xyz + ray.direction_and_tmax.xyz * t_aabb,
                    t_aabb
                );
                hit.normal_and_user_data = vec4<f32>(0.0, 0.0, 0.0, f32(node_idx));
            } else { // internal → push
                if (stack_size < 32u) {
                    // Check all 4 children against each other to find the closest one, and swap the closest one to the front
                    var node_idx = node.children[0];
                    var node = bvh4_nodes[node_idx];
                    var decoded    = decode_quant_aabb(parent_min, parent_extent,
                                              node.q_min_max.x,
                                              node.q_min_max.y);
                    let t_aabb1    = intersect_aabb(ray, decoded.min.xyz, decoded.max.xyz);
                    node_stack[stack_size] = node_idx;
                    min_stack[stack_size] = decoded.min.xyz;
                    extent_stack[stack_size] = decoded.max.xyz - decoded.min.xyz;
                    stack_size = stack_size + 1u;

                    var min_index = 0u;
                    var min_t = t_aabb1;

                    node_idx = node.children[1];
                    node = bvh4_nodes[node_idx];
                    decoded    = decode_quant_aabb(parent_min, parent_extent,
                                              node.q_min_max.x,
                                              node.q_min_max.y);
                    let t_aabb2    = intersect_aabb(ray, decoded.min.xyz, decoded.max.xyz);
                    node_stack[stack_size] = node_idx;
                    min_stack[stack_size] = decoded.min.xyz;
                    extent_stack[stack_size] = decoded.max.xyz - decoded.min.xyz;
                    stack_size = stack_size + 1u;

                    min_index = select(min_index, 1u, t_aabb2 < min_t);
                    min_t = select(min_t, t_aabb2, t_aabb2 < min_t);

                    node_idx = node.children[2];
                    node = bvh4_nodes[node_idx];
                    decoded    = decode_quant_aabb(parent_min, parent_extent,
                                              node.q_min_max.x,
                                              node.q_min_max.y);
                    let t_aabb3    = intersect_aabb(ray, decoded.min.xyz, decoded.max.xyz);
                    node_stack[stack_size] = node_idx;
                    min_stack[stack_size] = decoded.min.xyz;
                    extent_stack[stack_size] = decoded.max.xyz - decoded.min.xyz;
                    stack_size = stack_size + 1u;

                    min_index = select(min_index, 2u, t_aabb3 < min_t);
                    min_t = select(min_t, t_aabb3, t_aabb3 < min_t);

                    node_idx = node.children[3];
                    node = bvh4_nodes[node_idx];
                    decoded    = decode_quant_aabb(parent_min, parent_extent,
                                              node.q_min_max.x,
                                              node.q_min_max.y);
                    let t_aabb4    = intersect_aabb(ray, decoded.min.xyz, decoded.max.xyz);
                    node_stack[stack_size] = node_idx;
                    min_stack[stack_size] = decoded.min.xyz;
                    extent_stack[stack_size] = decoded.max.xyz - decoded.min.xyz;
                    stack_size = stack_size + 1u;

                    min_index = select(min_index, 3u, t_aabb4 < min_t);
                    min_t = select(min_t, t_aabb4, t_aabb4 < min_t);

                    // Swap closest node to the front of the stack
                    let tmp_node = node_stack[min_index];
                    node_stack[min_index] = node_stack[stack_size - 1u];
                    node_stack[stack_size - 1u] = tmp_node;
                }
            }
        }
    }

    hits[global_id.x] = hit;
}

// TODO: Before we can get this to work, we need to find a way to attach triangle indices to this dispatch
// @compute @workgroup_size(256)
// fn traverse_triangle_bvh(@builtin(global_invocation_id) global_id: vec3<u32>) {
//     let ray_count = arrayLength(&rays);
//     if (global_id.x >= ray_count) { return; }

//     let ray = rays[global_id.x];

//     var hit: RayHit;
//     hit.position_and_t = vec4<f32>(ray.origin_and_tmin.xyz, ray.direction_and_tmax.w);
//     hit.normal_and_user_data = vec4<f32>(0.0, 0.0, 0.0, -1.0);

//     // Traversal stacks
//     var node_stack: array<u32, 32>;
//     var min_stack:  array<vec3<f32>, 32>;
//     var extent_stack: array<vec3<f32>, 32>;

//     // Scene-root bounding box
//     let scene_min   = scene_bounds[0].xyz;
//     let scene_max   = scene_bounds[1].xyz;
//     let scene_extent = scene_max - scene_min;

//     // Seed the stack with the root node (index 0)
//     node_stack[0]   = 0u;
//     min_stack[0]    = scene_min;
//     extent_stack[0] = scene_extent;
//     var stack_size  = 1u;

//     loop {
//         if (stack_size == 0u) { break; }
//         stack_size = stack_size - 1u;

//         let node_idx      = node_stack[stack_size];
//         if (node_idx == 0xffffffffu) { continue; }

//         let parent_min    = min_stack[stack_size];
//         let parent_extent = extent_stack[stack_size];
//         let node = bvh4_nodes[node_idx];

//         let decoded    = decode_quant_aabb(parent_min, parent_extent,
//                                               node.q_min_max.x,
//                                               node.q_min_max.y);
//         let t_aabb    = intersect_aabb(ray, decoded.min.xyz, decoded.max.xyz);

//         if (t_aabb > 0.0 && t_aabb < hit.position_and_t.w) {
//             if ((node_idx & 0x80000000u) != 0u) { // leaf
//                 let tri_idx = node_idx & 0x7fffffffu;

//                 // Triangle lookup
//                 let i0 = indices[tri_idx * 3u + 0u];
//                 let i1 = indices[tri_idx * 3u + 1u];
//                 let i2 = indices[tri_idx * 3u + 2u];

//                 let v0_off = i0 * SIZEOF_VERTEX;
//                 let v1_off = i1 * SIZEOF_VERTEX;
//                 let v2_off = i2 * SIZEOF_VERTEX;

//                 let v0 = vec3<f32>(vertices[v0_off + 0u], vertices[v0_off + 1u], vertices[v0_off + 2u]);
//                 let v1 = vec3<f32>(vertices[v1_off + 0u], vertices[v1_off + 1u], vertices[v1_off + 2u]);
//                 let v2 = vec3<f32>(vertices[v2_off + 0u], vertices[v2_off + 1u], vertices[v2_off + 2u]);

//                 let t_tri = intersect_triangle(ray, v0, v1, v2);
//                 if (t_tri > 0.0 && t_tri < hit.position_and_t.w) {
//                     let hit_pos = ray.origin_and_tmin.xyz + ray.direction_and_tmax.xyz * t_tri;
//                     hit.position_and_t = vec4<f32>(hit_pos, t_tri);
//                     let tri_normal = normalize(cross(v1 - v0, v2 - v0));
//                     hit.normal_and_user_data = vec4<f32>(tri_normal, f32(tri_idx));
//                 }
//             } else { // internal → push
//                 if (stack_size < 32u) {
//                     // Check all 4 children against each other to find the closest one, and swap the closest one to the front
//                     var node_idx = node.children[0];
//                     var node = bvh4_nodes[node_idx];
//                     var decoded    = decode_quant_aabb(parent_min, parent_extent,
//                                               node.q_min_max.x,
//                                               node.q_min_max.y);
//                     let t_aabb1    = intersect_aabb(ray, decoded.min.xyz, decoded.max.xyz);
//                     node_stack[stack_size] = node_idx;
//                     min_stack[stack_size] = decoded.min.xyz;
//                     extent_stack[stack_size] = decoded.max.xyz - decoded.min.xyz;
//                     stack_size = stack_size + 1u;

//                     var min_index = 0u;
//                     var min_t = t_aabb1;

//                     node_idx = node.children[1];
//                     node = bvh4_nodes[node_idx];
//                     decoded    = decode_quant_aabb(parent_min, parent_extent,
//                                               node.q_min_max.x,
//                                               node.q_min_max.y);
//                     let t_aabb2    = intersect_aabb(ray, decoded.min.xyz, decoded.max.xyz);
//                     node_stack[stack_size] = node_idx;
//                     min_stack[stack_size] = decoded.min.xyz;
//                     extent_stack[stack_size] = decoded.max.xyz - decoded.min.xyz;
//                     stack_size = stack_size + 1u;

//                     min_index = select(min_index, 1u, t_aabb2 < min_t);
//                     min_t = select(min_t, t_aabb2, t_aabb2 < min_t);

//                     node_idx = node.children[2];
//                     node = bvh4_nodes[node_idx];
//                     decoded    = decode_quant_aabb(parent_min, parent_extent,
//                                               node.q_min_max.x,
//                                               node.q_min_max.y);
//                     let t_aabb3    = intersect_aabb(ray, decoded.min.xyz, decoded.max.xyz);
//                     node_stack[stack_size] = node_idx;
//                     min_stack[stack_size] = decoded.min.xyz;
//                     extent_stack[stack_size] = decoded.max.xyz - decoded.min.xyz;
//                     stack_size = stack_size + 1u;

//                     min_index = select(min_index, 2u, t_aabb3 < min_t);
//                     min_t = select(min_t, t_aabb3, t_aabb3 < min_t);

//                     node_idx = node.children[3];
//                     node = bvh4_nodes[node_idx];
//                     decoded    = decode_quant_aabb(parent_min, parent_extent,
//                                               node.q_min_max.x,
//                                               node.q_min_max.y);
//                     let t_aabb4    = intersect_aabb(ray, decoded.min.xyz, decoded.max.xyz);
//                     node_stack[stack_size] = node_idx;
//                     min_stack[stack_size] = decoded.min.xyz;
//                     extent_stack[stack_size] = decoded.max.xyz - decoded.min.xyz;
//                     stack_size = stack_size + 1u;

//                     min_index = select(min_index, 3u, t_aabb4 < min_t);
//                     min_t = select(min_t, t_aabb4, t_aabb4 < min_t);

//                     // Swap closest node to the front of the stack
//                     let tmp_node = node_stack[min_index];
//                     node_stack[min_index] = node_stack[stack_size - 1u];
//                     node_stack[stack_size - 1u] = tmp_node;
//                 }
//             }
//         }
//     }

//     hits[global_id.x] = hit;
// }
