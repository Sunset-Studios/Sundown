// Ray structure
struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
    tmin: f32,
    tmax: f32,
    inv_direction: vec3<f32>,
}

// Returns true if the ray hits any entity, and outputs the user_data (entity id)
fn raycast_aabb_tree(ray: Ray, max_traversal_steps: u32) -> u32 {
    var hit_entity: u32 = 0xffffffffu;
    var stack: array<u32, 64>;
    var stack_ptr = 0u;
    stack[stack_ptr] = 0u;
    stack_ptr += 1u;
    var traversal_steps = 0u;
    while (stack_ptr > 0u && traversal_steps < max_traversal_steps) {
        traversal_steps += 1u;
        stack_ptr -= 1u;
        let node_index = stack[stack_ptr];
        let node = aabb_nodes[node_index];
        let bounds = aabb_bounds[node_index];
        // Skip free nodes
        if ((u32(node.flags_and_node_data.x) & AABB_NODE_FLAGS_FREE) != 0u) {
            continue;
        }
        // Ray-AABB intersection
        let t = ray_aabb_intersection(ray, bounds.min_point.xyz, bounds.max_point.xyz);
        if (t < 0.0 || t > ray.tmax) {
            continue;
        }
        if (u32(node.flags_and_node_data.y) == AABB_NODE_TYPE_LEAF) {
            // Hit! Return user_data (entity id)
            hit_entity = u32(node.left_right_parent_ud.w);
            break;
        } else {
            let left_child = u32(node.left_right_parent_ud.x);
            let right_child = u32(node.left_right_parent_ud.y);
            if (left_child != 0u && stack_ptr < 64u) {
                stack[stack_ptr] = left_child;
                stack_ptr += 1u;
            }
            if (right_child != 0u && stack_ptr < 64u) {
                stack[stack_ptr] = right_child;
                stack_ptr += 1u;
            }
        }
    }
    return hit_entity;
}

// Ray-AABB intersection (returns tmin or -1.0 if no hit)
fn ray_aabb_intersection(ray: Ray, min_point: vec3<f32>, max_point: vec3<f32>) -> f32 {
    var tmin = ray.tmin;
    var tmax = ray.tmax;
    for (var i = 0; i < 3; i++) {
        let inv_d = ray.inv_direction[i];
        var t1 = (min_point[i] - ray.origin[i]) * inv_d;
        var t2 = (max_point[i] - ray.origin[i]) * inv_d;
        if (inv_d < 0.0) {
            let temp = t1;
            t1 = t2;
            t2 = temp;
        }
        tmin = max(tmin, t1);
        tmax = min(tmax, t2);
        if (tmax < tmin) {
            return -1.0;
        }
    }
    return tmin;
}

