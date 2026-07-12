#include "ray_traversal_common.wgsl"

fn bvh_build_local_ray(ray_world: ptr<function, Ray>, entity_resolved: u32) -> Ray {
    #if RAY_TRAVERSAL_USE_RAY_INSTANCE_TRANSFORMS
    return build_local_ray_from_instance(
        ray_world,
        ray_instance_transforms[entity_resolved]
    );
    #else
    let entity_transform = entity_transforms[entity_resolved];
    return build_local_ray(
        ray_world,
        entity_transform.transform,
        entity_transform.transpose_inverse_model_matrix
    );
    #endif
}

fn bvh_trace_blas_closest(
    ray_local: ptr<function, Ray>,
    mesh_asset_id: u32
) -> RayHitCompact {
    var result = make_miss_ray_hit_compact((*ray_local).direction_and_tmax.w);
    result.mesh_id = mesh_asset_id;

    let mesh_directory_entry = blas_directory[mesh_asset_id];
    let leaf_count = mesh_directory_entry.leaf_count;
    if (leaf_count == 0u) {
        return result;
    }
    let bvh2_node_count = 2u * leaf_count - 1u;
    let root_node_idx = mesh_directory_entry.bvh2_base + bvh2_node_count - 1u;

    var current_ray = *ray_local;

    var node_stack: array<u32, NODE_STACK_SIZE>;
    node_stack[0] = root_node_idx;
    var stack_size = 1u;

    while (stack_size > 0u) {
        stack_size = stack_size - 1u;

        let node_idx = node_stack[stack_size];
        if (node_idx == INVALID_IDX) { continue; }

        let node = blas_bvh2_nodes[node_idx];
        if (is_leaf(node)) {
            let tri_id = u32(node.min.w);
            let tri_base = mesh_directory_entry.first_index + tri_id * 3u;
            let v0i = mesh_directory_entry.first_vertex + index_buffer[tri_base + 0u];
            let v1i = mesh_directory_entry.first_vertex + index_buffer[tri_base + 1u];
            let v2i = mesh_directory_entry.first_vertex + index_buffer[tri_base + 2u];
            let v0 = vertex_position(vertex_buffer[v0i]);
            let v1 = vertex_position(vertex_buffer[v1i]);
            let v2 = vertex_position(vertex_buffer[v2i]);
            let t_tri = intersect_triangle(&current_ray, v0, v1, v2);
            let is_better_hit = t_tri >= current_ray.origin_and_tmin.w && t_tri < current_ray.direction_and_tmax.w;
            if (is_better_hit) {
                result.t_hit = t_tri;
                result.tri_id_local = tri_id;
                result.tri_indices = vec4<u32>(v0i, v1i, v2i, 0u);
                result.has_hit = 1u;
                current_ray.direction_and_tmax.w = t_tri;
            }
        } else {
            var pending_child_idx = 0u;
            var pending_child_tmin = 0.0;
            var have_pending_child = false;

            let child_idx = u32(node.min.w);
            let t_aabb_child = intersect_aabb(
                &current_ray,
                blas_bvh2_nodes[child_idx].min.xyz,
                blas_bvh2_nodes[child_idx].max.xyz
            );
            let is_better_child = t_aabb_child.x <= t_aabb_child.y
                && t_aabb_child.x >= current_ray.origin_and_tmin.w
                && t_aabb_child.x < current_ray.direction_and_tmax.w;
            if (is_better_child) {
                #if BVH_TRAVERSAL_ORDER_CHILDREN
                if (!have_pending_child) {
                    pending_child_idx = child_idx;
                    pending_child_tmin = t_aabb_child.x;
                    have_pending_child = true;
                } else {
                    let current_is_farther = t_aabb_child.x > pending_child_tmin;
                    let push_idx = select(pending_child_idx, child_idx, current_is_farther);
                    let keep_idx = select(child_idx, pending_child_idx, current_is_farther);
                    let keep_tmin = select(t_aabb_child.x, pending_child_tmin, current_is_farther);

                    node_stack[stack_size] = push_idx;
                    stack_size = stack_size + 1u;

                    pending_child_idx = keep_idx;
                    pending_child_tmin = keep_tmin;
                    have_pending_child = true;
                }
                #else
                node_stack[stack_size] = child_idx;
                stack_size = stack_size + 1u;
                #endif
            }

            let child_idx_1 = u32(node.max.w);
            let t_aabb_child_1 = intersect_aabb(
                &current_ray,
                blas_bvh2_nodes[child_idx_1].min.xyz,
                blas_bvh2_nodes[child_idx_1].max.xyz
            );
            let is_better_child_1 = t_aabb_child_1.x <= t_aabb_child_1.y
                && t_aabb_child_1.x >= current_ray.origin_and_tmin.w
                && t_aabb_child_1.x < current_ray.direction_and_tmax.w;
            if (is_better_child_1) {
                #if BVH_TRAVERSAL_ORDER_CHILDREN
                if (!have_pending_child) {
                    pending_child_idx = child_idx_1;
                    pending_child_tmin = t_aabb_child_1.x;
                    have_pending_child = true;
                } else {
                    let current_is_farther = t_aabb_child_1.x > pending_child_tmin;
                    let push_idx = select(pending_child_idx, child_idx_1, current_is_farther);
                    let keep_idx = select(child_idx_1, pending_child_idx, current_is_farther);
                    let keep_tmin = select(t_aabb_child_1.x, pending_child_tmin, current_is_farther);

                    node_stack[stack_size] = push_idx;
                    stack_size = stack_size + 1u;

                    pending_child_idx = keep_idx;
                    pending_child_tmin = keep_tmin;
                    have_pending_child = true;
                }
                #else
                node_stack[stack_size] = child_idx_1;
                stack_size = stack_size + 1u;
                #endif
            }

            #if BVH_TRAVERSAL_ORDER_CHILDREN
            if (have_pending_child) {
                node_stack[stack_size] = pending_child_idx;
                stack_size = stack_size + 1u;
            }
            #endif
        }
    }

    return result;
}

fn bvh_trace_closest(ray: ptr<function, Ray>, tlas_only: bool) -> RayHitCompact {
    var result = make_miss_ray_hit_compact((*ray).direction_and_tmax.w);

    if (tlas_bvh_info.bvh2_count == 0u) {
        return result;
    }

    var node_stack: array<u32, NODE_STACK_SIZE>;
    node_stack[0] = tlas_bvh_info.bvh2_count - 1u;
    var stack_size = 1u;

    var current_ray = *ray;

    while (stack_size > 0u) {
        stack_size = stack_size - 1u;

        let node_idx = node_stack[stack_size];
        if (node_idx == INVALID_IDX) { continue; }

        let node = tlas_bvh2_bounds[node_idx];
        if (is_leaf(node)) {
            let t_leaf = intersect_aabb(&current_ray, node.min.xyz, node.max.xyz);
            if (t_leaf.x <= t_leaf.y && max(t_leaf.x, current_ray.origin_and_tmin.w) < result.t_hit) {
                let mesh_id = u32(node.min.w);
                if (mesh_id == INVALID_IDX) { continue; }

                let prim_store = u32(-node.max.w - 1.0);
                let entity_resolved = entity_index_lookup[prim_store];

                if (tlas_only) {
                    result.t_hit = max(t_leaf.x, current_ray.origin_and_tmin.w);
                    result.prim_store = prim_store;
                    result.mesh_id = mesh_id;
                    result.has_hit = 1u;
                    current_ray.direction_and_tmax.w = result.t_hit;
                } else {
                    var ray_local = bvh_build_local_ray(&current_ray, entity_resolved);
                    let blas_hit = bvh_trace_blas_closest(&ray_local, mesh_id);

                    if (blas_hit.has_hit != 0u) {
                        result = blas_hit;
                        result.prim_store = prim_store;
                        result.mesh_id = mesh_id;
                        current_ray.direction_and_tmax.w = min(current_ray.direction_and_tmax.w, result.t_hit);
                    }
                }
            }
        } else {
            var pending_child_idx = 0u;
            var pending_child_tmin = 0.0;
            var have_pending_child = false;

            let child_idx = u32(node.min.w);
            let t_aabb_child = intersect_aabb(
                &current_ray,
                tlas_bvh2_bounds[child_idx].min.xyz,
                tlas_bvh2_bounds[child_idx].max.xyz
            );
            let is_better_child = t_aabb_child.x <= t_aabb_child.y
                && t_aabb_child.x >= current_ray.origin_and_tmin.w
                && t_aabb_child.x < result.t_hit;
            if (is_better_child) {
                #if BVH_TRAVERSAL_ORDER_CHILDREN
                if (!have_pending_child) {
                    pending_child_idx = child_idx;
                    pending_child_tmin = t_aabb_child.x;
                    have_pending_child = true;
                } else {
                    let current_is_farther = t_aabb_child.x > pending_child_tmin;
                    let push_idx = select(pending_child_idx, child_idx, current_is_farther);
                    let keep_idx = select(child_idx, pending_child_idx, current_is_farther);
                    let keep_tmin = select(t_aabb_child.x, pending_child_tmin, current_is_farther);

                    node_stack[stack_size] = push_idx;
                    stack_size = stack_size + 1u;

                    pending_child_idx = keep_idx;
                    pending_child_tmin = keep_tmin;
                    have_pending_child = true;
                }
                #else
                node_stack[stack_size] = child_idx;
                stack_size = stack_size + 1u;
                #endif
            }

            let child_idx_1 = u32(node.max.w);
            let t_aabb_child_1 = intersect_aabb(
                &current_ray,
                tlas_bvh2_bounds[child_idx_1].min.xyz,
                tlas_bvh2_bounds[child_idx_1].max.xyz
            );
            let is_better_child_1 = t_aabb_child_1.x <= t_aabb_child_1.y
                && t_aabb_child_1.x >= current_ray.origin_and_tmin.w
                && t_aabb_child_1.x < result.t_hit;
            if (is_better_child_1) {
                #if BVH_TRAVERSAL_ORDER_CHILDREN
                if (!have_pending_child) {
                    pending_child_idx = child_idx_1;
                    pending_child_tmin = t_aabb_child_1.x;
                    have_pending_child = true;
                } else {
                    let current_is_farther = t_aabb_child_1.x > pending_child_tmin;
                    let push_idx = select(pending_child_idx, child_idx_1, current_is_farther);
                    let keep_idx = select(child_idx_1, pending_child_idx, current_is_farther);
                    let keep_tmin = select(t_aabb_child_1.x, pending_child_tmin, current_is_farther);

                    node_stack[stack_size] = push_idx;
                    stack_size = stack_size + 1u;

                    pending_child_idx = keep_idx;
                    pending_child_tmin = keep_tmin;
                    have_pending_child = true;
                }
                #else
                node_stack[stack_size] = child_idx_1;
                stack_size = stack_size + 1u;
                #endif
            }

            #if BVH_TRAVERSAL_ORDER_CHILDREN
            if (have_pending_child) {
                node_stack[stack_size] = pending_child_idx;
                stack_size = stack_size + 1u;
            }
            #endif
        }
    }

    return result;
}

fn bvh_trace_blas_any(
    ray_local: ptr<function, Ray>,
    mesh_asset_id: u32,
) -> bool {
    let mesh_directory_entry = blas_directory[mesh_asset_id];
    let leaf_count = mesh_directory_entry.leaf_count;
    if (leaf_count == 0u) {
        return false;
    }
    let bvh2_node_count = 2u * leaf_count - 1u;
    let root_node_idx = mesh_directory_entry.bvh2_base + bvh2_node_count - 1u;

    var node_stack: array<u32, NODE_STACK_SIZE>;
    node_stack[0] = root_node_idx;
    var stack_size = 1u;

    loop {
        if (stack_size == 0u) { break; }
        stack_size = stack_size - 1u;

        let node_idx = node_stack[stack_size];
        if (node_idx == INVALID_IDX) { continue; }

        let node = blas_bvh2_nodes[node_idx];
        if (is_leaf(node)) {
            let tri_id = u32(node.min.w);
            let tri_base = mesh_directory_entry.first_index + tri_id * 3u;
            let v0i = mesh_directory_entry.first_vertex + index_buffer[tri_base + 0u];
            let v1i = mesh_directory_entry.first_vertex + index_buffer[tri_base + 1u];
            let v2i = mesh_directory_entry.first_vertex + index_buffer[tri_base + 2u];
            let v0 = vertex_position(vertex_buffer[v0i]);
            let v1 = vertex_position(vertex_buffer[v1i]);
            let v2 = vertex_position(vertex_buffer[v2i]);
            let t_tri = intersect_triangle(ray_local, v0, v1, v2);
            if (t_tri >= ray_local.origin_and_tmin.w && t_tri < ray_local.direction_and_tmax.w) {
                return true;
            }
        } else {
            var pending_child_idx = 0u;
            var pending_child_tmin = 0.0;
            var have_pending_child = false;

            let child_idx = u32(node.min.w);
            let t_aabb_child = intersect_aabb(ray_local, blas_bvh2_nodes[child_idx].min.xyz, blas_bvh2_nodes[child_idx].max.xyz);
            let is_better_child = t_aabb_child.x <= t_aabb_child.y && t_aabb_child.x >= ray_local.origin_and_tmin.w && t_aabb_child.x < ray_local.direction_and_tmax.w;
            if (is_better_child) {
                #if BVH_TRAVERSAL_ORDER_CHILDREN
                if (!have_pending_child) {
                    pending_child_idx = child_idx;
                    pending_child_tmin = t_aabb_child.x;
                    have_pending_child = true;
                } else {
                    let current_is_farther = t_aabb_child.x > pending_child_tmin;
                    let push_idx = select(pending_child_idx, child_idx, current_is_farther);
                    let keep_idx = select(child_idx, pending_child_idx, current_is_farther);
                    let keep_tmin = select(t_aabb_child.x, pending_child_tmin, current_is_farther);

                    node_stack[stack_size] = push_idx;
                    stack_size = stack_size + 1u;

                    pending_child_idx = keep_idx;
                    pending_child_tmin = keep_tmin;
                    have_pending_child = true;
                }
                #else
                node_stack[stack_size] = child_idx;
                stack_size = stack_size + 1u;
                #endif
            }

            let child_idx_1 = u32(node.max.w);
            let t_aabb_child_1 = intersect_aabb(ray_local, blas_bvh2_nodes[child_idx_1].min.xyz, blas_bvh2_nodes[child_idx_1].max.xyz);
            let is_better_child_1 = t_aabb_child_1.x <= t_aabb_child_1.y && t_aabb_child_1.x >= ray_local.origin_and_tmin.w && t_aabb_child_1.x < ray_local.direction_and_tmax.w;
            if (is_better_child_1) {
                #if BVH_TRAVERSAL_ORDER_CHILDREN
                if (!have_pending_child) {
                    pending_child_idx = child_idx_1;
                    pending_child_tmin = t_aabb_child_1.x;
                    have_pending_child = true;
                } else {
                    let current_is_farther = t_aabb_child_1.x > pending_child_tmin;
                    let push_idx = select(pending_child_idx, child_idx_1, current_is_farther);
                    let keep_idx = select(child_idx_1, pending_child_idx, current_is_farther);
                    let keep_tmin = select(t_aabb_child_1.x, pending_child_tmin, current_is_farther);

                    node_stack[stack_size] = push_idx;
                    stack_size = stack_size + 1u;

                    pending_child_idx = keep_idx;
                    pending_child_tmin = keep_tmin;
                    have_pending_child = true;
                }
                #else
                node_stack[stack_size] = child_idx_1;
                stack_size = stack_size + 1u;
                #endif
            }

            #if BVH_TRAVERSAL_ORDER_CHILDREN
            if (have_pending_child) {
                node_stack[stack_size] = pending_child_idx;
                stack_size = stack_size + 1u;
            }
            #endif
        }
    }

    return false;
}

fn bvh_trace_any(ray: ptr<function, Ray>) -> bool {
    if (tlas_bvh_info.bvh2_count == 0u) {
        return false;
    }

    var node_stack: array<u32, NODE_STACK_SIZE>;
    node_stack[0] = tlas_bvh_info.bvh2_count - 1u;
    var stack_size = 1u;

    var current_ray = *ray;

    loop {
        if (stack_size == 0u) { break; }
        stack_size = stack_size - 1u;

        let node_idx = node_stack[stack_size];
        if (node_idx == INVALID_IDX) { continue; }

        let current_node = tlas_bvh2_bounds[node_idx];
        if (is_leaf(current_node)) {
            let t_leaf = intersect_aabb(&current_ray, current_node.min.xyz, current_node.max.xyz);

            if (t_leaf.x <= t_leaf.y && max(t_leaf.x, current_ray.origin_and_tmin.w) < current_ray.direction_and_tmax.w) {
                let mesh_id = u32(current_node.min.w);
                if (mesh_id == INVALID_IDX) { continue; }

                let prim_store = u32(-current_node.max.w - 1.0);
                let entity_resolved = entity_index_lookup[prim_store];
                var ray_local = bvh_build_local_ray(&current_ray, entity_resolved);

                if (bvh_trace_blas_any(&ray_local, mesh_id)) {
                    return true;
                }
            }
        } else {
            var pending_child_idx = 0u;
            var pending_child_tmin = 0.0;
            var have_pending_child = false;

            let child_idx = u32(current_node.min.w);
            let t_aabb_child = intersect_aabb(&current_ray, tlas_bvh2_bounds[child_idx].min.xyz, tlas_bvh2_bounds[child_idx].max.xyz);
            let is_better_child = t_aabb_child.x <= t_aabb_child.y && max(t_aabb_child.x, current_ray.origin_and_tmin.w) < current_ray.direction_and_tmax.w;
            if (is_better_child) {
                #if BVH_TRAVERSAL_ORDER_CHILDREN
                if (!have_pending_child) {
                    pending_child_idx = child_idx;
                    pending_child_tmin = t_aabb_child.x;
                    have_pending_child = true;
                } else {
                    let current_is_farther = t_aabb_child.x > pending_child_tmin;
                    let push_idx = select(pending_child_idx, child_idx, current_is_farther);
                    let keep_idx = select(child_idx, pending_child_idx, current_is_farther);
                    let keep_tmin = select(t_aabb_child.x, pending_child_tmin, current_is_farther);

                    node_stack[stack_size] = push_idx;
                    stack_size = stack_size + 1u;

                    pending_child_idx = keep_idx;
                    pending_child_tmin = keep_tmin;
                    have_pending_child = true;
                }
                #else
                node_stack[stack_size] = child_idx;
                stack_size = stack_size + 1u;
                #endif
            }

            let child_idx_1 = u32(current_node.max.w);
            let t_aabb_child_1 = intersect_aabb(&current_ray, tlas_bvh2_bounds[child_idx_1].min.xyz, tlas_bvh2_bounds[child_idx_1].max.xyz);
            let is_better_child_1 = t_aabb_child_1.x <= t_aabb_child_1.y && max(t_aabb_child_1.x, current_ray.origin_and_tmin.w) < current_ray.direction_and_tmax.w;
            if (is_better_child_1) {
                #if BVH_TRAVERSAL_ORDER_CHILDREN
                if (!have_pending_child) {
                    pending_child_idx = child_idx_1;
                    pending_child_tmin = t_aabb_child_1.x;
                    have_pending_child = true;
                } else {
                    let current_is_farther = t_aabb_child_1.x > pending_child_tmin;
                    let push_idx = select(pending_child_idx, child_idx_1, current_is_farther);
                    let keep_idx = select(child_idx_1, pending_child_idx, current_is_farther);
                    let keep_tmin = select(t_aabb_child_1.x, pending_child_tmin, current_is_farther);

                    node_stack[stack_size] = push_idx;
                    stack_size = stack_size + 1u;

                    pending_child_idx = keep_idx;
                    pending_child_tmin = keep_tmin;
                    have_pending_child = true;
                }
                #else
                node_stack[stack_size] = child_idx_1;
                stack_size = stack_size + 1u;
                #endif
            }

            #if BVH_TRAVERSAL_ORDER_CHILDREN
            if (have_pending_child) {
                node_stack[stack_size] = pending_child_idx;
                stack_size = stack_size + 1u;
            }
            #endif
        }
    }

    return false;
}

fn trace_ray_closest(ray: ptr<function, Ray>) -> RayHitCompact {
    return bvh_trace_closest(ray, false);
}

fn trace_ray_closest_tlas(ray: ptr<function, Ray>, tlas_only: bool) -> RayHitCompact {
    return bvh_trace_closest(ray, tlas_only);
}

fn trace_ray_any(ray: ptr<function, Ray>) -> bool {
    return bvh_trace_any(ray);
}
