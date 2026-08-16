const BVH_TRAVERSAL_WORKGROUP_SIZE = 128u;

// BLAS traversal is nested inside TLAS traversal, so only the deeper BLAS stack moves to
// workgroup memory. Lane-interleaving keeps each wave's stack accesses bank-friendly while
// avoiding the dynamically indexed private array that commonly spills to thread-local memory.
var<workgroup> bvh_blas_node_stack: array<u32, NODE_STACK_SIZE * BVH_TRAVERSAL_WORKGROUP_SIZE>;
var<private> bvh_stack_lane: u32;

#if BVH_TRAVERSAL_COLLECT_STATS
struct BVHTraversalStats {
    tlas_aabb_tests: u32,
    blas_aabb_tests: u32,
    triangle_tests: u32,
};

var<private> bvh_traversal_stats: BVHTraversalStats;

fn bvh_reset_traversal_stats() {
    bvh_traversal_stats = BVHTraversalStats(0u, 0u, 0u);
}
#endif

fn bvh_build_local_ray(ray_world: ptr<function, Ray>, entity_resolved: u32) -> Ray {
    return build_local_ray_from_instance(
        ray_world,
        compact_transforms[entity_resolved]
    );
}

// A direct descent keeps the selected child record resident across the loop backedge.
// This removes one redundant 32-byte node-buffer fetch per descended hierarchy level.
fn bvh_trace_blas_closest(
    ray_local: ptr<function, Ray>,
    mesh_asset_id: u32,
    cull_backfaces: bool
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

    // Keep only deferred far children on the lane-interleaved stack. The near child remains in
    // node_idx, avoiding a stack write followed by a stack read at every internal node.
    var stack_size = 0u;
    var node_idx = root_node_idx;
    var node_bounds_valid = false;
    var current_ray = *ray_local;
    var node = blas_bvh2_nodes[node_idx];

    loop {
        if (node_idx == INVALID_IDX) {
            if (stack_size == 0u) { break; }
            stack_size = stack_size - 1u;
            node_idx = bvh_blas_node_stack[stack_size * BVH_TRAVERSAL_WORKGROUP_SIZE + bvh_stack_lane];
            node_bounds_valid = false;
            node = blas_bvh2_nodes[node_idx];
            continue;
        }

        if (is_leaf(node)) {
            // Direct children were just bounds-tested against the current closest distance.
            // Deferred leaves revalidate because a nearer subtree may have shortened the ray.
            var leaf_is_visible = node_bounds_valid;
            if (!leaf_is_visible) {
#if BVH_TRAVERSAL_COLLECT_STATS
                bvh_traversal_stats.blas_aabb_tests += 1u;
#endif
                let t_leaf = intersect_aabb(&current_ray, node.min.xyz, node.max.xyz);
                leaf_is_visible = t_leaf.x <= t_leaf.y
                    && t_leaf.x < current_ray.direction_and_tmax.w;
            }

            if (leaf_is_visible) {
                let tri_id = u32(node.min.w);
                let tri_base = mesh_directory_entry.first_index + tri_id * 3u;
                let v0i = mesh_directory_entry.first_vertex + index_buffer[tri_base + 0u];
                let v1i = mesh_directory_entry.first_vertex + index_buffer[tri_base + 1u];
                let v2i = mesh_directory_entry.first_vertex + index_buffer[tri_base + 2u];
                let v0 = vertex_position(vertex_buffer[v0i]);
                let v1 = vertex_position(vertex_buffer[v1i]);
                let v2 = vertex_position(vertex_buffer[v2i]);
#if BVH_TRAVERSAL_COLLECT_STATS
                bvh_traversal_stats.triangle_tests += 1u;
#endif
                var triangle_hit: vec4<f32>;
                if (cull_backfaces) {
                    triangle_hit = intersect_triangle_front_face(
                        &current_ray,
                        v0,
                        v1,
                        v2
                    );
                } else {
                    triangle_hit = intersect_triangle(
                        &current_ray,
                        v0,
                        v1,
                        v2
                    );
                }
                let t_tri = triangle_hit.x;
                let is_better_hit = t_tri >= current_ray.origin_and_tmin.w && t_tri < current_ray.direction_and_tmax.w;
                if (is_better_hit) {
                    result.t_hit = t_tri;
                    result.barycentrics = triangle_hit.yz;
                    result.tri_id_local = tri_id;
                    result.tri_indices = vec4<u32>(
                        v0i,
                        v1i,
                        v2i,
                        u32(triangle_hit.w)
                    );
                    result.has_hit = 1u;
                    current_ray.direction_and_tmax.w = t_tri;
                }
            }
        } else {
            let child_idx = u32(node.min.w);
            let child_node = blas_bvh2_nodes[child_idx];
#if BVH_TRAVERSAL_COLLECT_STATS
            bvh_traversal_stats.blas_aabb_tests += 2u;
#endif
            let t_aabb_child = intersect_aabb(
                &current_ray,
                child_node.min.xyz,
                child_node.max.xyz
            );
            let is_better_child = t_aabb_child.x <= t_aabb_child.y
                && t_aabb_child.x < current_ray.direction_and_tmax.w;

            let child_idx_1 = u32(node.max.w);
            let child_node_1 = blas_bvh2_nodes[child_idx_1];
            let t_aabb_child_1 = intersect_aabb(
                &current_ray,
                child_node_1.min.xyz,
                child_node_1.max.xyz
            );
            let is_better_child_1 = t_aabb_child_1.x <= t_aabb_child_1.y
                && t_aabb_child_1.x < current_ray.direction_and_tmax.w;

            if (is_better_child && is_better_child_1) {
                let child_1_is_nearer = t_aabb_child_1.x < t_aabb_child.x;
                node_idx = select(child_idx, child_idx_1, child_1_is_nearer);
                node.min = select(child_node.min, child_node_1.min, child_1_is_nearer);
                node.max = select(child_node.max, child_node_1.max, child_1_is_nearer);
                bvh_blas_node_stack[stack_size * BVH_TRAVERSAL_WORKGROUP_SIZE + bvh_stack_lane] =
                    select(child_idx_1, child_idx, child_1_is_nearer);
                stack_size = stack_size + 1u;
                node_bounds_valid = true;
                continue;
            } else if (is_better_child) {
                node_idx = child_idx;
                node = child_node;
                node_bounds_valid = true;
                continue;
            } else if (is_better_child_1) {
                node_idx = child_idx_1;
                node = child_node_1;
                node_bounds_valid = true;
                continue;
            }
        }

        if (stack_size == 0u) {
            break;
        }
        stack_size = stack_size - 1u;
        node_idx = bvh_blas_node_stack[stack_size * BVH_TRAVERSAL_WORKGROUP_SIZE + bvh_stack_lane];
        node_bounds_valid = false;
        node = blas_bvh2_nodes[node_idx];
    }

    return result;
}

fn bvh_trace_closest(
    ray: ptr<function, Ray>,
    tlas_only: bool,
    cull_backfaces: bool
) -> RayHitCompact {
    var result = make_miss_ray_hit_compact((*ray).direction_and_tmax.w);

    if (tlas_bvh_info.bvh2_count == 0u) {
        return result;
    }

    var node_stack: array<u32, NODE_STACK_SIZE>;
    var stack_size = 0u;
    var node_idx = tlas_bvh_info.bvh2_count - 1u;
    var node_bounds_valid = false;
    var current_ray = *ray;
    var node_tmin = current_ray.origin_and_tmin.w;
    var node = tlas_bvh2_bounds[node_idx];

    loop {
        if (node_idx == INVALID_IDX) {
            if (stack_size == 0u) { break; }
            stack_size = stack_size - 1u;
            node_idx = node_stack[stack_size];
            node_bounds_valid = false;
            node = tlas_bvh2_bounds[node_idx];
            continue;
        }

        if (is_leaf(node)) {
            // Direct children were just bounds-tested; deferred leaves must account for a
            // closest hit found while they waited on the stack.
            var leaf_is_visible = node_bounds_valid;
            var t_leaf_min = node_tmin;
            if (!leaf_is_visible) {
#if BVH_TRAVERSAL_COLLECT_STATS
                bvh_traversal_stats.tlas_aabb_tests += 1u;
#endif
                let t_leaf = intersect_aabb(&current_ray, node.min.xyz, node.max.xyz);
                t_leaf_min = t_leaf.x;
                leaf_is_visible = t_leaf.x <= t_leaf.y
                    && t_leaf.x < result.t_hit;
            }

            if (leaf_is_visible) {
                let mesh_id = u32(node.min.w);
                if (mesh_id != INVALID_IDX) {
                    let prim_store = u32(-node.max.w - 1.0);
                    let entity_resolved = entity_index_lookup[prim_store];

                    if (tlas_only) {
                        result.t_hit = max(t_leaf_min, current_ray.origin_and_tmin.w);
                        result.prim_store = prim_store;
                        result.mesh_id = mesh_id;
                        result.has_hit = 1u;
                        current_ray.direction_and_tmax.w = result.t_hit;
                    } else {
                        var ray_local = bvh_build_local_ray(&current_ray, entity_resolved);
                        let blas_hit = bvh_trace_blas_closest(
                            &ray_local,
                            mesh_id,
                            cull_backfaces
                        );

                        if (blas_hit.has_hit != 0u) {
                            result = blas_hit;
                            result.prim_store = prim_store;
                            result.mesh_id = mesh_id;
                            current_ray.direction_and_tmax.w = min(current_ray.direction_and_tmax.w, result.t_hit);
                        }
                    }
                }
            }
        } else {
            let child_idx = u32(node.min.w);
            let child_node = tlas_bvh2_bounds[child_idx];
#if BVH_TRAVERSAL_COLLECT_STATS
            bvh_traversal_stats.tlas_aabb_tests += 2u;
#endif
            let t_aabb_child = intersect_aabb(
                &current_ray,
                child_node.min.xyz,
                child_node.max.xyz
            );
            let is_better_child = t_aabb_child.x <= t_aabb_child.y
                && t_aabb_child.x < result.t_hit;

            let child_idx_1 = u32(node.max.w);
            let child_node_1 = tlas_bvh2_bounds[child_idx_1];
            let t_aabb_child_1 = intersect_aabb(
                &current_ray,
                child_node_1.min.xyz,
                child_node_1.max.xyz
            );
            let is_better_child_1 = t_aabb_child_1.x <= t_aabb_child_1.y
                && t_aabb_child_1.x < result.t_hit;

            if (is_better_child && is_better_child_1) {
                let child_1_is_nearer = t_aabb_child_1.x < t_aabb_child.x;
                node_idx = select(child_idx, child_idx_1, child_1_is_nearer);
                node.min = select(child_node.min, child_node_1.min, child_1_is_nearer);
                node.max = select(child_node.max, child_node_1.max, child_1_is_nearer);
                node_tmin = select(t_aabb_child.x, t_aabb_child_1.x, child_1_is_nearer);
                node_stack[stack_size] = select(child_idx_1, child_idx, child_1_is_nearer);
                stack_size = stack_size + 1u;
                node_bounds_valid = true;
                continue;
            } else if (is_better_child) {
                node_idx = child_idx;
                node = child_node;
                node_tmin = t_aabb_child.x;
                node_bounds_valid = true;
                continue;
            } else if (is_better_child_1) {
                node_idx = child_idx_1;
                node = child_node_1;
                node_tmin = t_aabb_child_1.x;
                node_bounds_valid = true;
                continue;
            }
        }

        if (stack_size == 0u) {
            break;
        }
        stack_size = stack_size - 1u;
        node_idx = node_stack[stack_size];
        node_bounds_valid = false;
        node = tlas_bvh2_bounds[node_idx];
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

    var stack_size = 0u;
    var node_idx = root_node_idx;
    var node = blas_bvh2_nodes[node_idx];

    loop {
        if (node_idx == INVALID_IDX) {
            if (stack_size == 0u) { break; }
            stack_size = stack_size - 1u;
            node_idx = bvh_blas_node_stack[stack_size * BVH_TRAVERSAL_WORKGROUP_SIZE + bvh_stack_lane];
            node = blas_bvh2_nodes[node_idx];
            continue;
        }

        if (is_leaf(node)) {
            // Any-hit rays never shorten, so a leaf accepted by its parent cannot become stale.
            let tri_id = u32(node.min.w);
            let tri_base = mesh_directory_entry.first_index + tri_id * 3u;
            let v0i = mesh_directory_entry.first_vertex + index_buffer[tri_base + 0u];
            let v1i = mesh_directory_entry.first_vertex + index_buffer[tri_base + 1u];
            let v2i = mesh_directory_entry.first_vertex + index_buffer[tri_base + 2u];
            let v0 = vertex_position(vertex_buffer[v0i]);
            let v1 = vertex_position(vertex_buffer[v1i]);
            let v2 = vertex_position(vertex_buffer[v2i]);
#if BVH_TRAVERSAL_COLLECT_STATS
            bvh_traversal_stats.triangle_tests += 1u;
#endif
            if (intersect_triangle_any(ray_local, v0, v1, v2)) {
                return true;
            }
        } else {
            let child_idx = u32(node.min.w);
            let child_node = blas_bvh2_nodes[child_idx];
#if BVH_TRAVERSAL_COLLECT_STATS
            bvh_traversal_stats.blas_aabb_tests += 2u;
#endif
            let t_aabb_child = intersect_aabb(ray_local, child_node.min.xyz, child_node.max.xyz);
            let is_better_child = t_aabb_child.x <= t_aabb_child.y && t_aabb_child.x < ray_local.direction_and_tmax.w;

            let child_idx_1 = u32(node.max.w);
            let child_node_1 = blas_bvh2_nodes[child_idx_1];
            let t_aabb_child_1 = intersect_aabb(ray_local, child_node_1.min.xyz, child_node_1.max.xyz);
            let is_better_child_1 = t_aabb_child_1.x <= t_aabb_child_1.y && t_aabb_child_1.x < ray_local.direction_and_tmax.w;

            if (is_better_child && is_better_child_1) {
                let child_1_is_nearer = t_aabb_child_1.x < t_aabb_child.x;
                node_idx = select(child_idx, child_idx_1, child_1_is_nearer);
                node.min = select(child_node.min, child_node_1.min, child_1_is_nearer);
                node.max = select(child_node.max, child_node_1.max, child_1_is_nearer);
                bvh_blas_node_stack[stack_size * BVH_TRAVERSAL_WORKGROUP_SIZE + bvh_stack_lane] =
                    select(child_idx_1, child_idx, child_1_is_nearer);
                stack_size = stack_size + 1u;
                continue;
            } else if (is_better_child) {
                node_idx = child_idx;
                node = child_node;
                continue;
            } else if (is_better_child_1) {
                node_idx = child_idx_1;
                node = child_node_1;
                continue;
            }
        }

        if (stack_size == 0u) {
            break;
        }
        stack_size = stack_size - 1u;
        node_idx = bvh_blas_node_stack[stack_size * BVH_TRAVERSAL_WORKGROUP_SIZE + bvh_stack_lane];
        node = blas_bvh2_nodes[node_idx];
    }

    return false;
}

fn bvh_trace_any(ray: ptr<function, Ray>) -> bool {
    if (tlas_bvh_info.bvh2_count == 0u) {
        return false;
    }

    var node_stack: array<u32, NODE_STACK_SIZE>;
    var stack_size = 0u;
    var node_idx = tlas_bvh_info.bvh2_count - 1u;
    var current_ray = *ray;
    var current_node = tlas_bvh2_bounds[node_idx];

    loop {
        if (node_idx == INVALID_IDX) {
            if (stack_size == 0u) { break; }
            stack_size = stack_size - 1u;
            node_idx = node_stack[stack_size];
            current_node = tlas_bvh2_bounds[node_idx];
            continue;
        }

        if (is_leaf(current_node)) {
            // The immutable any-hit interval makes the parent's bounds test sufficient here.
            let mesh_id = u32(current_node.min.w);
            if (mesh_id != INVALID_IDX) {
                let prim_store = u32(-current_node.max.w - 1.0);
                let entity_resolved = entity_index_lookup[prim_store];
                var ray_local = bvh_build_local_ray(&current_ray, entity_resolved);

                if (bvh_trace_blas_any(&ray_local, mesh_id)) {
                    return true;
                }
            }
        } else {
            let child_idx = u32(current_node.min.w);
            let child_node = tlas_bvh2_bounds[child_idx];
#if BVH_TRAVERSAL_COLLECT_STATS
            bvh_traversal_stats.tlas_aabb_tests += 2u;
#endif
            let t_aabb_child = intersect_aabb(&current_ray, child_node.min.xyz, child_node.max.xyz);
            let is_better_child = t_aabb_child.x <= t_aabb_child.y && t_aabb_child.x < current_ray.direction_and_tmax.w;

            let child_idx_1 = u32(current_node.max.w);
            let child_node_1 = tlas_bvh2_bounds[child_idx_1];
            let t_aabb_child_1 = intersect_aabb(&current_ray, child_node_1.min.xyz, child_node_1.max.xyz);
            let is_better_child_1 = t_aabb_child_1.x <= t_aabb_child_1.y && t_aabb_child_1.x < current_ray.direction_and_tmax.w;

            if (is_better_child && is_better_child_1) {
                let child_1_is_nearer = t_aabb_child_1.x < t_aabb_child.x;
                node_idx = select(child_idx, child_idx_1, child_1_is_nearer);
                current_node.min = select(child_node.min, child_node_1.min, child_1_is_nearer);
                current_node.max = select(child_node.max, child_node_1.max, child_1_is_nearer);
                node_stack[stack_size] = select(child_idx_1, child_idx, child_1_is_nearer);
                stack_size = stack_size + 1u;
                continue;
            } else if (is_better_child) {
                node_idx = child_idx;
                current_node = child_node;
                continue;
            } else if (is_better_child_1) {
                node_idx = child_idx_1;
                current_node = child_node_1;
                continue;
            }
        }

        if (stack_size == 0u) {
            break;
        }
        stack_size = stack_size - 1u;
        node_idx = node_stack[stack_size];
        current_node = tlas_bvh2_bounds[node_idx];
    }

    return false;
}

fn trace_ray_closest(ray: ptr<function, Ray>) -> RayHitCompact {
    return bvh_trace_closest(ray, false, false);
}

fn trace_ray_closest_front_faces(ray: ptr<function, Ray>) -> RayHitCompact {
    return bvh_trace_closest(ray, false, true);
}

fn trace_ray_closest_tlas(ray: ptr<function, Ray>, tlas_only: bool) -> RayHitCompact {
    return bvh_trace_closest(ray, tlas_only, false);
}

fn trace_ray_any(ray: ptr<function, Ray>) -> bool {
    return bvh_trace_any(ray);
}
