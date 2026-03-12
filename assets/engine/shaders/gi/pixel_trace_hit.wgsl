// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║               PER-PIXEL PATH TRACING - BVH TRAVERSAL                      ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Traces rays through the BVH acceleration structure:                      ║
// ║  • TLAS (Top-Level) traversal for instance culling                        ║
// ║  • BLAS (Bottom-Level) traversal for triangle intersection                ║
// ║  • Shadow ray any-hit queries for NEE                                     ║
// ║  • Outputs hit attributes for shading pass                                ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================
#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "gi/gi_common.wgsl"

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read_write> gi_counters: GICounters;
@group(1) @binding(2) var<storage, read_write> pixel_path_state: array<PixelPathState>;
@group(1) @binding(3) var<storage, read_write> ray_work_queue: array<u32>;
@group(1) @binding(4) var<storage, read> tlas_bvh2_bounds: array<AABB>;
@group(1) @binding(5) var<uniform> tlas_bvh_info: BVHInfo;
@group(1) @binding(6) var<storage, read> blas_bvh2_nodes: array<AABB>;
@group(1) @binding(7) var<storage, read> blas_directory: array<MeshDirectoryEntry>;
@group(1) @binding(8) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(9) var<storage, read> index_buffer: array<u32>;

// =============================================================================
// BLAS TRAVERSAL
// =============================================================================

fn trace_blas(
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

    var pending_child_idx = 0u;
    var pending_child_tmin = 0.0;
    var have_pending_child = false;
    var node_idx = INVALID_IDX;
    var child_idx = INVALID_IDX;
    var is_better_hit = false;
    var is_better_child = false;
    var current_is_farther = false;
    var push_idx = INVALID_IDX;
    var keep_idx = INVALID_IDX;
    var keep_tmin = 0.0;
    var t_tri = 0.0;
    var t_aabb_child = vec2<f32>(0.0, 0.0);
    var node: AABB;
    var leaf_indices = vec4<u32>(0u, 0u, 0u, 0u);
    var v0 = vec3<f32>(0.0, 0.0, 0.0);
    var v1 = vec3<f32>(0.0, 0.0, 0.0);
    var v2 = vec3<f32>(0.0, 0.0, 0.0);
    var tri_id = 0u;
    var tri_base = 0u;
    var v0i = 0u;
    var v1i = 0u;
    var v2i = 0u;

    while (stack_size > 0u) {
        stack_size = stack_size - 1u;

        node_idx = node_stack[stack_size];
        if (node_idx == INVALID_IDX) { continue; }

        node = blas_bvh2_nodes[node_idx];
        if (is_leaf(node)) {
            tri_id = u32(node.min.w);
            tri_base = mesh_directory_entry.first_index + tri_id * 3u;
            v0i = mesh_directory_entry.first_vertex + index_buffer[tri_base + 0u];
            v1i = mesh_directory_entry.first_vertex + index_buffer[tri_base + 1u];
            v2i = mesh_directory_entry.first_vertex + index_buffer[tri_base + 2u];
            v0 = vertex_buffer[v0i].position.xyz;
            v1 = vertex_buffer[v1i].position.xyz;
            v2 = vertex_buffer[v2i].position.xyz;
            t_tri = intersect_triangle(&current_ray, v0, v1, v2);
            is_better_hit = t_tri >= current_ray.origin_and_tmin.w && t_tri < current_ray.direction_and_tmax.w;
            if (is_better_hit) {
                leaf_indices = vec4<u32>(v0i, v1i, v2i, 0u);
                result.t_hit = t_tri;
                result.tri_id_local = tri_id;
                result.tri_indices = leaf_indices;
                result.has_hit = 1u;
                current_ray.direction_and_tmax.w = t_tri;
            }
        } else {
            pending_child_idx = 0u;
            pending_child_tmin = 0.0;
            have_pending_child = false;

            child_idx = u32(node.min.w);
            t_aabb_child = intersect_aabb(
                &current_ray,
                blas_bvh2_nodes[child_idx].min.xyz,
                blas_bvh2_nodes[child_idx].max.xyz
            );
            is_better_child = t_aabb_child.x <= t_aabb_child.y
                && t_aabb_child.x >= current_ray.origin_and_tmin.w
                && t_aabb_child.x < current_ray.direction_and_tmax.w;
            if (is_better_child) {
                #if BVH_TRAVERSAL_ORDER_CHILDREN
                if (!have_pending_child) {
                    pending_child_idx = child_idx;
                    pending_child_tmin = t_aabb_child.x;
                    have_pending_child = true;
                } else {
                    current_is_farther = t_aabb_child.x > pending_child_tmin;
                    push_idx = select(pending_child_idx, child_idx, current_is_farther);
                    keep_idx = select(child_idx, pending_child_idx, current_is_farther);
                    keep_tmin = select(t_aabb_child.x, pending_child_tmin, current_is_farther);

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

            child_idx = u32(node.max.w);
            t_aabb_child = intersect_aabb(
                &current_ray,
                blas_bvh2_nodes[child_idx].min.xyz,
                blas_bvh2_nodes[child_idx].max.xyz
            );
            is_better_child = t_aabb_child.x <= t_aabb_child.y
                && t_aabb_child.x >= current_ray.origin_and_tmin.w
                && t_aabb_child.x < current_ray.direction_and_tmax.w;
            if (is_better_child) {
                #if BVH_TRAVERSAL_ORDER_CHILDREN
                if (!have_pending_child) {
                    pending_child_idx = child_idx;
                    pending_child_tmin = t_aabb_child.x;
                    have_pending_child = true;
                } else {
                    current_is_farther = t_aabb_child.x > pending_child_tmin;
                    push_idx = select(pending_child_idx, child_idx, current_is_farther);
                    keep_idx = select(child_idx, pending_child_idx, current_is_farther);
                    keep_tmin = select(t_aabb_child.x, pending_child_tmin, current_is_farther);

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

// =============================================================================
// TLAS TRAVERSAL (CLOSEST HIT)
// =============================================================================

fn trace_hit(ray: ptr<function, Ray>) -> RayHitCompact {
    var result = make_miss_ray_hit_compact((*ray).direction_and_tmax.w);

    if (tlas_bvh_info.bvh2_count == 0u) {
        return result;
    }

    var node_stack: array<u32, NODE_STACK_SIZE>;
    node_stack[0] = tlas_bvh_info.bvh2_count - 1u;
    var stack_size = 1u;

    var current_ray = *ray;

    var pending_child_idx = 0u;
    var pending_child_tmin = 0.0;
    var have_pending_child = false;
    var node_idx = INVALID_IDX;
    var child_idx = INVALID_IDX;
    var is_better_hit = false;
    var is_better_child = false;
    var current_is_farther = false;
    var push_idx = INVALID_IDX;
    var keep_idx = INVALID_IDX;
    var keep_tmin = 0.0;
    var t_leaf = vec2<f32>(0.0, 0.0);
    var t_aabb_child = vec2<f32>(0.0, 0.0);
    var prim_store = 0u;
    var mesh_id = 0u;
    var node: AABB;

    while (stack_size > 0u) {
        stack_size = stack_size - 1u;

        node_idx = node_stack[stack_size];
        if (node_idx == INVALID_IDX) { continue; }

        node = tlas_bvh2_bounds[node_idx];
        if (is_leaf(node)) {
            t_leaf = intersect_aabb(&current_ray, node.min.xyz, node.max.xyz);
            if (t_leaf.x <= t_leaf.y && max(t_leaf.x, current_ray.origin_and_tmin.w) < result.t_hit) {
                mesh_id = u32(node.min.w);
                prim_store = u32(-node.max.w - 1.0);
                if (mesh_id == INVALID_IDX) { continue; }

                var ray_local = build_local_ray(
                    &current_ray,
                    entity_transforms[prim_store].transform,
                    entity_transforms[prim_store].transpose_inverse_model_matrix
                );
                let blas_hit = trace_blas(&ray_local, mesh_id);

                if (blas_hit.has_hit != 0u) {
                    result = blas_hit;
                    result.prim_store = prim_store;
                    result.mesh_id = mesh_id;
                    current_ray.direction_and_tmax.w = min(current_ray.direction_and_tmax.w, result.t_hit);
                }
            }
        } else {
            pending_child_idx = 0u;
            pending_child_tmin = 0.0;
            have_pending_child = false;

            child_idx = u32(node.min.w);
            t_aabb_child = intersect_aabb(
                &current_ray,
                tlas_bvh2_bounds[child_idx].min.xyz,
                tlas_bvh2_bounds[child_idx].max.xyz
            );
            is_better_child = t_aabb_child.x <= t_aabb_child.y
                && t_aabb_child.x >= current_ray.origin_and_tmin.w
                && t_aabb_child.x < result.t_hit;
            if (is_better_child) {
                #if BVH_TRAVERSAL_ORDER_CHILDREN
                if (!have_pending_child) {
                    pending_child_idx = child_idx;
                    pending_child_tmin = t_aabb_child.x;
                    have_pending_child = true;
                } else {
                    current_is_farther = t_aabb_child.x > pending_child_tmin;
                    push_idx = select(pending_child_idx, child_idx, current_is_farther);
                    keep_idx = select(child_idx, pending_child_idx, current_is_farther);
                    keep_tmin = select(t_aabb_child.x, pending_child_tmin, current_is_farther);

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

            child_idx = u32(node.max.w);
            t_aabb_child = intersect_aabb(
                &current_ray,
                tlas_bvh2_bounds[child_idx].min.xyz,
                tlas_bvh2_bounds[child_idx].max.xyz
            );
            is_better_child = t_aabb_child.x <= t_aabb_child.y
                && t_aabb_child.x >= current_ray.origin_and_tmin.w
                && t_aabb_child.x < result.t_hit;
            if (is_better_child) {
                #if BVH_TRAVERSAL_ORDER_CHILDREN
                if (!have_pending_child) {
                    pending_child_idx = child_idx;
                    pending_child_tmin = t_aabb_child.x;
                    have_pending_child = true;
                } else {
                    current_is_farther = t_aabb_child.x > pending_child_tmin;
                    push_idx = select(pending_child_idx, child_idx, current_is_farther);
                    keep_idx = select(child_idx, pending_child_idx, current_is_farther);
                    keep_tmin = select(t_aabb_child.x, pending_child_tmin, current_is_farther);

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

// =============================================================================
// BLAS ANY-HIT (SHADOW RAYS)
// =============================================================================

fn trace_blas_any_hit(
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

    var pending_child_idx = 0u;
    var pending_child_tmin = 0.0;
    var have_pending_child = false;
    var node_idx = INVALID_IDX;
    var child_idx = INVALID_IDX;
    var is_better_child = false;
    var current_is_farther = false;
    var push_idx = INVALID_IDX;
    var keep_idx = INVALID_IDX;
    var keep_tmin = 0.0;
    var t_tri = 0.0;
    var t_aabb_child = vec2<f32>(0.0, 0.0);
    var v0 = vec3<f32>(0.0, 0.0, 0.0);
    var v1 = vec3<f32>(0.0, 0.0, 0.0);
    var v2 = vec3<f32>(0.0, 0.0, 0.0);
    var node: AABB;

    loop {
        if (stack_size == 0u) { break; }
        stack_size = stack_size - 1u;

        node_idx = node_stack[stack_size];
        if (node_idx == INVALID_IDX) { continue; }

        node = blas_bvh2_nodes[node_idx];
        if (is_leaf(node)) {
            let tri_id = u32(node.min.w);
            let tri_base = mesh_directory_entry.first_index + tri_id * 3u;
            let v0i = mesh_directory_entry.first_vertex + index_buffer[tri_base + 0u];
            let v1i = mesh_directory_entry.first_vertex + index_buffer[tri_base + 1u];
            let v2i = mesh_directory_entry.first_vertex + index_buffer[tri_base + 2u];
            v0 = vertex_buffer[v0i].position.xyz;
            v1 = vertex_buffer[v1i].position.xyz;
            v2 = vertex_buffer[v2i].position.xyz;
            t_tri = intersect_triangle(ray_local, v0, v1, v2);
            if (t_tri >= ray_local.origin_and_tmin.w && t_tri < ray_local.direction_and_tmax.w) {
                return true;
            }
        } else {
            pending_child_idx = 0u;
            pending_child_tmin = 0.0;
            have_pending_child = false;

            child_idx = u32(node.min.w);
            t_aabb_child = intersect_aabb(ray_local, blas_bvh2_nodes[child_idx].min.xyz, blas_bvh2_nodes[child_idx].max.xyz);
            is_better_child = t_aabb_child.x <= t_aabb_child.y && t_aabb_child.x >= ray_local.origin_and_tmin.w && t_aabb_child.x < ray_local.direction_and_tmax.w;
            if (is_better_child) {
                #if BVH_TRAVERSAL_ORDER_CHILDREN
                if (!have_pending_child) {
                    pending_child_idx = child_idx;
                    pending_child_tmin = t_aabb_child.x;
                    have_pending_child = true;
                } else {
                    current_is_farther = t_aabb_child.x > pending_child_tmin;
                    push_idx = select(pending_child_idx, child_idx, current_is_farther);
                    keep_idx = select(child_idx, pending_child_idx, current_is_farther);
                    keep_tmin = select(t_aabb_child.x, pending_child_tmin, current_is_farther);

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

            child_idx = u32(node.max.w);
            t_aabb_child = intersect_aabb(ray_local, blas_bvh2_nodes[child_idx].min.xyz, blas_bvh2_nodes[child_idx].max.xyz);
            is_better_child = t_aabb_child.x <= t_aabb_child.y && t_aabb_child.x >= ray_local.origin_and_tmin.w && t_aabb_child.x < ray_local.direction_and_tmax.w;
            if (is_better_child) {
                #if BVH_TRAVERSAL_ORDER_CHILDREN
                if (!have_pending_child) {
                    pending_child_idx = child_idx;
                    pending_child_tmin = t_aabb_child.x;
                    have_pending_child = true;
                } else {
                    current_is_farther = t_aabb_child.x > pending_child_tmin;
                    push_idx = select(pending_child_idx, child_idx, current_is_farther);
                    keep_idx = select(child_idx, pending_child_idx, current_is_farther);
                    keep_tmin = select(t_aabb_child.x, pending_child_tmin, current_is_farther);

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

// =============================================================================
// TLAS ANY-HIT (SHADOW RAYS)
// =============================================================================

fn trace_hit_any(ray: ptr<function, Ray>) -> bool {
    if (tlas_bvh_info.bvh2_count == 0u) {
        return false;
    }

    var node_stack: array<u32, NODE_STACK_SIZE>;
    node_stack[0] = tlas_bvh_info.bvh2_count - 1u;
    var stack_size = 1u;

    var current_ray = *ray;

    var pending_child_idx = 0u;
    var pending_child_tmin = 0.0;
    var have_pending_child = false;
    var node_idx = INVALID_IDX;
    var child_idx = INVALID_IDX;
    var is_better_child = false;
    var current_is_farther = false;
    var push_idx = INVALID_IDX;
    var keep_idx = INVALID_IDX;
    var keep_tmin = 0.0;
    var t_leaf = vec2<f32>(0.0, 0.0);
    var t_aabb_child = vec2<f32>(0.0, 0.0);
    var leaf_bounds: AABB;
    var prim_store = 0u;
    var mesh_id = 0u;
    var entity_transform: EntityTransform;
    var ray_local: Ray;
    var current_node: AABB;

    loop {
        if (stack_size == 0u) { break; }
        stack_size = stack_size - 1u;

        node_idx = node_stack[stack_size];
        if (node_idx == INVALID_IDX) { continue; }

        current_node = tlas_bvh2_bounds[node_idx];
        if (is_leaf(current_node)) {
            leaf_bounds = current_node;
            t_leaf = intersect_aabb(&current_ray, leaf_bounds.min.xyz, leaf_bounds.max.xyz);

            if (t_leaf.x <= t_leaf.y && max(t_leaf.x, current_ray.origin_and_tmin.w) < current_ray.direction_and_tmax.w) {
                mesh_id = u32(leaf_bounds.min.w);
                prim_store = u32(-leaf_bounds.max.w - 1.0);
                if (mesh_id == INVALID_IDX) { continue; }
                entity_transform = entity_transforms[prim_store];

                ray_local = build_local_ray(
                    &current_ray,
                    entity_transform.transform,
                    entity_transform.transpose_inverse_model_matrix
                );

                if (trace_blas_any_hit(&ray_local, mesh_id)) {
                    return true;
                }
            }
        } else {
            pending_child_idx = 0u;
            pending_child_tmin = 0.0;
            have_pending_child = false;

            child_idx = u32(current_node.min.w);
            t_aabb_child = intersect_aabb(&current_ray, tlas_bvh2_bounds[child_idx].min.xyz, tlas_bvh2_bounds[child_idx].max.xyz);
            is_better_child = t_aabb_child.x <= t_aabb_child.y && max(t_aabb_child.x, current_ray.origin_and_tmin.w) < current_ray.direction_and_tmax.w;
            if (is_better_child) {
                #if BVH_TRAVERSAL_ORDER_CHILDREN
                if (!have_pending_child) {
                    pending_child_idx = child_idx;
                    pending_child_tmin = t_aabb_child.x;
                    have_pending_child = true;
                } else {
                    current_is_farther = t_aabb_child.x > pending_child_tmin;
                    push_idx = select(pending_child_idx, child_idx, current_is_farther);
                    keep_idx = select(child_idx, pending_child_idx, current_is_farther);
                    keep_tmin = select(t_aabb_child.x, pending_child_tmin, current_is_farther);

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

            child_idx = u32(current_node.max.w);
            t_aabb_child = intersect_aabb(&current_ray, tlas_bvh2_bounds[child_idx].min.xyz, tlas_bvh2_bounds[child_idx].max.xyz);
            is_better_child = t_aabb_child.x <= t_aabb_child.y && max(t_aabb_child.x, current_ray.origin_and_tmin.w) < current_ray.direction_and_tmax.w;
            if (is_better_child) {
                #if BVH_TRAVERSAL_ORDER_CHILDREN
                if (!have_pending_child) {
                    pending_child_idx = child_idx;
                    pending_child_tmin = t_aabb_child.x;
                    have_pending_child = true;
                } else {
                    current_is_farther = t_aabb_child.x > pending_child_tmin;
                    push_idx = select(pending_child_idx, child_idx, current_is_farther);
                    keep_idx = select(child_idx, pending_child_idx, current_is_farther);
                    keep_tmin = select(t_aabb_child.x, pending_child_tmin, current_is_farther);

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

// =============================================================================
// HELPER: Process a shadow ray and write result
// =============================================================================

fn process_shadow_ray(ray_index: u32) {
    var ray: Ray;
    ray.origin_and_tmin = pixel_path_state[ray_index].shadow_origin;
    ray.direction_and_tmax = pixel_path_state[ray_index].shadow_direction;
    ray.inv_direction = vec4<f32>(
        1.0 / max(abs(ray.direction_and_tmax.x), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.x < 0.0),
        1.0 / max(abs(ray.direction_and_tmax.y), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.y < 0.0),
        1.0 / max(abs(ray.direction_and_tmax.z), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.z < 0.0),
        0.0
    );

    if (!trace_hit_any(&ray)) {
        // No shadow hit - light is visible
        pixel_path_state[ray_index].state_u32.z = 1u;
    }
}

// =============================================================================
// HELPER: Process a primary ray and write hit attributes
// =============================================================================

fn process_primary_ray(ray_index: u32) {
    var ray: Ray;
    ray.origin_and_tmin = pixel_path_state[ray_index].origin_tmin;
    ray.direction_and_tmax = pixel_path_state[ray_index].direction_tmax;
    ray.inv_direction = vec4<f32>(
        1.0 / max(abs(ray.direction_and_tmax.x), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.x < 0.0),
        1.0 / max(abs(ray.direction_and_tmax.y), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.y < 0.0),
        1.0 / max(abs(ray.direction_and_tmax.z), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.z < 0.0),
        0.0
    );

    let hit_result = trace_hit(&ray);

    if (hit_result.has_hit != 0u) {
        let tri_id_local = hit_result.tri_id_local;
        let prim_store = hit_result.prim_store;

        let entity_transform = entity_transforms[prim_store];

        var ray_local = build_local_ray(
            &ray,
            entity_transform.transform,
            entity_transform.transpose_inverse_model_matrix
        );

        let t_tri = hit_result.t_hit;
        let p_local = ray_local.origin_and_tmin.xyz + ray_local.direction_and_tmax.xyz * t_tri;
        let p_world = (entity_transform.transform * vec4<f32>(p_local, 1.0)).xyz;

        let v0i = hit_result.tri_indices.x;
        let v1i = hit_result.tri_indices.y;
        let v2i = hit_result.tri_indices.z;

        // Load vertex positions for barycentric calculation
        let v0 = vertex_buffer[v0i].position.xyz;
        let v1 = vertex_buffer[v1i].position.xyz;
        let v2 = vertex_buffer[v2i].position.xyz;

        // Compute barycentric coordinates
        let e0 = v1 - v0;
        let e1 = v2 - v0;
        let vp = p_local - v0;
        let d00 = dot(e0, e0);
        let d01 = dot(e0, e1);
        let d11 = dot(e1, e1);
        let d20 = dot(vp, e0);
        let d21 = dot(vp, e1);
        let denom = max(d00 * d11 - d01 * d01, 1e-8);
        let v_bc = (d00 * d21 - d01 * d20) / denom;
        let u_bc = (d11 * d20 - d01 * d21) / denom;
        let w_bc = 1.0 - u_bc - v_bc;

        // Interpolate UVs
        let uv_hit = vertex_buffer[v0i].uv.xy * w_bc +
                     vertex_buffer[v1i].uv.xy * u_bc +
                     vertex_buffer[v2i].uv.xy * v_bc;

        // Interpolate and transform normals
        let n_local = vertex_buffer[v0i].normal.xyz * w_bc +
                      vertex_buffer[v1i].normal.xyz * u_bc +
                      vertex_buffer[v2i].normal.xyz * v_bc;
        var world_n = safe_normalize((entity_transform.transform * vec4<f32>(n_local, 0.0)).xyz);

        // Interpolate and transform tangents
        let t_local = vertex_buffer[v0i].tangent.xyz * w_bc +
                      vertex_buffer[v1i].tangent.xyz * u_bc +
                      vertex_buffer[v2i].tangent.xyz * v_bc;
        var world_t = safe_normalize((entity_transform.transform * vec4<f32>(t_local, 0.0)).xyz);

        // Interpolate and transform bitangents
        let b_local = vertex_buffer[v0i].bitangent.xyz * w_bc +
                      vertex_buffer[v1i].bitangent.xyz * u_bc +
                      vertex_buffer[v2i].bitangent.xyz * v_bc;
        var world_b = safe_normalize((entity_transform.transform * vec4<f32>(b_local, 0.0)).xyz);

        // Handle backfacing geometry
        let ray_dir = ray.direction_and_tmax.xyz;
        let ray_is_backfacing = dot(world_n, ray_dir) > 0.0;
        world_n = select(world_n, -world_n, ray_is_backfacing);
        world_t = select(world_t, -world_t, ray_is_backfacing);
        world_b = select(world_b, -world_b, ray_is_backfacing);

        // Store hit information
        pixel_path_state[ray_index].origin_tmin = vec4<f32>(p_world, t_tri);
        pixel_path_state[ray_index].direction_tmax = vec4<f32>(ray_dir, f32(prim_store));
        pixel_path_state[ray_index].normal_section_index = vec4<f32>(world_n, f32(vertex_buffer[v0i].section_index));
        pixel_path_state[ray_index].hit_attr0 = vec4<f32>(world_t, uv_hit.x);
        pixel_path_state[ray_index].hit_attr1 = vec4<f32>(world_b, uv_hit.y);
        pixel_path_state[ray_index].state_u32.w = tri_id_local;
    }
}

// =============================================================================
// MAIN COMPUTE SHADER
// =============================================================================
//
// Dispatched over ALL pixels (2x for shadow + primary ray processing).
// Uses work queue for efficient processing of only active rays.
//
//   - First half of invocations: Shadow ray traces (NEE)
//   - Second half: Primary ray traces (indirect bounce)
//
// Each thread independently consumes from a work queue, processing rays until
// the queue is exhausted.
// =============================================================================

@compute @workgroup_size(128, 1, 1)
fn cs(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(subgroup_invocation_id) lane_id: u32,
    @builtin(subgroup_size) warp_size: u32
) {
    // ─────────────────────────────────────────────────────────────────────────
    // Compute ray counts and determine thread role
    // ─────────────────────────────────────────────────────────────────────────
    let rays_per_pixel = u32(gi_params.screen_ray_count);
    let gi_resolution = vec2<u32>(u32(gi_params.gi_resolution_x), u32(gi_params.gi_resolution_y));
    let total_pixels = gi_resolution.x * gi_resolution.y;
    let total_rays = total_pixels * rays_per_pixel;

#if USE_RADIANCE_CACHE_AS_DEFERRED_LIGHTING
    // Determine if this thread handles shadow rays or primary rays
    // First half = shadow threads, Second half = primary threads
    let is_shadow_thread = gid.x < total_rays;
    let thread_id = select(gid.x - total_rays, gid.x, is_shadow_thread);
#else
    // In this case, we only have primary rays
    let is_shadow_thread = false;
    let thread_id = gid.x;
#endif

    // Early exit if thread is outside valid range
    if (thread_id >= total_rays) {
        return;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Read queue count once at start (all active rays added by init pass)
    // ─────────────────────────────────────────────────────────────────────────
    let queue_count = atomicLoad(&gi_counters.ray_queue_count);

    // ─────────────────────────────────────────────────────────────────────────
    // Work queue consumption loop
    // ─────────────────────────────────────────────────────────────────────────
    loop {
        // Each thread type uses its own atomic counter
        var queue_index = 0u;
        if (is_shadow_thread) {
            queue_index = atomicAdd(&gi_counters.ray_queue_shadow_head, 1u);
        } else {
            queue_index = atomicAdd(&gi_counters.ray_queue_primary_head, 1u);
        }

        if (queue_index >= queue_count) {
            break;
        }

        // Work queue stores ray slots (each slot corresponds to one PixelPathState entry).
        let ray_slot = ray_work_queue[queue_index];

        if (is_shadow_thread) {
            process_shadow_ray(ray_slot);
        } else {
            process_primary_ray(ray_slot);
        }
    }
}
