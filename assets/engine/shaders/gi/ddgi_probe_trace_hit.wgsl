// =============================================================================
// DDGI Probe Ray Trace - Hit Pass
// - Traces primary rays from world-space probes against the BVH
// - Writes compact hit attributes for the shade pass
// - Intentionally avoids ALL material + texture bindings to reduce binding count
// =============================================================================
#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "gi/ddgi_common.wgsl"

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read_write> probe_ray_data: DDGIProbeRayDataBufferReadOnlyHeader;
@group(1) @binding(2) var<storage, read> tlas_bvh2_bounds: array<AABB>;
@group(1) @binding(3) var<uniform> tlas_bvh_info: BVHInfo;
@group(1) @binding(4) var<storage, read> blas_bvh2_nodes: array<AABB>;
@group(1) @binding(5) var<storage, read> blas_directory: array<MeshDirectoryEntry>;
@group(1) @binding(6) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(7) var<storage, read> index_buffer: array<u32>;
@group(1) @binding(8) var<storage, read> dense_lights_buffer: DenseLightsBuffer;
@group(1) @binding(9) var<storage, read> emissive_lights_buffer: EmissiveLightsBuffer;

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

    while (stack_size > 0u) {
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
fn process_shadow_visibility(index: u32, ray_origin: vec3<f32>, ray_dir: vec3<f32>, t_max: f32) {
    var ray: Ray;
    ray.origin_and_tmin = vec4<f32>(ray_origin + ray_dir * 0.001, 0.0);
    ray.direction_and_tmax = vec4<f32>(ray_dir, t_max);
    ray.inv_direction = vec4<f32>(
        1.0 / max(abs(ray.direction_and_tmax.x), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.x < 0.0),
        1.0 / max(abs(ray.direction_and_tmax.y), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.y < 0.0),
        1.0 / max(abs(ray.direction_and_tmax.z), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.z < 0.0),
        0.0
    );

    if (!trace_hit_any(&ray)) {
        // No shadow hit - light is visible
        probe_ray_data.rays[index].state_u32.z = 1u;
    }
}

fn sample_weighted_emissive_light(
    rng: ptr<function, u32>,
    num_emissive_lights: u32,
    emissive_pdf: ptr<function, f32>
) -> u32 {
    let safe_emissive_count = max(num_emissive_lights, 1u);
    let uniform_pdf = 1.0 / f32(safe_emissive_count);
    (*emissive_pdf) = uniform_pdf;

    (*rng) = random_seed((*rng));
    let uniform_rand = rand_float((*rng));
    var selected_emissive_idx = u32(uniform_rand * f32(safe_emissive_count)) % safe_emissive_count;

    if (num_emissive_lights == 0u) {
        return selected_emissive_idx;
    }

    let total_sampling_weight =
        f32(emissive_lights_buffer.header._pad0) * EMISSIVE_WEIGHT_QUANTIZATION_INV;
    let max_sampling_weight =
        f32(emissive_lights_buffer.header._pad1) * EMISSIVE_WEIGHT_QUANTIZATION_INV;
    let can_use_weighted_sampling =
        total_sampling_weight > 0.0 && max_sampling_weight > 0.0;

    var accepted = false;
    if (can_use_weighted_sampling) {
        for (var attempt_idx = 0u; attempt_idx < EMISSIVE_WEIGHTED_SAMPLE_ATTEMPTS; attempt_idx = attempt_idx + 1u) {
            (*rng) = random_seed((*rng));
            let candidate_rand = rand_float((*rng));
            let candidate_idx = u32(candidate_rand * f32(num_emissive_lights)) % num_emissive_lights;
            let candidate_weight = max(emissive_lights_buffer.lights[candidate_idx].radiance_weight.w, 0.0);
            let accept_prob = min(candidate_weight / max_sampling_weight, 1.0);

            (*rng) = random_seed((*rng));
            let accept_rand = rand_float((*rng));
            if (accept_rand <= accept_prob) {
                selected_emissive_idx = candidate_idx;
                accepted = true;
                break;
            }
        }
    }

    if (accepted) {
        let selected_weight = max(emissive_lights_buffer.lights[selected_emissive_idx].radiance_weight.w, 0.0);
        (*emissive_pdf) = selected_weight / max(total_sampling_weight, 1e-6);
    }

    return selected_emissive_idx;
}

// =============================================================================
// HELPER: Process a primary ray and write hit attributes
// =============================================================================
fn process_primary_ray(
    index: u32,
    probe_position: vec3<f32>,
    ray_dir: vec3<f32>,
    probe_index: u32,
    ray_index_in_probe: u32,
) {
    var ray: Ray;
    ray.origin_and_tmin = vec4<f32>(probe_position + ray_dir * 0.001, 0.0);
    ray.direction_and_tmax = vec4<f32>(ray_dir, 1e30);
    ray.inv_direction = vec4<f32>(
        1.0 / max(abs(ray_dir.x), 1e-8) * select(1.0, -1.0, ray_dir.x < 0.0),
        1.0 / max(abs(ray_dir.y), 1e-8) * select(1.0, -1.0, ray_dir.y < 0.0),
        1.0 / max(abs(ray_dir.z), 1e-8) * select(1.0, -1.0, ray_dir.z < 0.0),
        0.0
    );

    // Always write per-ray direction so the shade pass can handle ray misses.
    // Preserve ray_dir_prim.w which stores the per-ray PDF written by the init pass.
    probe_ray_data.rays[index].meta_u32.x = probe_index;
    let ray_pdf = probe_ray_data.rays[index].ray_dir_prim.w;
    probe_ray_data.rays[index].ray_dir_prim = vec4<f32>(ray_dir, ray_pdf);
    probe_ray_data.rays[index].nee_light_dir_type = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    probe_ray_data.rays[index].nee_light_radiance = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    probe_ray_data.rays[index].hit_pos_t = vec4<f32>(0.0, 0.0, 0.0, 0.0);

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

        let v0 = vertex_buffer[v0i].position.xyz;
        let v1 = vertex_buffer[v1i].position.xyz;
        let v2 = vertex_buffer[v2i].position.xyz;

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

        let uv_hit = vertex_buffer[v0i].uv.xy * w_bc +
            vertex_buffer[v1i].uv.xy * u_bc +
            vertex_buffer[v2i].uv.xy * v_bc;

        let n_local = vertex_buffer[v0i].normal.xyz * w_bc +
            vertex_buffer[v1i].normal.xyz * u_bc +
            vertex_buffer[v2i].normal.xyz * v_bc;
        var world_n = safe_normalize((entity_transform.transform * vec4<f32>(n_local, 0.0)).xyz);

        let t_local = vertex_buffer[v0i].tangent.xyz * w_bc +
            vertex_buffer[v1i].tangent.xyz * u_bc +
            vertex_buffer[v2i].tangent.xyz * v_bc;
        var world_t = safe_normalize((entity_transform.transform * vec4<f32>(t_local, 0.0)).xyz);

        let b_local = vertex_buffer[v0i].bitangent.xyz * w_bc +
            vertex_buffer[v1i].bitangent.xyz * u_bc +
            vertex_buffer[v2i].bitangent.xyz * v_bc;
        var world_b = safe_normalize((entity_transform.transform * vec4<f32>(b_local, 0.0)).xyz);

        let ray_is_backfacing = dot(world_n, ray_dir) > 0.0;
        world_n = select(world_n, -world_n, ray_is_backfacing);
        world_t = select(world_t, -world_t, ray_is_backfacing);
        world_b = select(world_b, -world_b, ray_is_backfacing);

        let section_index = u32(vertex_buffer[v0i].section_index);

        // Backface rays:
        // - Mark with NEGATIVE distance so the shade pass can zero irradiance (leak reduction).
        // - Shorten their stored depth by 80% (multiply by 0.2) for conservative visibility.
        //   (World hit position stays unmodified; we only adjust the stored "t".)
        // - Using negative distance allows efficient backface counting without extra flags,
        //   enabling robust dead probe detection even with non-manifold geometry.
        let stored_t = select(t_tri, -t_tri, ray_is_backfacing);
        probe_ray_data.rays[index].hit_pos_t = vec4<f32>(p_world, stored_t);
        probe_ray_data.rays[index].ray_dir_prim = vec4<f32>(ray_dir, ray_pdf);
        probe_ray_data.rays[index].world_n_section = vec4<f32>(world_n, f32(section_index));
        probe_ray_data.rays[index].world_t_uvx = vec4<f32>(world_t, uv_hit.x);
        probe_ray_data.rays[index].world_b_uvy = vec4<f32>(world_b, uv_hit.y);
        probe_ray_data.rays[index].state_u32.w = tri_id_local;
        probe_ray_data.rays[index].state_u32.x = prim_store;

        // One-sample NEE visibility test at the primary hit point.
        // We do the expensive shadow trace here (hit pass has BVH bindings),
        // and the shade pass replays the same light selection to compute the
        // direct lighting contribution.
        let num_lights = dense_lights_buffer.header.light_count;
        let num_emissive_lights = emissive_lights_buffer.header.light_count;
        let total_light_count = num_lights + num_emissive_lights;
        if (total_light_count > 0u) {
            var nee_rng = hash(
                probe_index
                    ^ (ray_index_in_probe * 0xA24BAEDDu)
                    ^ (u32(ddgi_params.frame_index) * 0x9E3779B9u)
            );
            nee_rng = random_seed(nee_rng);
            let light_rand = rand_float(nee_rng);
            let emissive_bucket_pdf = f32(num_emissive_lights) / f32(total_light_count);
            let analytic_bucket_pdf = 1.0 - emissive_bucket_pdf;
            let select_emissive =
                num_emissive_lights > 0u && (num_lights == 0u || light_rand >= analytic_bucket_pdf);

            if (!select_emissive) {
                nee_rng = random_seed(nee_rng);
                let analytic_rand = rand_float(nee_rng);
                let selected_light_idx = u32(analytic_rand * f32(num_lights)) % max(num_lights, 1u);
                let light = dense_lights_buffer.lights[selected_light_idx];
                let shadow_dir = get_light_dir(light, p_world);
                let attenuation = get_light_attenuation(light, p_world);
                let light_distance = select(1e30, length(light.position.xyz - p_world), light.light_type != 0.0);
                let shadow_t_max = light_distance * 0.999;
                let analytic_light_pdf = analytic_bucket_pdf * (1.0 / max(f32(num_lights), 1.0));
                let analytic_light_scale = 1.0 / max(analytic_light_pdf, 1e-6);

                probe_ray_data.rays[index].nee_light_dir_type = vec4<f32>(shadow_dir, 0.0);
                probe_ray_data.rays[index].nee_light_radiance = vec4<f32>(
                    light.color.rgb * light.intensity * attenuation * analytic_light_scale,
                    0.0
                );
                process_shadow_visibility(index, p_world, shadow_dir, shadow_t_max);
            } else {
                var emissive_pdf = 0.0;
                let emissive_idx = sample_weighted_emissive_light(
                    &nee_rng,
                    num_emissive_lights,
                    &emissive_pdf
                );
                let emissive_light = emissive_lights_buffer.lights[emissive_idx];
                let to_emissive = emissive_light.position_radius.xyz - p_world;
                let distance_sq = max(dot(to_emissive, to_emissive), 1e-6);
                let distance = sqrt(distance_sq);
                let shadow_dir = to_emissive / distance;
                let light_facing = max(dot(emissive_light.normal_area.xyz, -shadow_dir), 0.0);
                let solid_angle_scale = emissive_light.normal_area.w / distance_sq;
                let shadow_t_max = max(0.0, distance - emissive_light.position_radius.w) * 0.999;
                let emissive_light_pdf = emissive_bucket_pdf * emissive_pdf;
                let emissive_light_scale = 1.0 / max(emissive_light_pdf, 1e-6);

                probe_ray_data.rays[index].nee_light_dir_type = vec4<f32>(shadow_dir, 1.0);
                probe_ray_data.rays[index].nee_light_radiance = vec4<f32>(
                    emissive_light.radiance_weight.xyz * light_facing * solid_angle_scale * emissive_light_scale,
                    0.0
                );
                process_shadow_visibility(index, p_world, shadow_dir, shadow_t_max);
            }
        }
    }
}

// =============================================================================
// Main
// =============================================================================
@compute @workgroup_size(256, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let active_ray_count = probe_ray_data.header.active_ray_count;

    if (gid.x >= active_ray_count || probe_ray_data.rays[gid.x].state_u32.y == 0u) {
        return;
    }

    let probe_index = probe_ray_data.rays[gid.x].meta_u32.x;
    let ray_index_in_probe = probe_ray_data.rays[gid.x].meta_u32.y;
    let probe_position = ddgi_probe_world_position_from_index(&ddgi_params, probe_index);

    let ray_dir = probe_ray_data.rays[gid.x].ray_dir_prim.xyz;
    process_primary_ray(gid.x, probe_position, ray_dir, probe_index, ray_index_in_probe);
}

