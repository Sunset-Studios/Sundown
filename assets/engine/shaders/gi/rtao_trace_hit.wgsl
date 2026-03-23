// =============================================================================
// RTAO BVH TRAVERSAL (SIMPLIFIED)
// =============================================================================
//
// Traces rays for Ray Traced Ambient Occlusion only:
//   - Closest-hit TLAS/BLAS traversal (same as full path tracer)
//   - No shadow rays, no NEE, no any-hit
//   - On hit: set state_u32.w = 0 for resolve; on miss: leave 0xffffffff
//
// =============================================================================

#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "gi/gi_common.wgsl"

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read_write> gi_counters: GICounters;
@group(1) @binding(2) var<storage, read_write> pixel_path_state: array<AOPixelPathState>;
@group(1) @binding(3) var<storage, read_write> ray_work_queue: array<u32>;
@group(1) @binding(4) var<storage, read> tlas_bvh2_bounds: array<AABB>;
@group(1) @binding(5) var<uniform> tlas_bvh_info: BVHInfo;
@group(1) @binding(6) var<storage, read> blas_bvh2_nodes: array<AABB>;
@group(1) @binding(7) var<storage, read> blas_directory: array<MeshDirectoryEntry>;
@group(1) @binding(8) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(9) var<storage, read> index_buffer: array<u32>;

// =============================================================================
// BLAS TRAVERSAL (CLOSEST HIT)
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
            v0 = vertex_position(vertex_buffer[v0i]);
            v1 = vertex_position(vertex_buffer[v1i]);
            v2 = vertex_position(vertex_buffer[v2i]);
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
// RTAO: Process primary ray — record hit for AO (state_u32.w only)
// =============================================================================

fn process_rtao_ray(ray_index: u32) {
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
        pixel_path_state[ray_index].state_u32.w = 0u;
        pixel_path_state[ray_index].origin_tmin.w = hit_result.t_hit;
    }
}

// =============================================================================
// MAIN COMPUTE
// =============================================================================

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rays_per_pixel = u32(gi_params.screen_ray_count);
    let gi_resolution = vec2<u32>(u32(gi_params.gi_resolution_x), u32(gi_params.gi_resolution_y));
    let total_pixels = gi_resolution.x * gi_resolution.y;
    let total_rays = total_pixels * rays_per_pixel;

    if (gid.x >= total_rays) {
        return;
    }

    let queue_count = atomicLoad(&gi_counters.ray_queue_count);

    loop {
        let queue_index = atomicAdd(&gi_counters.ray_queue_primary_head, 1u);
        if (queue_index >= queue_count) {
            break;
        }
        let ray_slot = ray_work_queue[queue_index];
        process_rtao_ray(ray_slot);
    }
}
