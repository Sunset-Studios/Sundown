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
#include "blas_common.wgsl"
#include "gi/gi_common.wgsl"

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read_write> gi_counters: GICounters;
@group(1) @binding(2) var<storage, read_write> pixel_path_state: array<PixelPathState>;
@group(1) @binding(3) var<storage, read_write> ray_work_queue: array<u32>;
@group(1) @binding(4) var<storage, read> tlas_bvh2_bounds: array<AABB>;
@group(1) @binding(5) var<storage, read> tlas_bvh8_nodes: array<BVH8Node>;
@group(1) @binding(6) var<storage, read> blas_atlas: BLASAtlas;
@group(1) @binding(7) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(8) var<storage, read> mesh_asset_ids: array<u32>;

// =============================================================================
// BLAS TRAVERSAL
// =============================================================================

fn trace_blas(
    ray_world: ptr<function, Ray>,
    ray_local: ptr<function, Ray>,
    mesh_asset_id: u32
) -> RayHit {
    var hit: RayHit;
    hit.position_and_t = vec4f((*ray_world).origin_and_tmin.xyz, (*ray_local).direction_and_tmax.w);
    hit.normal_and_user_data = vec4f(0.0, 0.0, 0.0, -1.0);

    var current_ray = *ray_local;

    var node_stack: array<u32, NODE_STACK_SIZE>;
    node_stack[0] = atlas_load_directory_entry_bvh8_base(mesh_asset_id);
    var stack_size = 1u;

    while (stack_size > 0u) {
        stack_size = stack_size - 1u;

        let node_idx = node_stack[stack_size];
        if (node_idx == INVALID_IDX) { continue; }

        let leaf_mask = atlas_load_bvh8_leaf_mask(node_idx);
        
        for (var i = 0u; i < 8u; i = i + 1u) {
            let child_raw = atlas_load_bvh8_child(node_idx, i);
            if (child_raw < 0.0) { continue; }

            let child_idx = u32(child_raw);

            if (((leaf_mask >> i) & 1u) != 0u) {
                // Leaf node: triangle intersection
                let leaf_indices = atlas_load_bvh8_leaf_indices(node_idx, i);
                let t_tri = intersect_triangle(
                    &current_ray,
                    vertex_buffer[leaf_indices.x].position.xyz,
                    vertex_buffer[leaf_indices.y].position.xyz,
                    vertex_buffer[leaf_indices.z].position.xyz
                );
                if (t_tri >= current_ray.origin_and_tmin.w && t_tri < current_ray.direction_and_tmax.w) {
                    hit.position_and_t.w = t_tri;
                    hit.normal_and_user_data.w = f32(child_idx);
                    hit.hit_triangle_data = leaf_indices;
                    current_ray.direction_and_tmax.w = t_tri;
                }
            } else {
                // Internal node: AABB test before push
                let t_aabb_child = intersect_aabb(
                    &current_ray,
                    atlas_load_bvh8_node_min(child_idx),
                    atlas_load_bvh8_node_max(child_idx)
                );

                let is_better_child = t_aabb_child.x <= t_aabb_child.y
                    && t_aabb_child.x >= current_ray.origin_and_tmin.w
                    && t_aabb_child.x < current_ray.direction_and_tmax.w;
                if (is_better_child) {
                    node_stack[stack_size] = child_idx;
                    stack_size = stack_size + 1u;
                }
            }
        }
    }

    return hit;
}

// =============================================================================
// TLAS TRAVERSAL (CLOSEST HIT)
// =============================================================================

fn trace_hit(ray: ptr<function, Ray>) -> RayHit {
    var hit: RayHit;
    hit.position_and_t = vec4f((*ray).origin_and_tmin.xyz, (*ray).direction_and_tmax.w);
    hit.normal_and_user_data = vec4f(0.0, 0.0, 0.0, -1.0);

    var node_stack: array<u32, NODE_STACK_SIZE>;
    node_stack[0] = 0u;
    var stack_size = 1u;

    var current_ray = *ray;

    while (stack_size > 0u) {
        stack_size = stack_size - 1u;

        var node_idx = node_stack[stack_size];
        if (node_idx == INVALID_IDX) { continue; }

        let leaf_mask = bitcast<u32>(tlas_bvh8_nodes[node_idx].min.w);

        for (var i = 0u; i < 8u; i = i + 1u) {
            let child_raw = bvh8_child(tlas_bvh8_nodes[node_idx], i);
            if (child_raw < 0.0) { continue; }

            let child_idx = u32(child_raw);

            if (((leaf_mask >> i) & 1u) != 0u) {
                // Leaf: instance bounds test
                let t_leaf = intersect_aabb(
                    &current_ray,
                    tlas_bvh2_bounds[child_idx].min.xyz,
                    tlas_bvh2_bounds[child_idx].max.xyz
                );

                if (t_leaf.x <= t_leaf.y && max(t_leaf.x, current_ray.origin_and_tmin.w) < hit.position_and_t.w) {
                    let prim_store = u32(tlas_bvh2_bounds[child_idx].min.w);
                    let mesh_id = mesh_asset_ids[prim_store];

                    var ray_local = build_local_ray(
                        &current_ray,
                        entity_transforms[prim_store].transform,
                        entity_transforms[prim_store].transpose_inverse_model_matrix
                    );
                    
                    let blas_hit = trace_blas(
                        &current_ray,
                        &ray_local,
                        mesh_id
                    );

                    if (blas_hit.normal_and_user_data.w >= 0.0) {
                        hit = blas_hit;
                        hit.prim_meshid_padding = vec4f(f32(prim_store), f32(mesh_id), 0.0, 0.0);
                        hit.ray_local = ray_local;
                        current_ray.direction_and_tmax.w = min(current_ray.direction_and_tmax.w, hit.position_and_t.w);
                    }
                }
            } else {
                // Internal node: AABB test before push
                let t_aabb_child = intersect_aabb(
                    &current_ray,
                    tlas_bvh8_nodes[child_idx].min.xyz,
                    tlas_bvh8_nodes[child_idx].max.xyz
                );
                let is_better_child = t_aabb_child.x <= t_aabb_child.y
                    && t_aabb_child.x >= current_ray.origin_and_tmin.w
                    && t_aabb_child.x < hit.position_and_t.w;

                if (is_better_child) {
                    node_stack[stack_size] = child_idx;
                    stack_size = stack_size + 1u;
                }
            }
        }
    }

    return hit;
}

// =============================================================================
// BLAS ANY-HIT (SHADOW RAYS)
// =============================================================================

fn trace_blas_any_hit(
    ray_world: ptr<function, Ray>,
    ray_local: ptr<function, Ray>,
    entity_transform: mat4x4f,
    mesh_asset_id: u32,
) -> bool {
    let mesh_directory_entry = atlas_load_directory_entry(mesh_asset_id);
    let bvh8_base = mesh_directory_entry.bvh8_base;

    var node_stack: array<u32, NODE_STACK_SIZE>;
    node_stack[0] = bvh8_base;
    var stack_size = 1u;

    loop {
        if (stack_size == 0u) { break; }
        stack_size = stack_size - 1u;

        var node_idx = node_stack[stack_size];
        if (node_idx == INVALID_IDX) { continue; }

        let node = atlas_load_bvh8_node(node_idx);
        
        if (stack_size < NODE_STACK_SIZE) {
            let leaf_mask = bitcast<u32>(node.min.w);

            for (var i = 0u; i < 8u; i = i + 1u) {
                let child_raw = bvh8_child(node, i);
                if (child_raw < 0.0) { continue; }

                let child_idx = u32(child_raw);

                if (((leaf_mask >> i) & 1u) != 0u) {
                let leaf_indices = atlas_load_bvh8_leaf_indices(node_idx, i);
                    let v0 = vertex_buffer[leaf_indices.x].position.xyz;
                    let v1 = vertex_buffer[leaf_indices.y].position.xyz;
                    let v2 = vertex_buffer[leaf_indices.z].position.xyz;
                    let t_tri = intersect_triangle(ray_local, v0, v1, v2);
                    if (t_tri >= ray_local.origin_and_tmin.w && t_tri < ray_local.direction_and_tmax.w) {
                        return true;
                    }
                } else {
                let child_node = atlas_load_bvh8_node(child_idx);
                    let t_aabb_child = intersect_aabb(ray_local, child_node.min.xyz, child_node.max.xyz);

                    if (t_aabb_child.x <= t_aabb_child.y && t_aabb_child.x >= ray_local.origin_and_tmin.w && t_aabb_child.x < ray_local.direction_and_tmax.w) {
                        node_stack[stack_size] = child_idx;
                        stack_size = stack_size + 1u;
                    }
                }
            }
        }
    }

    return false;
}

// =============================================================================
// TLAS ANY-HIT (SHADOW RAYS)
// =============================================================================

fn trace_hit_any(ray: ptr<function, Ray>) -> bool {
    var node_stack: array<u32, NODE_STACK_SIZE>;
    node_stack[0] = 0u;
    var stack_size = 1u;

    var current_ray = *ray;

    loop {
        if (stack_size == 0u) { break; }
        stack_size = stack_size - 1u;

        var node_idx = node_stack[stack_size];
        if (node_idx == INVALID_IDX) { continue; }

        let current_node = tlas_bvh8_nodes[node_idx];
        
        if (stack_size < NODE_STACK_SIZE) {
            let leaf_mask = bitcast<u32>(current_node.min.w);

            for (var i = 0u; i < 8u; i = i + 1u) {
                let child_raw = bvh8_child(current_node, i);
                if (child_raw < 0.0) { continue; }

                let child_idx = u32(child_raw);

                if (((leaf_mask >> i) & 1u) != 0u) {
                    let leaf_bounds = tlas_bvh2_bounds[child_idx];
                    let t_leaf = intersect_aabb(&current_ray, leaf_bounds.min.xyz, leaf_bounds.max.xyz);

                    if (t_leaf.x <= t_leaf.y && max(t_leaf.x, current_ray.origin_and_tmin.w) < current_ray.direction_and_tmax.w) {
                        let prim_store = u32(leaf_bounds.min.w);
                        let mesh_id = mesh_asset_ids[prim_store];
                        let entity_transform = entity_transforms[prim_store];

                        var ray_local = build_local_ray(
                            &current_ray,
                            entity_transform.transform,
                            entity_transform.transpose_inverse_model_matrix
                        );
                        
                        if (trace_blas_any_hit(
                            &current_ray,
                            &ray_local,
                            entity_transform.transform,
                            mesh_id
                        )) {
                            return true;
                        } 
                    }
                } else {
                let child_node = tlas_bvh8_nodes[child_idx];
                    let t_aabb_child = intersect_aabb(&current_ray, child_node.min.xyz, child_node.max.xyz);

                    if (t_aabb_child.x <= t_aabb_child.y && max(t_aabb_child.x, current_ray.origin_and_tmin.w) < current_ray.direction_and_tmax.w) {
                        node_stack[stack_size] = child_idx;
                        stack_size = stack_size + 1u;
                    }
                }
            }
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
    ray.inv_direction = vec4f(
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
    ray.inv_direction = vec4f(
        1.0 / max(abs(ray.direction_and_tmax.x), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.x < 0.0),
        1.0 / max(abs(ray.direction_and_tmax.y), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.y < 0.0),
        1.0 / max(abs(ray.direction_and_tmax.z), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.z < 0.0),
        0.0
    );

    let hit_result = trace_hit(&ray);

    if (hit_result.normal_and_user_data.w >= 0.0) {
        let tri_id_local = u32(hit_result.normal_and_user_data.w);
        let prim_store = u32(hit_result.prim_meshid_padding.x);

        let entity_transform = entity_transforms[prim_store];

        let t_tri = hit_result.position_and_t.w;
        let p_local = hit_result.ray_local.origin_and_tmin.xyz + hit_result.ray_local.direction_and_tmax.xyz * t_tri;
        let p_world = (entity_transform.transform * vec4f(p_local, 1.0)).xyz;

        let v0i = hit_result.hit_triangle_data.x;
        let v1i = hit_result.hit_triangle_data.y;
        let v2i = hit_result.hit_triangle_data.z;

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
        pixel_path_state[ray_index].origin_tmin = vec4f(p_world, t_tri);
        pixel_path_state[ray_index].direction_tmax = vec4f(ray_dir, hit_result.prim_meshid_padding.x);
        pixel_path_state[ray_index].normal_section_index = vec4f(world_n, f32(vertex_buffer[v0i].section_index));
        pixel_path_state[ray_index].hit_attr0 = vec4f(world_t, uv_hit.x);
        pixel_path_state[ray_index].hit_attr1 = vec4f(world_b, uv_hit.y);
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
