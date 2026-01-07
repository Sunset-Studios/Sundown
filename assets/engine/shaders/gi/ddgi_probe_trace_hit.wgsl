// =============================================================================
// DDGI Probe Ray Trace - Hit Pass
// - Traces primary rays from world-space probes against the BVH
// - Writes compact hit attributes for the shade pass
// - Intentionally avoids ALL material + texture bindings to reduce binding count
// =============================================================================
#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "blas_common.wgsl"
#include "gi/ddgi_common.wgsl"

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(2) var<storage, read> probe_update_indices: array<u32>;
@group(1) @binding(3) var<storage, read_write> probe_ray_hits: array<DDGIProbeRayHit>;
@group(1) @binding(4) var<storage, read> tlas_bvh2_bounds: array<AABB>;
@group(1) @binding(5) var<storage, read> tlas_bvh8_nodes: array<BVH8Node>;
@group(1) @binding(6) var<storage, read> blas_atlas: BLASAtlas;
@group(1) @binding(7) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(8) var<storage, read> mesh_asset_ids: array<u32>;

// =============================================================================
// DDGI Probe Direction Sampling
// - Uniform spherical directions so probes trace in all
//   directions around the probe volume.
// - Uses a stochastically-rotated Fibonacci spiral point set, following the
//   "spherical Fibonacci" style sampling described by McGuire et al.
// =============================================================================
fn ddgi_fibonacci_sphere_direction(ray_index: u32, ray_count: u32, rotation_01: f32) -> vec3<f32> {
    let n = max(ray_count, 1u);
    let i = min(ray_index, n - 1u);

    // Stratified latitude, uniform in cos(theta) for uniform area on the sphere.
    let u = (f32(i) + 0.5) / f32(n);           // (0,1)
    let cos_theta = 1.0 - 2.0 * u;             // [-1,1]
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));

    // Fibonacci spiral azimuth with a per-probe Cranley-Patterson rotation.
    let phi = 2.0 * PI * fract(f32(i) * GOLDEN_RATIO_CONJUGATE + rotation_01);

    return vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

fn ddgi_probe_ray_direction_spherical_fibonacci(
    probe_index: u32,
    ray_index_in_probe: u32,
    rays_per_probe: u32
) -> vec3<f32> {
    // One stochastic rotation per probe per frame (shared across all rays in the probe).
    // This avoids static banding while preserving the uniform Fibonacci distribution.
    var probe_rng = hash(probe_index ^ (u32(gi_params.frame_index) * 0x9E3779B9u));
    // Rotation for the Fibonacci spiral parameterization (Cranley-Patterson offset).
    let rotation_01 = rand_float(probe_rng);
    // Randomly rotate the entire point set in 3D (avoid locking the pattern to world axes).
    probe_rng = random_seed(probe_rng);
    let r1 = rand_float(probe_rng);
    probe_rng = random_seed(probe_rng);
    let r2 = rand_float(probe_rng);

    let z = 1.0 - 2.0 * r1;
    let rot_phi = 2.0 * PI * r2;
    let r_xy = sqrt(max(1.0 - z * z, 0.0));
    let z_axis = vec3<f32>(cos(rot_phi) * r_xy, sin(rot_phi) * r_xy, z);

    let dir_local = ddgi_fibonacci_sphere_direction(ray_index_in_probe, rays_per_probe, rotation_01);
    return orthonormalize(z_axis) * dir_local;
}

// =============================================================================
// BLAS TRAVERSAL
// =============================================================================

fn trace_blas(
    ray_local: ptr<function, Ray>,
    mesh_asset_id: u32
) -> RayHitCompact {
    var result = make_miss_ray_hit_compact((*ray_local).direction_and_tmax.w);
    result.mesh_id = mesh_asset_id;

    var current_ray = *ray_local;

    var node_stack: array<u32, NODE_STACK_SIZE>;
    node_stack[0] = atlas_load_directory_entry_bvh8_base(mesh_asset_id);
    var stack_size = 1u;

    var leaf_mask = 0u;
    var pending_child_idx = 0u;
    var pending_child_tmin = 0.0;
    var have_pending_child = false;
    var node_idx = INVALID_IDX;
    var child_raw = -1.0;
    var child_idx = INVALID_IDX;
    var is_better_hit = false;
    var is_better_child = false;
    var current_is_farther = false;
    var push_idx = INVALID_IDX;
    var keep_idx = INVALID_IDX;
    var keep_tmin = 0.0;
    var t_tri = 0.0;
    var t_aabb_child = vec2<f32>(0.0, 0.0);

    while (stack_size > 0u) {
        stack_size = stack_size - 1u;

        node_idx = node_stack[stack_size];
        if (node_idx == INVALID_IDX) { continue; }

        leaf_mask = atlas_load_bvh8_leaf_mask(node_idx);
        pending_child_idx = 0u;
        pending_child_tmin = 0.0;
        have_pending_child = false;

        for (var i = 0u; i < 8u; i = i + 1u) {
            child_raw = atlas_load_bvh8_child(node_idx, i);
            if (child_raw < 0.0) { continue; }

            child_idx = u32(child_raw);

            if (((leaf_mask >> i) & 1u) != 0u) {
                // Leaf node: triangle intersection
                let leaf_indices = atlas_load_bvh8_leaf_indices(node_idx, i);
                t_tri = intersect_triangle(
                    &current_ray,
                    vertex_buffer[leaf_indices.x].position.xyz,
                    vertex_buffer[leaf_indices.y].position.xyz,
                    vertex_buffer[leaf_indices.z].position.xyz
                );
                is_better_hit = t_tri >= current_ray.origin_and_tmin.w && t_tri < current_ray.direction_and_tmax.w;
                if (is_better_hit) {
                    result.t_hit = t_tri;
                    result.tri_id_local = child_idx;
                    result.tri_indices = leaf_indices;
                    result.has_hit = 1u;
                    current_ray.direction_and_tmax.w = t_tri;
                }
            } else {
                // Internal node: AABB test before push
                t_aabb_child = intersect_aabb(
                    &current_ray,
                    atlas_load_bvh8_node_min(child_idx),
                    atlas_load_bvh8_node_max(child_idx)
                );

                is_better_child = t_aabb_child.x <= t_aabb_child.y
                    && t_aabb_child.x >= current_ray.origin_and_tmin.w
                    && t_aabb_child.x < current_ray.direction_and_tmax.w;
                if (is_better_child) {
                    #if BVH_TRAVERSAL_ORDER_CHILDREN
                    // Pairwise ordering: keep the nearest as pending, push the farther now.
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
            }
        }

        #if BVH_TRAVERSAL_ORDER_CHILDREN
        if (have_pending_child) {
            node_stack[stack_size] = pending_child_idx;
            stack_size = stack_size + 1u;
        }
        #endif
    }

    return result;
}

// =============================================================================
// TLAS TRAVERSAL (CLOSEST HIT)
// =============================================================================

fn trace_hit(ray: ptr<function, Ray>) -> RayHitCompact {
    var result = make_miss_ray_hit_compact((*ray).direction_and_tmax.w);

    var node_stack: array<u32, NODE_STACK_SIZE>;
    node_stack[0] = 0u;
    var stack_size = 1u;

    var current_ray = *ray;

    var leaf_mask = 0u;
    var pending_child_idx = 0u;
    var pending_child_tmin = 0.0;
    var have_pending_child = false;
    var node_idx = INVALID_IDX;
    var child_raw = -1.0;
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

    while (stack_size > 0u) {
        stack_size = stack_size - 1u;

        node_idx = node_stack[stack_size];
        if (node_idx == INVALID_IDX) { continue; }

        var node = tlas_bvh8_nodes[node_idx];

        leaf_mask = bitcast<u32>(node.min.w);
        pending_child_idx = 0u;
        pending_child_tmin = 0.0;
        have_pending_child = false;

        for (var i = 0u; i < 8u; i = i + 1u) {
            child_raw = bvh8_child(&node, i);
            if (child_raw < 0.0) { continue; }

            child_idx = u32(child_raw);

            if (((leaf_mask >> i) & 1u) != 0u) {
                // Leaf: instance bounds test
                t_leaf = intersect_aabb(
                    &current_ray,
                    tlas_bvh2_bounds[child_idx].min.xyz,
                    tlas_bvh2_bounds[child_idx].max.xyz
                );

                if (t_leaf.x <= t_leaf.y && max(t_leaf.x, current_ray.origin_and_tmin.w) < result.t_hit) {
                    prim_store = u32(tlas_bvh2_bounds[child_idx].min.w);
                    mesh_id = mesh_asset_ids[prim_store];

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
                // Internal node: AABB test before push
                t_aabb_child = intersect_aabb(
                    &current_ray,
                    tlas_bvh8_nodes[child_idx].min.xyz,
                    tlas_bvh8_nodes[child_idx].max.xyz
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
            }
        }

        #if BVH_TRAVERSAL_ORDER_CHILDREN
        if (have_pending_child) {
            node_stack[stack_size] = pending_child_idx;
            stack_size = stack_size + 1u;
        }
        #endif
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
    let mesh_directory_entry = atlas_load_directory_entry(mesh_asset_id);
    let bvh8_base = mesh_directory_entry.bvh8_base;

    var node_stack: array<u32, NODE_STACK_SIZE>;
    node_stack[0] = bvh8_base;
    var stack_size = 1u;

    var node_idx = INVALID_IDX;
    var leaf_mask = 0u;
    var child_raw = -1.0;
    var child_idx = INVALID_IDX;
    var t_tri = 0.0;
    var t_aabb_child = vec2<f32>(0.0, 0.0);
    var leaf_indices = vec4<u32>(0u, 0u, 0u, 0u);
    var v0 = vec3<f32>(0.0, 0.0, 0.0);
    var v1 = vec3<f32>(0.0, 0.0, 0.0);
    var v2 = vec3<f32>(0.0, 0.0, 0.0);
    var node: BVH8Node;
    var child_node: BVH8Node;

    loop {
        if (stack_size == 0u) { break; }
        stack_size = stack_size - 1u;

        node_idx = node_stack[stack_size];
        if (node_idx == INVALID_IDX) { continue; }

        node = atlas_load_bvh8_node(node_idx);
        
        leaf_mask = bitcast<u32>(node.min.w);

        for (var i = 0u; i < 8u; i = i + 1u) {
            child_raw = bvh8_child(&node, i);
            if (child_raw < 0.0) { continue; }

            child_idx = u32(child_raw);

            if (((leaf_mask >> i) & 1u) != 0u) {
                leaf_indices = atlas_load_bvh8_leaf_indices(node_idx, i);
                v0 = vertex_buffer[leaf_indices.x].position.xyz;
                v1 = vertex_buffer[leaf_indices.y].position.xyz;
                v2 = vertex_buffer[leaf_indices.z].position.xyz;
                t_tri = intersect_triangle(ray_local, v0, v1, v2);
                if (t_tri >= ray_local.origin_and_tmin.w && t_tri < ray_local.direction_and_tmax.w) {
                    return true;
                }
            } else {
                child_node = atlas_load_bvh8_node(child_idx);
                t_aabb_child = intersect_aabb(ray_local, child_node.min.xyz, child_node.max.xyz);

                if (t_aabb_child.x <= t_aabb_child.y && t_aabb_child.x >= ray_local.origin_and_tmin.w && t_aabb_child.x < ray_local.direction_and_tmax.w) {
                    node_stack[stack_size] = child_idx;
                    stack_size = stack_size + 1u;
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

    var node_idx = INVALID_IDX;
    var child_raw = -1.0;
    var child_idx = INVALID_IDX;
    var leaf_mask = 0u;
    var t_leaf = vec2<f32>(0.0, 0.0);
    var t_aabb_child = vec2<f32>(0.0, 0.0);
    var leaf_bounds: AABB;
    var prim_store = 0u;
    var mesh_id = 0u;
    var entity_transform: EntityTransform;
    var ray_local: Ray;
    var current_node: BVH8Node;
    var child_node: BVH8Node;

    loop {
        if (stack_size == 0u) { break; }
        stack_size = stack_size - 1u;

        node_idx = node_stack[stack_size];
        if (node_idx == INVALID_IDX) { continue; }

        current_node = tlas_bvh8_nodes[node_idx];
        
        leaf_mask = bitcast<u32>(current_node.min.w);

        for (var i = 0u; i < 8u; i = i + 1u) {
            child_raw = bvh8_child(&current_node, i);
            if (child_raw < 0.0) { continue; }

            child_idx = u32(child_raw);

            if (((leaf_mask >> i) & 1u) != 0u) {
                leaf_bounds = tlas_bvh2_bounds[child_idx];
                t_leaf = intersect_aabb(&current_ray, leaf_bounds.min.xyz, leaf_bounds.max.xyz);

                if (t_leaf.x <= t_leaf.y && max(t_leaf.x, current_ray.origin_and_tmin.w) < current_ray.direction_and_tmax.w) {
                    prim_store = u32(leaf_bounds.min.w);
                    mesh_id = mesh_asset_ids[prim_store];
                    entity_transform = entity_transforms[prim_store];

                    ray_local = build_local_ray(
                        &current_ray,
                        entity_transform.transform,
                        entity_transform.transpose_inverse_model_matrix
                    );
                    
                    if (trace_blas_any_hit(
                        &ray_local,
                        mesh_id
                    )) {
                        return true;
                    } 
                }
            } else {
                child_node = tlas_bvh8_nodes[child_idx];
                t_aabb_child = intersect_aabb(&current_ray, child_node.min.xyz, child_node.max.xyz);

                if (t_aabb_child.x <= t_aabb_child.y && max(t_aabb_child.x, current_ray.origin_and_tmin.w) < current_ray.direction_and_tmax.w) {
                    node_stack[stack_size] = child_idx;
                    stack_size = stack_size + 1u;
                }
            }
        }
    }

    return false;
}

// =============================================================================
// HELPER: Process a primary ray and write hit attributes
// =============================================================================
fn process_primary_ray(index: u32, probe_position: vec3<f32>, ray_dir: vec3<f32>) {
    var ray: Ray;
    ray.origin_and_tmin = vec4f(probe_position + ray_dir * 0.001, 0.001);
    ray.direction_and_tmax = vec4f(ray_dir, 1e30);
    ray.inv_direction = vec4f(
        1.0 / max(abs(ray_dir.x), 1e-8) * select(1.0, -1.0, ray_dir.x < 0.0),
        1.0 / max(abs(ray_dir.y), 1e-8) * select(1.0, -1.0, ray_dir.y < 0.0),
        1.0 / max(abs(ray_dir.z), 1e-8) * select(1.0, -1.0, ray_dir.z < 0.0),
        0.0
    );

    let hit_result = trace_hit(&ray);

    // Default: miss
    probe_ray_hits[index].state_u32 = vec4<u32>(0u, 1u, 0u, 0xffffffffu);

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
        let p_world = (entity_transform.transform * vec4f(p_local, 1.0)).xyz;

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

        probe_ray_hits[index].hit_pos_t = vec4f(p_world, t_tri);
        probe_ray_hits[index].ray_dir_prim = vec4f(ray_dir, f32(prim_store));
        probe_ray_hits[index].world_n_section = vec4f(world_n, f32(section_index));
        probe_ray_hits[index].world_t_uvx = vec4f(world_t, uv_hit.x);
        probe_ray_hits[index].world_b_uvy = vec4f(world_b, uv_hit.y);
        probe_ray_hits[index].state_u32.w = tri_id_local;
    }
}

// =============================================================================
// Main
// =============================================================================
@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rays_per_probe = u32(ddgi_params.probe_counts.y);
    let probes_per_frame = u32(ddgi_params.probe_counts.z);
    let total_rays = probes_per_frame * rays_per_probe;

    if (gid.x >= total_rays) {
        return;
    }

    let probe_slot = gid.x / rays_per_probe;
    let probe_index = probe_update_indices[probe_slot];
    let probe_position = ddgi_probe_world_position_from_index(&ddgi_params, probe_index);

    // Uniform spherical directions (stochastically-rotated Fibonacci spiral).
    let ray_dir = ddgi_probe_ray_direction_spherical_fibonacci(
        probe_index,
        gid.x - probe_slot * rays_per_probe,
        rays_per_probe
    );

    process_primary_ray(gid.x, probe_position, ray_dir);
}

