// =============================================================================
// GI-1.0 Screen Probe Ray Tracing - Hit Pass
// - Traces rays from screen probes against BVH
// - Writes hit information to path state
// - Uses same traversal logic as path tracer for consistency
// =============================================================================
#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "blas_common.wgsl"

const NODE_STACK_SIZE = 16;

struct GIParams {
    screen_probe_spawn_rate: u32,
    screen_probe_size: u32,
    screen_ray_count: u32,
    world_cache_size: u32,
    max_screen_probes: u32,
    frame_index: u32,
    reset_caches: u32,
    indirect_boost: u32,
    upscale_x: u32,
    upscale_y: u32,
    cell_size_heuristic: u32,
    padding: u32,
};

struct ProbePathState {
    origin_tmin: vec4<f32>,
    direction_tmax: vec4<f32>,
    normal_section_index: vec4<f32>,
    state_u32: vec4<u32>,
    hit_attr0: vec4<f32>,
    hit_attr1: vec4<f32>,
    shadow_origin: vec4<f32>,
    shadow_direction: vec4<f32>,
    shadow_radiance: vec4<f32>,
};

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read_write> probe_path_state: array<ProbePathState>;
@group(1) @binding(2) var<storage, read> screen_probe_counter: array<u32>;
@group(1) @binding(3) var<storage, read> tlas_bvh2_bounds: array<AABB>;
@group(1) @binding(4) var<storage, read> tlas_bvh4_nodes: array<BVH4Node>;
@group(1) @binding(5) var<storage, read> blas_atlas: BLASAtlas;
@group(1) @binding(6) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(7) var<storage, read> index_buffer: array<u32>;
@group(1) @binding(8) var<storage, read> mesh_asset_ids: array<u32>;

// =============================================================================
// BVH Traversal 
// =============================================================================

fn trace_blas(
    ray_world: ptr<function, Ray>,
    ray_local: ptr<function, Ray>,
    entity_transform: mat4x4f,
    transpose_inverse_model_matrix: mat4x4f,
    mesh_asset_id: u32
) -> RayHit {
    let mesh_directory_entry = atlas_load_directory_entry(mesh_asset_id);
    let bvh4_base = mesh_directory_entry.bvh4_base;
    let first_vertex = mesh_directory_entry.first_vertex;
    let first_index = mesh_directory_entry.first_index;

    var hit: RayHit;
    hit.position_and_t = vec4f((*ray_world).origin_and_tmin.xyz, (*ray_local).direction_and_tmax.w);
    hit.normal_and_user_data = vec4f(0.0, 0.0, 0.0, -1.0);

    var current_ray = *ray_local;

    var node_stack: array<u32, NODE_STACK_SIZE>;
    node_stack[0] = bvh4_base;
    var stack_size = 1u;

    loop {
        if (stack_size == 0u) { break; }
        stack_size = stack_size - 1u;

        var node_idx = node_stack[stack_size];
        if (node_idx == INVALID_IDX) { continue; }

        let node = atlas_load_bvh4_node(node_idx);
        let t_aabb = intersect_aabb(current_ray, node.min.xyz, node.max.xyz);

        if (t_aabb.x <= t_aabb.y && t_aabb.x >= current_ray.origin_and_tmin.w && t_aabb.x < hit.position_and_t.w) {
            if (stack_size < NODE_STACK_SIZE) {
                let leaf_mask = bitcast<u32>(node.min.w);
                var child_data: array<vec2<f32>, 4>;
                var valid_children = 0u;

                for (var i = 0u; i < 4u; i = i + 1u) {
                    let child_raw = node.children[i];
                    if (child_raw < 0.0) { continue; }

                    let is_leaf_child = ((leaf_mask >> i) & 1u) != 0u;
                    let child_idx = u32(child_raw);

                    if (is_leaf_child) {
                        // BLAS BVH4 leaf nodes store triangle IDs directly - no AABB indirection!
                        let tri_id_local = child_idx;
                        let i0 = index_buffer[first_index + tri_id_local * 3u + 0u];
                        let i1 = index_buffer[first_index + tri_id_local * 3u + 1u];
                        let i2 = index_buffer[first_index + tri_id_local * 3u + 2u];

                        let v0 = vertex_buffer[first_vertex + i0].position.xyz;
                        let v1 = vertex_buffer[first_vertex + i1].position.xyz;
                        let v2 = vertex_buffer[first_vertex + i2].position.xyz;

                        let t_tri = intersect_triangle(current_ray, v0, v1, v2);
                        if (t_tri >= current_ray.origin_and_tmin.w && t_tri < current_ray.direction_and_tmax.w) {
                            hit.position_and_t.w = t_tri;
                            hit.normal_and_user_data.w = f32(tri_id_local);
                        }
                    } else {
                        let child_node = atlas_load_bvh4_node(child_idx);
                        let t_aabb_child = intersect_aabb(current_ray, child_node.min.xyz, child_node.max.xyz);

                        if (t_aabb_child.x <= t_aabb_child.y && t_aabb_child.x >= current_ray.origin_and_tmin.w && t_aabb_child.x < hit.position_and_t.w) {
                            child_data[valid_children] = vec2<f32>(f32(child_idx), t_aabb_child.x);
                            valid_children = valid_children + 1u;
                        }
                    }
                }

                for (var i = 0u; i < valid_children; i = i + 1u) {
                    for (var j = i + 1u; j < valid_children; j = j + 1u) {
                        if (child_data[j].y < child_data[i].y) {
                            let temp = child_data[i];
                            child_data[i] = child_data[j];
                            child_data[j] = temp;
                        }
                    }
                }

                for (var i = 0u; i < valid_children; i = i + 1u) {
                    if (stack_size < NODE_STACK_SIZE) {
                        let idx = valid_children - 1u - i;
                        node_stack[stack_size] = u32(child_data[idx].x);
                        stack_size = stack_size + 1u;
                    }
                }
            }
        }
    }

    return hit;
}

fn trace_blas_any_hit(
    ray_world: ptr<function, Ray>,
    ray_local: ptr<function, Ray>,
    entity_transform: mat4x4f,
    mesh_asset_id: u32,
) -> bool {
    let mesh_directory_entry = atlas_load_directory_entry(mesh_asset_id);
    let bvh4_base = mesh_directory_entry.bvh4_base;
    let first_vertex = mesh_directory_entry.first_vertex;
    let first_index = mesh_directory_entry.first_index;

    var node_stack: array<u32, NODE_STACK_SIZE>;
    node_stack[0] = bvh4_base;
    var stack_size = 1u;

    loop {
        if (stack_size == 0u) { break; }
        stack_size = stack_size - 1u;

        var node_idx = node_stack[stack_size];
        if (node_idx == INVALID_IDX) { continue; }

        let node = atlas_load_bvh4_node(node_idx);
        let t_aabb = intersect_aabb(*ray_local, node.min.xyz, node.max.xyz);

        if (t_aabb.x <= t_aabb.y && t_aabb.x >= ray_local.origin_and_tmin.w && t_aabb.x < ray_local.direction_and_tmax.w) {
            if (stack_size < NODE_STACK_SIZE) {
                let leaf_mask = bitcast<u32>(node.min.w);
                var child_data: array<vec2<f32>, 4>;
                var valid_children = 0u;

                for (var i = 0u; i < 4u; i = i + 1u) {
                    let child_raw = node.children[i];
                    if (child_raw < 0.0) { continue; }

                    let is_leaf_child = ((leaf_mask >> i) & 1u) != 0u;
                    let child_idx = u32(child_raw);

                    if (is_leaf_child) {
                        // BLAS BVH4 leaf nodes store triangle IDs directly - no AABB indirection!
                        let tri_id_local = child_idx;
                        let i0 = index_buffer[first_index + tri_id_local * 3u + 0u];
                        let i1 = index_buffer[first_index + tri_id_local * 3u + 1u];
                        let i2 = index_buffer[first_index + tri_id_local * 3u + 2u];

                        let v0 = vertex_buffer[first_vertex + i0].position.xyz;
                        let v1 = vertex_buffer[first_vertex + i1].position.xyz;
                        let v2 = vertex_buffer[first_vertex + i2].position.xyz;

                        let t_tri = intersect_triangle(*ray_local, v0, v1, v2);
                        if (t_tri >= ray_local.origin_and_tmin.w && t_tri < ray_local.direction_and_tmax.w) {
                            return true;
                        }
                    } else {
                        let child_node = atlas_load_bvh4_node(child_idx);
                        let t_aabb_child = intersect_aabb(*ray_local, child_node.min.xyz, child_node.max.xyz);

                        if (t_aabb_child.x <= t_aabb_child.y && t_aabb_child.x >= ray_local.origin_and_tmin.w && t_aabb_child.x < ray_local.direction_and_tmax.w) {
                            child_data[valid_children] = vec2<f32>(f32(child_idx), t_aabb_child.x);
                            valid_children = valid_children + 1u;
                        }
                    }
                }

                // Near-to-far sorting
                for (var i = 0u; i < valid_children; i = i + 1u) {
                    for (var j = i + 1u; j < valid_children; j = j + 1u) {
                        if (child_data[j].y < child_data[i].y) {
                            let temp = child_data[i];
                            child_data[i] = child_data[j];
                            child_data[j] = temp;
                        }
                    }
                }

                for (var i = 0u; i < valid_children; i = i + 1u) {
                    if (stack_size < NODE_STACK_SIZE) {
                        let idx = valid_children - 1u - i;
                        node_stack[stack_size] = u32(child_data[idx].x);
                        stack_size = stack_size + 1u;
                    }
                }
            }
        }
    }

    return false;
}

fn trace_hit(ray: ptr<function, Ray>) -> RayHit {
    var hit: RayHit;
    hit.position_and_t = vec4f((*ray).origin_and_tmin.xyz, (*ray).direction_and_tmax.w);
    hit.normal_and_user_data = vec4f(0.0, 0.0, 0.0, -1.0);

    var node_stack: array<u32, NODE_STACK_SIZE>;
    node_stack[0] = 0u;
    var stack_size = 1u;

    var current_ray = *ray;

    loop {
        if (stack_size == 0u) { break; }
        stack_size = stack_size - 1u;

        var node_idx = node_stack[stack_size];
        if (node_idx == INVALID_IDX) { continue; }

        let current_node = tlas_bvh4_nodes[node_idx];
        let t_aabb = intersect_aabb(current_ray, current_node.min.xyz, current_node.max.xyz);

        if (t_aabb.x <= t_aabb.y && t_aabb.x >= current_ray.origin_and_tmin.w && t_aabb.x < hit.position_and_t.w) {
            if (stack_size < NODE_STACK_SIZE) {
                let leaf_mask = bitcast<u32>(current_node.min.w);
                var child_data: array<vec2<f32>, 4>;
                var valid_children = 0u;

                for (var i = 0u; i < 4u; i = i + 1u) {
                    let child_raw = current_node.children[i];
                    if (child_raw < 0.0) { continue; }

                    let is_leaf_child = ((leaf_mask >> i) & 1u) != 0u;
                    let child_idx = u32(child_raw);

                    if (is_leaf_child) {
                        let leaf_bounds = tlas_bvh2_bounds[child_idx];
                        let t_leaf = intersect_aabb(current_ray, leaf_bounds.min.xyz, leaf_bounds.max.xyz);

                        // Allow rays that start inside AABBs (t_leaf.x < tmin) by using max(t_leaf.x, tmin)
                        if (t_leaf.x <= t_leaf.y && max(t_leaf.x, current_ray.origin_and_tmin.w) < hit.position_and_t.w) {
                            let prim_store = u32(leaf_bounds.min.w);
                            let mesh_id = mesh_asset_ids[prim_store];
                            var entity_transform = entity_transforms[prim_store];

                            var ray_local = build_local_ray(
                                &current_ray,
                                entity_transform.transform,
                                entity_transform.transpose_inverse_model_matrix
                            );
                            
                            let blas_hit = trace_blas(
                                &current_ray,
                                &ray_local,
                                entity_transform.transform,
                                entity_transform.transpose_inverse_model_matrix,
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
                        let child_node = tlas_bvh4_nodes[child_idx];
                        let t_aabb_child = intersect_aabb(current_ray, child_node.min.xyz, child_node.max.xyz);

                        if (t_aabb_child.x <= t_aabb_child.y && max(t_aabb_child.x, current_ray.origin_and_tmin.w) < hit.position_and_t.w) {
                            child_data[valid_children] = vec2<f32>(f32(child_idx), t_aabb_child.x);
                            valid_children = valid_children + 1u;
                        }
                    }
                }

                for (var i = 0u; i < valid_children; i = i + 1u) {
                    for (var j = i + 1u; j < valid_children; j = j + 1u) {
                        if (child_data[j].y < child_data[i].y) {
                            let temp = child_data[i];
                            child_data[i] = child_data[j];
                            child_data[j] = temp;
                        }
                    }
                }

                for (var i = 0u; i < valid_children; i = i + 1u) {
                    if (stack_size < NODE_STACK_SIZE) {
                        let idx = valid_children - 1u - i;
                        node_stack[stack_size] = u32(child_data[idx].x);
                        stack_size = stack_size + 1u;
                    }
                }
            }
        }
    }

    return hit;
}

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

        let current_node = tlas_bvh4_nodes[node_idx];
        let t_aabb = intersect_aabb(current_ray, current_node.min.xyz, current_node.max.xyz);

        if (t_aabb.x <= t_aabb.y && t_aabb.x >= current_ray.origin_and_tmin.w && t_aabb.x < current_ray.direction_and_tmax.w) {
            if (stack_size < NODE_STACK_SIZE) {
                let leaf_mask = bitcast<u32>(current_node.min.w);
                var child_data: array<vec2<f32>, 4>;
                var valid_children = 0u;

                for (var i = 0u; i < 4u; i = i + 1u) {
                    let child_raw = current_node.children[i];
                    if (child_raw < 0.0) { continue; }

                    let is_leaf_child = ((leaf_mask >> i) & 1u) != 0u;
                    let child_idx = u32(child_raw);

                    if (is_leaf_child) {
                        let leaf_bounds = tlas_bvh2_bounds[child_idx];
                        let t_leaf = intersect_aabb(current_ray, leaf_bounds.min.xyz, leaf_bounds.max.xyz);

                        if (t_leaf.x <= t_leaf.y && max(t_leaf.x, current_ray.origin_and_tmin.w) < current_ray.direction_and_tmax.w) {
                            let prim_store = u32(leaf_bounds.min.w);
                            let mesh_id = mesh_asset_ids[prim_store];
                            let entity_transform = entity_transforms[prim_store];

                            var ray_local = build_local_ray(
                                &current_ray,
                                entity_transform.transform,
                                entity_transform.transpose_inverse_model_matrix
                            );
                            
                            // Wave-optimized shadow ray test (massive win here!)
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
                        let child_node = tlas_bvh4_nodes[child_idx];
                        let t_aabb_child = intersect_aabb(current_ray, child_node.min.xyz, child_node.max.xyz);

                        if (t_aabb_child.x <= t_aabb_child.y && max(t_aabb_child.x, current_ray.origin_and_tmin.w) < current_ray.direction_and_tmax.w) {
                            child_data[valid_children] = vec2<f32>(f32(child_idx), t_aabb_child.x);
                            valid_children = valid_children + 1u;
                        }
                    }
                }

                for (var i = 0u; i < valid_children; i = i + 1u) {
                    for (var j = i + 1u; j < valid_children; j = j + 1u) {
                        if (child_data[j].y < child_data[i].y) {
                            let temp = child_data[i];
                            child_data[i] = child_data[j];
                            child_data[j] = temp;
                        }
                    }
                }

                for (var i = 0u; i < valid_children; i = i + 1u) {
                    if (stack_size < NODE_STACK_SIZE) {
                        let idx = valid_children - 1u - i;
                        node_stack[stack_size] = u32(child_data[idx].x);
                        stack_size = stack_size + 1u;
                    }
                }
            }
        }
    }

    return false;
}


@compute @workgroup_size(64, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let probe_count = screen_probe_counter[0];
    let rays_per_probe = gi_params.screen_ray_count;
    let total_rays = probe_count * rays_per_probe;
    
    if (gid.x >= total_rays) {
        return;
    }
    
    var ps = probe_path_state[gid.x];
    
    // Check if path is alive
    if (ps.state_u32.y == 0u) {
        return;
    }
    
    // Build ray structure
    var ray: Ray;
    ray.origin_and_tmin = ps.origin_tmin;
    ray.direction_and_tmax = ps.direction_tmax;
    let d = ray.direction_and_tmax.xyz;
    ray.inv_direction = vec4f(
        1.0 / max(abs(d.x), 1e-8) * select(1.0, -1.0, d.x < 0.0),
        1.0 / max(abs(d.y), 1e-8) * select(1.0, -1.0, d.y < 0.0),
        1.0 / max(abs(d.z), 1e-8) * select(1.0, -1.0, d.z < 0.0),
        0.0
    );
    
    // Trace ray using existing traversal logic
    let hit_result = trace_hit(&ray);
    
    // Check for miss
    if (hit_result.normal_and_user_data.w < 0.0) {
        ps.state_u32.w = 0xffffffffu;
        probe_path_state[gid.x] = ps;
        return;
    }
    
    // Extract hit information
    let tri_id_local = u32(hit_result.normal_and_user_data.w);
    let prim_store = u32(hit_result.prim_meshid_padding.x);
    let mesh_id = u32(hit_result.prim_meshid_padding.y);
    let entity_transform = entity_transforms[prim_store];
    
    // Compute world-space hit position
    let t_tri = hit_result.position_and_t.w;
    let p_local = hit_result.ray_local.origin_and_tmin.xyz + hit_result.ray_local.direction_and_tmax.xyz * t_tri;
    let p_world = (entity_transform.transform * vec4f(p_local, 1.0)).xyz;
    
    // Fetch triangle data
    let mesh_entry = atlas_load_directory_entry(mesh_id);
    let first_vertex = mesh_entry.first_vertex;
    let first_index = mesh_entry.first_index;
    
    let i0 = index_buffer[first_index + tri_id_local * 3u + 0u];
    let i1 = index_buffer[first_index + tri_id_local * 3u + 1u];
    let i2 = index_buffer[first_index + tri_id_local * 3u + 2u];
    let vertex0 = vertex_buffer[first_vertex + i0];
    let vertex1 = vertex_buffer[first_vertex + i1];
    let vertex2 = vertex_buffer[first_vertex + i2];
    
    // Get vertex data
    let v0 = vertex0.position.xyz;
    let v1 = vertex1.position.xyz;
    let v2 = vertex2.position.xyz;
    let uv0 = vertex0.uv.xy;
    let uv1 = vertex1.uv.xy;
    let uv2 = vertex2.uv.xy;
    let n0 = vertex0.normal.xyz;
    let n1 = vertex1.normal.xyz;
    let n2 = vertex2.normal.xyz;
    let t0 = vertex0.tangent.xyz;
    let t1 = vertex1.tangent.xyz;
    let t2 = vertex2.tangent.xyz;
    let b0 = vertex0.bitangent.xyz;
    let b1 = vertex1.bitangent.xyz;
    let b2 = vertex2.bitangent.xyz;
    
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
    
    // Interpolate attributes
    let uv_hit = uv0 * (1.0 - u_bc - v_bc) + uv1 * u_bc + uv2 * v_bc;
    let t_local = (t0 * (1.0 - u_bc - v_bc) + t1 * u_bc + t2 * v_bc);
    let b_local = (b0 * (1.0 - u_bc - v_bc) + b1 * u_bc + b2 * v_bc);
    let n_local = (n0 * (1.0 - u_bc - v_bc) + n1 * u_bc + n2 * v_bc);
    
    // Transform to world space
    var world_n = safe_normalize((entity_transform.transpose_inverse_model_matrix * vec4<f32>(n_local, 0.0)).xyz);
    var world_t = safe_normalize((entity_transform.transform * vec4<f32>(t_local, 0.0)).xyz);
    var world_b = safe_normalize((entity_transform.transform * vec4<f32>(b_local, 0.0)).xyz);
    
    // Flip normal if needed
    let ray_dir = ray.direction_and_tmax.xyz;
    if (dot(world_n, ray_dir) > 0.0) {
        world_n = -world_n;
        world_t = -world_t;
        world_b = -world_b;
    }
    
    // Update path state
    ps.origin_tmin = vec4f(p_world, t_tri);
    ps.direction_tmax = vec4f(ray_dir, hit_result.prim_meshid_padding.x);
    ps.normal_section_index = vec4f(world_n, f32(vertex0.section_index));
    ps.hit_attr0 = vec4f(world_t, uv_hit.x);
    ps.hit_attr1 = vec4f(world_b, uv_hit.y);
    ps.state_u32.w = tri_id_local;
    
    probe_path_state[gid.x] = ps;
}

