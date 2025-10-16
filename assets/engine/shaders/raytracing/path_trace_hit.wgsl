// =============================================================================
// Path Tracer - Hit Pass (Minimal Wave Optimizations)
// - ONLY adds wave-level early exits (highest ROI)
// - Keeps 8x8 workgroup for cache locality
// - No broadcasting overhead unless proven beneficial
// =============================================================================
diagnostic(off,subgroup_uniformity);

#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "blas_common.wgsl"

const NODE_STACK_SIZE = 16;

struct PathTracerParams {
    max_bounces: u32,
    spp_per_frame: u32,
    reset_accum_flag: u32,
    use_gbuffer: u32,
    trace_rate: u32,      // 1=full res, 2=half res, 4=quarter res, etc.
    frame_phase: u32,     // cycles 0 to trace_rate-1
    indirect_boost: u32,          // Multiplier for indirect bounces
    padding: u32,
};

struct PathState {
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

@group(1) @binding(0) var<uniform> pt_params: PathTracerParams;
@group(1) @binding(1) var<storage, read_write> path_state: array<PathState>;
@group(1) @binding(2) var<storage, read> tlas_bvh2_bounds: array<AABB>;
@group(1) @binding(3) var<storage, read> tlas_bvh4_nodes: array<BVH4Node>;
@group(1) @binding(4) var<storage, read> blas_atlas: BLASAtlas;
@group(1) @binding(5) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(6) var<storage, read> index_buffer: array<u32>;
@group(1) @binding(7) var<storage, read> mesh_asset_ids: array<u32>;
@group(1) @binding(8) var output_tex: texture_storage_2d<rgba16float, write>;

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
                        let leaf_bounds = atlas_load_aabb(child_idx);
                        let t_leaf = intersect_aabb(current_ray, leaf_bounds.min.xyz, leaf_bounds.max.xyz);
                        if (t_leaf.x <= t_leaf.y && t_leaf.x >= current_ray.origin_and_tmin.w && t_leaf.x < hit.position_and_t.w) {
                            let tri_id_local = u32(leaf_bounds.min.w);
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
                        let leaf_bounds = atlas_load_aabb(child_idx);
                        let t_leaf = intersect_aabb(*ray_local, leaf_bounds.min.xyz, leaf_bounds.max.xyz);
                        if (t_leaf.x <= t_leaf.y && t_leaf.x >= ray_local.origin_and_tmin.w && t_leaf.x < ray_local.direction_and_tmax.w) {
                            let tri_id_local = u32(leaf_bounds.min.w);
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


// Helper function to compute the Nth pixel that matches the frame_phase pattern
// Optimized: ~3-5 iterations max, independent of resolution
fn compute_pixel_coords(linear_index: u32, res: vec2<u32>, trace_rate: u32, frame_phase: u32) -> vec2<u32> {
    if (trace_rate <= 1u) {
        return vec2<u32>(linear_index % res.x, linear_index / res.x);
    }
    
    // Estimate which row the pixel is in
    // Most rows have approx res.x / trace_rate pixels
    let avg_pixels_per_row = res.x / trace_rate;
    let estimated_row = linear_index / max(avg_pixels_per_row, 1u);
    
    // Search a small window around the estimate (max ~5 iterations)
    let search_start = select(0u, estimated_row - 1u, estimated_row >= 1u);
    let search_end = min(estimated_row + 4u, res.y);
    
    // Estimate cumulative pixels before search_start
    var cumulative_pixels = search_start * avg_pixels_per_row;
    
    for (var y = search_start; y < search_end; y = y + 1u) {
        let first_x = (frame_phase + trace_rate - (y * 2u) % trace_rate) % trace_rate;
        let pixels_in_row = (res.x + trace_rate - 1u - first_x) / trace_rate;
        
        if (linear_index < cumulative_pixels + pixels_in_row) {
            let offset_in_row = linear_index - cumulative_pixels;
            let x = first_x + offset_in_row * trace_rate;
            return vec2<u32>(x, y);
        }
        
        cumulative_pixels += pixels_in_row;
    }
    
    return vec2<u32>(0xFFFFFFFFu, 0xFFFFFFFFu);
}

@compute @workgroup_size(128, 1, 1)
fn cs(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(subgroup_invocation_id) lane_id: u32,
    @builtin(subgroup_size) warp_size: u32
) {
    let res = textureDimensions(output_tex);
    
    // Compute actual pixel coordinates based on linear thread index and trace pattern
    let pixel_coords = compute_pixel_coords(gid.x, res, pt_params.trace_rate, pt_params.frame_phase);
    
    // Early exit if we're out of bounds
    if (pixel_coords.x >= res.x || pixel_coords.y >= res.y) { return; }

    let pixel_index = pixel_coords.y * res.x + pixel_coords.x;
    var ps = path_state[pixel_index];

    // Early exit for G-buffer mode on first bounce
    if (pt_params.use_gbuffer != 0u && ps.state_u32.x == 0u && ps.state_u32.w == 0x0u) {
        return;
    }

    var ray: Ray;

    // Shadow ray processing
    if (ps.shadow_radiance.a > 0.0) {
        ray.origin_and_tmin = ps.shadow_origin;
        ray.direction_and_tmax = ps.shadow_direction;
        let d = ray.direction_and_tmax.xyz;
        ray.inv_direction = vec4f(
            1.0 / max(abs(d.x), 1e-8) * select(1.0, -1.0, d.x < 0.0),
            1.0 / max(abs(d.y), 1e-8) * select(1.0, -1.0, d.y < 0.0),
            1.0 / max(abs(d.z), 1e-8) * select(1.0, -1.0, d.z < 0.0),
            0.0
        );
        
        // Wave-optimized shadow ray test (massive win here!)
        if (!trace_hit_any(&ray)) {
            ps.state_u32.z = 1u; // No shadow hit
        }
    }

    // Is ray alive?
    if (ps.state_u32.y != 0u) {
        ray.origin_and_tmin = ps.origin_tmin;
        ray.direction_and_tmax = ps.direction_tmax;
        let d = ray.direction_and_tmax.xyz;
        ray.inv_direction = vec4f(
            1.0 / max(abs(d.x), 1e-8) * select(1.0, -1.0, d.x < 0.0),
            1.0 / max(abs(d.y), 1e-8) * select(1.0, -1.0, d.y < 0.0),
            1.0 / max(abs(d.z), 1e-8) * select(1.0, -1.0, d.z < 0.0),
            0.0
        );

        let hit_result = trace_hit(&ray);
        
        if (hit_result.normal_and_user_data.w >= 0.0) {
            let tri_id_local = u32(hit_result.normal_and_user_data.w);
            let prim_store = u32(hit_result.prim_meshid_padding.x);
            let mesh_id = u32(hit_result.prim_meshid_padding.y);

            let entity_transform = entity_transforms[prim_store];

            let t_tri = hit_result.position_and_t.w;
            let p_local = hit_result.ray_local.origin_and_tmin.xyz + hit_result.ray_local.direction_and_tmax.xyz * t_tri;
            let p_world = (entity_transform.transform * vec4f(p_local, 1.0)).xyz;

            let mesh_entry = atlas_load_directory_entry(mesh_id);
            let first_vertex = mesh_entry.first_vertex;
            let first_index = mesh_entry.first_index;

            let i0 = index_buffer[first_index + tri_id_local * 3u + 0u];
            let i1 = index_buffer[first_index + tri_id_local * 3u + 1u];
            let i2 = index_buffer[first_index + tri_id_local * 3u + 2u];
            let vertex0 = vertex_buffer[first_vertex + i0];
            let vertex1 = vertex_buffer[first_vertex + i1];
            let vertex2 = vertex_buffer[first_vertex + i2];

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

            let uv_hit = uv0 * (1.0 - u_bc - v_bc) + uv1 * u_bc + uv2 * v_bc;
            let t_local = (t0 * (1.0 - u_bc - v_bc) + t1 * u_bc + t2 * v_bc);
            let b_local = (b0 * (1.0 - u_bc - v_bc) + b1 * u_bc + b2 * v_bc);
            let n_local = (n0 * (1.0 - u_bc - v_bc) + n1 * u_bc + n2 * v_bc);

            var world_n = safe_normalize((entity_transform.transpose_inverse_model_matrix * vec4<f32>(n_local, 0.0)).xyz);
            var world_t = safe_normalize((entity_transform.transform * vec4<f32>(t_local, 0.0)).xyz);
            var world_b = safe_normalize((entity_transform.transform * vec4<f32>(b_local, 0.0)).xyz);

            let ray_dir = ray.direction_and_tmax.xyz;
            if (dot(world_n, ray_dir) > 0.0) {
                world_n = -world_n;
                world_t = -world_t;
                world_b = -world_b;
            }

            // Store hit distance in origin_tmin.w for use in shade pass (for emissive attenuation)
            // This will be overwritten when we spawn the next ray, but shade pass reads it first
            ps.origin_tmin = vec4f(p_world, t_tri);
            ps.direction_tmax = vec4f(ray_dir, hit_result.prim_meshid_padding.x);
            ps.normal_section_index = vec4f(world_n, f32(vertex0.section_index));
            ps.hit_attr0 = vec4f(world_t, uv_hit.x);
            ps.hit_attr1 = vec4f(world_b, uv_hit.y);
            ps.state_u32.w = tri_id_local;
        }
    }

    path_state[pixel_index] = ps;
}

