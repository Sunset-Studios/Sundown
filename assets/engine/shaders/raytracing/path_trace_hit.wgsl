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

const NODE_STACK_SIZE = 12;

struct PathTracerParams {
    max_bounces: u32,
    reset_accum_flag: u32,
    use_gbuffer: u32,
    trace_rate: u32,      // 1=full res, 2=half res, 4=quarter res, etc.
    frame_phase: u32,     // cycles 0 to trace_rate-1
    indirect_boost: u32,          // Multiplier for indirect bounces
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
    mesh_asset_id: u32
) -> RayHit {
    var hit: RayHit;
    hit.position_and_t = vec4f((*ray_world).origin_and_tmin.xyz, (*ray_local).direction_and_tmax.w);
    hit.normal_and_user_data = vec4f(0.0, 0.0, 0.0, -1.0);

    var current_ray = *ray_local;

    var node_stack: array<u32, NODE_STACK_SIZE>;
    node_stack[0] = atlas_load_directory_entry_bvh4_base(mesh_asset_id);
    var stack_size = 1u;

    while (stack_size > 0u) {
        stack_size = stack_size - 1u;

        let node_idx = node_stack[stack_size];
        if (node_idx == INVALID_IDX) { continue; }

        let leaf_mask = atlas_load_bvh4_leaf_mask(node_idx);
        let children = atlas_load_bvh4_node_children(node_idx);
        
        // Node already tested before push - no redundant AABB test here!
        for (var i = 0u; i < 4u; i = i + 1u) {
            if (children[i] < 0.0) { continue; }

            let child_idx = u32(children[i]);

            if (((leaf_mask >> i) & 1u) != 0u) { // Is leaf?
                // Load vertex indices from co-located leaf data (cache-adjacent to node!)
                let leaf_indices = atlas_load_bvh4_leaf_indices(node_idx, i);
                let t_tri = intersect_triangle(
                    &current_ray,
                    vertex_buffer[leaf_indices.x].position.xyz,
                    vertex_buffer[leaf_indices.y].position.xyz,
                    vertex_buffer[leaf_indices.z].position.xyz
                );
                let is_better_hit = t_tri >= current_ray.origin_and_tmin.w && t_tri < current_ray.direction_and_tmax.w;
                hit.position_and_t.w = select(hit.position_and_t.w, t_tri, is_better_hit);
                hit.normal_and_user_data.w = select(hit.normal_and_user_data.w, f32(child_idx), is_better_hit);
                hit.hit_triangle_data = select(hit.hit_triangle_data, leaf_indices, is_better_hit);
                current_ray.direction_and_tmax.w = select(current_ray.direction_and_tmax.w, t_tri, is_better_hit);
            } else {
                // Only test AABB before pushing - guarantees single test per node
                let t_aabb_child = intersect_aabb(
                    &current_ray,
                    atlas_load_bvh4_node_min(child_idx),
                    atlas_load_bvh4_node_max(child_idx)
                );

                let is_better_child = t_aabb_child.x <= t_aabb_child.y
                    && t_aabb_child.x >= current_ray.origin_and_tmin.w
                    && t_aabb_child.x < current_ray.direction_and_tmax.w;
                node_stack[stack_size] = select(node_stack[stack_size], child_idx, is_better_child);
                stack_size = select(stack_size, stack_size + 1u, is_better_child);
            }
        }
    }

    return hit;
}

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

        let leaf_mask = bitcast<u32>(tlas_bvh4_nodes[node_idx].min.w);

        for (var i = 0u; i < 4u; i = i + 1u) {
            if (tlas_bvh4_nodes[node_idx].children[i] < 0.0) { continue; }

            let child_idx = u32(tlas_bvh4_nodes[node_idx].children[i]);

            if (((leaf_mask >> i) & 1u) != 0u) { // Is leaf?
                let t_leaf = intersect_aabb(
                    &current_ray,
                    tlas_bvh2_bounds[child_idx].min.xyz,
                    tlas_bvh2_bounds[child_idx].max.xyz
                );

                // Allow rays that start inside AABBs (t_leaf.x < tmin) by using max(t_leaf.x, tmin)
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
                // Only test AABB before pushing - guarantees single test per node
                let t_aabb_child = intersect_aabb(
                    &current_ray,
                    tlas_bvh4_nodes[child_idx].min.xyz,
                    tlas_bvh4_nodes[child_idx].max.xyz
                );
                let is_better_child = t_aabb_child.x <= t_aabb_child.y
                    && t_aabb_child.x >= current_ray.origin_and_tmin.w
                    && t_aabb_child.x < hit.position_and_t.w;

                node_stack[stack_size] =  select(node_stack[stack_size], child_idx, is_better_child);
                stack_size = select(stack_size, stack_size + 1u, is_better_child);
            }
        }
    }

    return hit;
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
    let pixel_coords = compute_phased_pixel_coords(gid.x, res, pt_params.trace_rate, pt_params.frame_phase);
    let pixel_index = pixel_coords.y * res.x + pixel_coords.x;

    var ray: Ray;

    // Is ray alive?
    if (path_state[pixel_index].state_u32.y != 0u) {
        ray.origin_and_tmin = path_state[pixel_index].origin_tmin;
        ray.direction_and_tmax = path_state[pixel_index].direction_tmax;
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

            let entity_transform = entity_transforms[prim_store];

            let t_tri = hit_result.position_and_t.w;
            let p_local = hit_result.ray_local.origin_and_tmin.xyz + hit_result.ray_local.direction_and_tmax.xyz * t_tri;
            let p_world = (entity_transform.transform * vec4f(p_local, 1.0)).xyz;

            let v0i = hit_result.hit_triangle_data.x;
            let v1i = hit_result.hit_triangle_data.y;
            let v2i = hit_result.hit_triangle_data.z;
            
            // Load positions only for barycentric calculation
            let v0 = vertex_buffer[v0i].position.xyz;
            let v1 = vertex_buffer[v1i].position.xyz;
            let v2 = vertex_buffer[v2i].position.xyz;

            // Compute barycentric coordinates immediately
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

            // Load and interpolate UVs immediately (reduce live ranges)
            let uv_hit = vertex_buffer[v0i].uv.xy * w_bc + 
                         vertex_buffer[v1i].uv.xy * u_bc + 
                         vertex_buffer[v2i].uv.xy * v_bc;

            // Load and interpolate normals, transform immediately
            let n_local = vertex_buffer[v0i].normal.xyz * w_bc + 
                          vertex_buffer[v1i].normal.xyz * u_bc + 
                          vertex_buffer[v2i].normal.xyz * v_bc;
            var world_n = safe_normalize((entity_transform.transpose_inverse_model_matrix * vec4<f32>(n_local, 0.0)).xyz);

            // Load and interpolate tangents, transform immediately
            let t_local = vertex_buffer[v0i].tangent.xyz * w_bc + 
                          vertex_buffer[v1i].tangent.xyz * u_bc + 
                          vertex_buffer[v2i].tangent.xyz * v_bc;
            var world_t = safe_normalize((entity_transform.transform * vec4<f32>(t_local, 0.0)).xyz);

            // Load and interpolate bitangents, transform immediately
            let b_local = vertex_buffer[v0i].bitangent.xyz * w_bc + 
                          vertex_buffer[v1i].bitangent.xyz * u_bc + 
                          vertex_buffer[v2i].bitangent.xyz * v_bc;
            var world_b = safe_normalize((entity_transform.transform * vec4<f32>(b_local, 0.0)).xyz);
            
            // Get section_index from first vertex only
            let section_idx = vertex_buffer[v0i].section_index;

            let ray_dir = ray.direction_and_tmax.xyz;
            let ray_is_backfacing = dot(world_n, ray_dir) > 0.0;
            world_n = select(world_n, -world_n, ray_is_backfacing);
            world_t = select(world_t, -world_t, ray_is_backfacing);
            world_b = select(world_b, -world_b, ray_is_backfacing);

            // Store hit distance in origin_tmin.w for use in shade pass (for emissive attenuation)
            // This will be overwritten when we spawn the next ray, but shade pass reads it first
            path_state[pixel_index].origin_tmin = vec4f(p_world, t_tri);
            path_state[pixel_index].direction_tmax = vec4f(ray_dir, hit_result.prim_meshid_padding.x);
            path_state[pixel_index].normal_section_index = vec4f(world_n, f32(section_idx));
            path_state[pixel_index].hit_attr0 = vec4f(world_t, uv_hit.x);
            path_state[pixel_index].hit_attr1 = vec4f(world_b, uv_hit.y);
            path_state[pixel_index].state_u32.w = tri_id_local;
        }
    }
}

