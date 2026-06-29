// =============================================================================
// GI-1.0 World Cache Ray Tracing - Primary Hit Pass
// - Traces primary rays from active world cache cells against BVH
// - Writes hit information to path state
// - Uses optimized traversal consistent with probe tracing
// =============================================================================
#define RAY_TRAVERSAL_USE_RAY_INSTANCE_TRANSFORMS

#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "gi/gi_common.wgsl"
#include "acceleration/bvh_ray_traversal.wgsl"

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read_write> world_cache_path_state: array<WorldCachePathState>;
@group(1) @binding(2) var<storage, read> tlas_bvh2_bounds: array<AABB>;
@group(1) @binding(3) var<uniform> tlas_bvh_info: BVHInfo;
@group(1) @binding(4) var<storage, read> blas_bvh2_nodes: array<AABB>;
@group(1) @binding(5) var<storage, read> blas_directory: array<MeshDirectoryEntry>;
@group(1) @binding(6) var<storage, read> ray_instance_transforms: array<RayInstanceTransform>;
@group(1) @binding(7) var<storage, read> index_buffer: array<u32>;
@group(1) @binding(8) var<storage, read> gi_counters: GICountersReadOnly;
@group(1) @binding(9) var<storage, read> entity_index_lookup: array<u32>;


// =============================================================================
// HELPER: Process a primary ray and write hit attributes
// =============================================================================

fn process_primary_ray(active_index: u32) {
    var ray: Ray;
    ray.origin_and_tmin = world_cache_path_state[active_index].origin_tmin;
    ray.direction_and_tmax = world_cache_path_state[active_index].direction_tmax;
    ray.inv_direction = vec4<f32>(
        1.0 / max(abs(ray.direction_and_tmax.x), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.x < 0.0),
        1.0 / max(abs(ray.direction_and_tmax.y), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.y < 0.0),
        1.0 / max(abs(ray.direction_and_tmax.z), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.z < 0.0),
        0.0
    );

    let hit_result = trace_ray_closest(&ray);

    if (hit_result.has_hit != 0u) {
        let tri_id_local = hit_result.tri_id_local;
        let prim_store = hit_result.prim_store;
        let entity_resolved = entity_index_lookup[prim_store];

        let instance_transform = ray_instance_transforms[entity_resolved];

        var ray_local = build_local_ray_from_instance(
            &ray,
            instance_transform
        );

        let t_tri = hit_result.t_hit;
        let p_local = ray_local.origin_and_tmin.xyz + ray_local.direction_and_tmax.xyz * t_tri;
        let p_world = transform_local_point_from_instance(instance_transform, p_local);

        let v0i = hit_result.tri_indices.x;
        let v1i = hit_result.tri_indices.y;
        let v2i = hit_result.tri_indices.z;
        
        // Load positions only for barycentric calculation
        let vertex0 = decode_vertex(vertex_buffer[v0i]);
        let vertex1 = decode_vertex(vertex_buffer[v1i]);
        let vertex2 = decode_vertex(vertex_buffer[v2i]);
        let v0 = vertex0.position.xyz;
        let v1 = vertex1.position.xyz;
        let v2 = vertex2.position.xyz;

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
        let uv_hit = vertex0.uv * w_bc + 
                        vertex1.uv * u_bc + 
                        vertex2.uv * v_bc;

        // Load and interpolate normals, transform immediately
        let n_local = vertex0.normal.xyz * w_bc + 
                        vertex1.normal.xyz * u_bc + 
                        vertex2.normal.xyz * v_bc;
        var world_n = safe_normalize(transform_local_direction_from_instance(instance_transform, n_local));

        // Load and interpolate tangents, transform immediately
        let t_local = vertex0.tangent.xyz * w_bc + 
                        vertex1.tangent.xyz * u_bc + 
                        vertex2.tangent.xyz * v_bc;
        var world_t = safe_normalize(transform_local_direction_from_instance(instance_transform, t_local));

        // Load and interpolate bitangents, transform immediately
        let b_local = vertex0.bitangent.xyz * w_bc + 
                        vertex1.bitangent.xyz * u_bc + 
                        vertex2.bitangent.xyz * v_bc;
        var world_b = safe_normalize(transform_local_direction_from_instance(instance_transform, b_local));
        
        let ray_dir = ray.direction_and_tmax.xyz;
        let ray_is_backfacing = dot(world_n, ray_dir) > 0.0;
        world_n = select(world_n, -world_n, ray_is_backfacing);
        world_t = select(world_t, -world_t, ray_is_backfacing);
        world_b = select(world_b, -world_b, ray_is_backfacing);

        // Store hit distance in origin_tmin.w for use in shade pass (for emissive attenuation)
        world_cache_path_state[active_index].origin_tmin = vec4<f32>(p_world, t_tri);
        world_cache_path_state[active_index].direction_tmax = vec4<f32>(ray_dir, f32(prim_store));
        world_cache_path_state[active_index].normal_section_index = vec4<f32>(world_n, vertex0.section_index);
        world_cache_path_state[active_index].hit_attr0 = vec4<f32>(world_t, uv_hit.x);
        world_cache_path_state[active_index].hit_attr1 = vec4<f32>(world_b, uv_hit.y);
        world_cache_path_state[active_index].state_u32.w = tri_id_local;
    }
}

@compute @workgroup_size(128, 1, 1)
fn cs(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(subgroup_invocation_id) lane_id: u32,
    @builtin(subgroup_size) warp_size: u32
) {
    // Thread ID maps to index in compacted active cell array
    let active_index = gid.x;
    let active_cache_cell_count = gi_counters.active_cache_cell_count;
    if (active_index >= active_cache_cell_count) {
        return;
    }

    // Is ray alive?
    if (world_cache_path_state[active_index].state_u32.y != 0u) {
        process_primary_ray(active_index);
    }
}
