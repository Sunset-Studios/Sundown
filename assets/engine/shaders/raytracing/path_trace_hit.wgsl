// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║               PATH TRACER - COMBINED HIT PASS                             ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Traces both shadow rays and bounce rays in a single pass:               ║
// ║  • Shadow ray any-hit queries for Next Event Estimation (NEE)            ║
// ║  • TLAS (Top-Level) traversal for instance culling                       ║
// ║  • BLAS (Bottom-Level) traversal for triangle intersection               ║
// ║  • Outputs hit attributes for shading pass                               ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================
#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "acceleration/bvh_ray_traversal.wgsl"

// ─────────────────────────────────────────────────────────────────────────────
// Structures
// ─────────────────────────────────────────────────────────────────────────────
struct PathTracerParams {
    max_bounces: u32,
    reset_accum_flag: u32,
    use_gbuffer: u32,
    trace_rate: u32,
    frame_phase: u32,
    samples_per_pixel: u32,
    sample_index: u32,
    sampling_tile_width: u32,
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
    path_weight: vec4<f32>,
    rng_sample_count: vec4<f32>,
    accumulated_radiance: vec4<f32>,
    primary_albedo: vec4<f32>,
};

// ─────────────────────────────────────────────────────────────────────────────
// Bindings
// ─────────────────────────────────────────────────────────────────────────────
@group(1) @binding(0) var<uniform> pt_params: PathTracerParams;
@group(1) @binding(1) var<storage, read_write> path_state: array<PathState>;
@group(1) @binding(2) var<storage, read> tlas_bvh2_bounds: array<AABB>;
@group(1) @binding(3) var<uniform> tlas_bvh_info: BVHInfo;
@group(1) @binding(4) var<storage, read> blas_bvh2_nodes: array<AABB>;
@group(1) @binding(5) var<storage, read> blas_directory: array<MeshDirectoryEntry>;
@group(1) @binding(6) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(7) var<storage, read> index_buffer: array<u32>;
@group(1) @binding(8) var<storage, read> entity_index_lookup: array<u32>;
@group(1) @binding(9) var output_tex: texture_storage_2d<rgba16float, write>;

// =============================================================================
// MAIN COMPUTE SHADER
// =============================================================================

@compute @workgroup_size(128, 1, 1)
fn cs(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(subgroup_invocation_id) lane_id: u32,
    @builtin(subgroup_size) warp_size: u32
) {
    let res = textureDimensions(output_tex);
    let pixel_coords = compute_phased_pixel_coords(
        gid.x,
        res,
        pt_params.trace_rate,
        pt_params.frame_phase,
        pt_params.sampling_tile_width
    );
    
    if (pixel_coords.x >= res.x || pixel_coords.y >= res.y) { return; }
    
    let pixel_index = pixel_coords.y * res.x + pixel_coords.x;

    var ray: Ray;

    // Only process if path is alive
    if (path_state[pixel_index].state_u32.y != 0u) {
        // ─────────────────────────────────────────────────────────────────────
        // Shadow Ray Trace (NEE visibility) - only if shadow ray is pending
        // ─────────────────────────────────────────────────────────────────────
        if (path_state[pixel_index].shadow_radiance.a > 0.0) {
            ray.origin_and_tmin = path_state[pixel_index].shadow_origin;
            ray.direction_and_tmax = path_state[pixel_index].shadow_direction;
            ray.inv_direction = vec4<f32>(
                1.0 / max(abs(ray.direction_and_tmax.x), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.x < 0.0),
                1.0 / max(abs(ray.direction_and_tmax.y), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.y < 0.0),
                1.0 / max(abs(ray.direction_and_tmax.z), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.z < 0.0),
                0.0
            );
            
            let has_hit = trace_ray_any(&ray);
            path_state[pixel_index].state_u32.z = select(1u, 0u, has_hit);
        }

        // ─────────────────────────────────────────────────────────────────────
        // Primary/Bounce Ray Trace
        // ─────────────────────────────────────────────────────────────────────
        ray.origin_and_tmin = path_state[pixel_index].origin_tmin;
        ray.direction_and_tmax = path_state[pixel_index].direction_tmax;
        ray.inv_direction = vec4<f32>(
            1.0 / max(abs(ray.direction_and_tmax.x), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.x < 0.0),
            1.0 / max(abs(ray.direction_and_tmax.y), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.y < 0.0),
            1.0 / max(abs(ray.direction_and_tmax.z), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.z < 0.0),
            0.0
        );

        let hit_result = trace_ray_closest(&ray);
        
        if (hit_result.has_hit != 0u) {
            // ─────────────────────────────────────────────────────────────────
            // Process hit: compute barycentric coords and interpolate attributes
            // ─────────────────────────────────────────────────────────────────
            let tri_id_local = hit_result.tri_id_local;
            let prim_store = hit_result.prim_store;
            let entity_resolved = entity_index_lookup[prim_store];

            let entity_transform = entity_transforms[entity_resolved];

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
            let vertex0 = decode_vertex(vertex_buffer[v0i]);
            let vertex1 = decode_vertex(vertex_buffer[v1i]);
            let vertex2 = decode_vertex(vertex_buffer[v2i]);
            let v0 = vertex0.position.xyz;
            let v1 = vertex1.position.xyz;
            let v2 = vertex2.position.xyz;

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
            let uv_hit = vertex0.uv * w_bc + 
                         vertex1.uv * u_bc + 
                         vertex2.uv * v_bc;

            // Interpolate and transform normals
            let n_local = vertex0.normal.xyz * w_bc + 
                          vertex1.normal.xyz * u_bc + 
                          vertex2.normal.xyz * v_bc;
            var world_n = safe_normalize(
                (entity_transform.transpose_inverse_model_matrix * vec4<f32>(n_local, 0.0)).xyz
            );

            // Interpolate and transform tangents
            let t_local = vertex0.tangent.xyz * w_bc + 
                          vertex1.tangent.xyz * u_bc + 
                          vertex2.tangent.xyz * v_bc;
            var world_t = safe_normalize(
                (entity_transform.transpose_inverse_model_matrix * vec4<f32>(t_local, 0.0)).xyz
            );

            // Interpolate and transform bitangents
            let b_local = vertex0.bitangent.xyz * w_bc + 
                          vertex1.bitangent.xyz * u_bc + 
                          vertex2.bitangent.xyz * v_bc;
            var world_b = safe_normalize(
                (entity_transform.transpose_inverse_model_matrix * vec4<f32>(b_local, 0.0)).xyz
            );
            
            // Handle backfacing geometry
            let ray_dir = ray.direction_and_tmax.xyz;
            let ray_is_backfacing = dot(world_n, ray_dir) > 0.0;
            world_n = select(world_n, -world_n, ray_is_backfacing);
            world_t = select(world_t, -world_t, ray_is_backfacing);
            world_b = select(world_b, -world_b, ray_is_backfacing);

            // Store hit information
            path_state[pixel_index].origin_tmin = vec4<f32>(p_world, t_tri);
            path_state[pixel_index].direction_tmax = vec4<f32>(ray_dir, f32(prim_store));
            path_state[pixel_index].normal_section_index = vec4<f32>(world_n, vertex0.section_index);
            path_state[pixel_index].hit_attr0 = vec4<f32>(world_t, uv_hit.x);
            path_state[pixel_index].hit_attr1 = vec4<f32>(world_b, uv_hit.y);
            path_state[pixel_index].state_u32.w = tri_id_local;
        }
    }
}
