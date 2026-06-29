#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "acceleration/bvh_ray_traversal.wgsl"

// Bindings for BVH traversal
@group(1) @binding(0) var<uniform> tlas_bvh_info: BVHInfo;
@group(1) @binding(1) var<storage, read> rays: array<Ray>; // The mesh's vertex buffer
@group(1) @binding(2) var<storage, read_write> hits: array<RayHit>;
@group(1) @binding(3) var<storage, read> tlas_bvh2_bounds: array<AABB>;
@group(1) @binding(4) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(5) var<storage, read> blas_bvh2_nodes: array<AABB>;
@group(1) @binding(6) var<storage, read> blas_directory: array<MeshDirectoryEntry>;
@group(1) @binding(7) var<storage, read> index_buffer: array<u32>;
@group(1) @binding(8) var<storage, read> entity_index_lookup: array<u32>;

@compute @workgroup_size(128)
fn traverse_tlas_bvh(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ray_count = arrayLength(&rays);
    if (global_id.x >= ray_count) { return; }

    var ray = rays[global_id.x];
    let tlas_only = ray.inv_direction.w > 0.5;

    let hit_result = trace_ray_closest_tlas(&ray, tlas_only);

    var hit: RayHit;
    if (hit_result.has_hit != 0u) {
        let prim_store = hit_result.prim_store;
        let entity_resolved = entity_index_lookup[prim_store];
        let entity_transform = entity_transforms[entity_resolved];

        if (tlas_only) {
            let p_world = ray.origin_and_tmin.xyz + ray.direction_and_tmax.xyz * hit_result.t_hit;
            hit.position_and_t = vec4<f32>(p_world, f32(hit_result.t_hit));
            hit.normal_and_user_data = vec4<f32>(vec3<f32>(0.0), f32(hit_result.prim_store));
        } else {
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

            let vertex0 = decode_vertex(vertex_buffer[v0i]);
            let vertex1 = decode_vertex(vertex_buffer[v1i]);
            let vertex2 = decode_vertex(vertex_buffer[v2i]);
            let v0 = vertex0.position.xyz;
            let v1 = vertex1.position.xyz;
            let v2 = vertex2.position.xyz;

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

            let n_local = vertex0.normal.xyz * w_bc +
                vertex1.normal.xyz * u_bc +
                vertex2.normal.xyz * v_bc;
            var world_n = safe_normalize(
                (entity_transform.transpose_inverse_model_matrix * vec4<f32>(n_local, 0.0)).xyz
            );

            let ray_is_backfacing = dot(world_n, ray.direction_and_tmax.xyz) > 0.0;
            world_n = select(world_n, -world_n, ray_is_backfacing);

            hit.position_and_t = vec4<f32>(p_world, f32(hit_result.t_hit));
            hit.normal_and_user_data = vec4<f32>(world_n, f32(hit_result.prim_store));
        }
    }

    hits[global_id.x] = hit;
}
