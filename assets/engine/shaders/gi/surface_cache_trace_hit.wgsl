#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "gi/surface_cache_common.wgsl"
#include "acceleration/bvh_ray_traversal.wgsl"

@group(1) @binding(0) var<uniform> surface_cache_params: SurfaceCacheParams;
@group(1) @binding(1) var<storage, read> counters: SurfaceCacheCountersReadOnly;
@group(1) @binding(2) var<storage, read_write> hit_info: array<SurfaceCacheHitInfo>;
@group(1) @binding(3) var<storage, read> tlas_bvh2_bounds: array<AABB>;
@group(1) @binding(4) var<uniform> tlas_bvh_info: BVHInfo;
@group(1) @binding(5) var<storage, read> blas_bvh2_nodes: array<AABB>;
@group(1) @binding(6) var<storage, read> blas_directory: array<MeshDirectoryEntry>;
@group(1) @binding(7) var<storage, read> compact_transforms: array<RayInstanceTransform>;
@group(1) @binding(8) var<storage, read> index_buffer: array<u32>;
@group(1) @binding(9) var<storage, read> entity_index_lookup: array<u32>;

fn trace_surface_cache_prepared_ray(ray_data_index: u32) {
    let direction = hit_info[ray_data_index].ray_direction_sampling_weight.xyz;
    let origin_tmin = hit_info[ray_data_index].hit_barycentrics_t;
    var ray: Ray;
    ray.origin_and_tmin = origin_tmin;
    ray.direction_and_tmax = vec4<f32>(direction, surface_cache_params.max_ray_length);
    ray.inv_direction = vec4<f32>(
        1.0 / max(abs(direction.x), 1e-8) * select(1.0, -1.0, direction.x < 0.0),
        1.0 / max(abs(direction.y), 1e-8) * select(1.0, -1.0, direction.y < 0.0),
        1.0 / max(abs(direction.z), 1e-8) * select(1.0, -1.0, direction.z < 0.0),
        0.0
    );

    hit_info[ray_data_index].hit_identity.x = INVALID_IDX;

    let hit_result = trace_ray_closest(&ray);
    if (hit_result.has_hit != 0u) {
        // A backface is still an opaque boundary. Preserve it as a blocker so
        // thin walls and creases cannot expose radiance from behind them, but
        // do not pass its attributes to the shading recurrence.
        if (hit_result.tri_indices.w == 0u) {
            hit_info[ray_data_index].hit_identity.x =
                SURFACE_CACHE_BACKFACE_HIT;
        } else {
            hit_info[ray_data_index].hit_identity = vec4<u32>(
                entity_index_lookup[hit_result.prim_store],
                hit_result.tri_indices.xyz
            );
        }
        hit_info[ray_data_index].hit_barycentrics_t = vec4<f32>(
            hit_result.barycentrics,
            hit_result.t_hit,
            0.0
        );
    }
}

@compute @workgroup_size(128, 1, 1)
fn cs(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {
    bvh_stack_lane = local_idx;
    if (gid.x >= surface_cache_total_ray_count(counters)) {
        return;
    }
    let work = surface_cache_ray_work(gid.x, counters);
    trace_surface_cache_prepared_ray(work.data_index);
}
