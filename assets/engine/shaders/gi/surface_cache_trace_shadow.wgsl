#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "gi/surface_cache_common.wgsl"
#include "acceleration/bvh_ray_traversal.wgsl"

@group(1) @binding(0) var<uniform> surface_cache_params: SurfaceCacheParams;
@group(1) @binding(1) var<storage, read> counters: SurfaceCacheCountersReadOnly;
@group(1) @binding(2) var<storage, read> tlas_bvh2_bounds: array<AABB>;
@group(1) @binding(3) var<uniform> tlas_bvh_info: BVHInfo;
@group(1) @binding(4) var<storage, read> blas_bvh2_nodes: array<AABB>;
@group(1) @binding(5) var<storage, read> blas_directory: array<MeshDirectoryEntry>;
@group(1) @binding(6) var<storage, read> compact_transforms: array<RayInstanceTransform>;
@group(1) @binding(7) var<storage, read> index_buffer: array<u32>;
@group(1) @binding(8) var<storage, read> entity_index_lookup: array<u32>;
@group(1) @binding(9) var<storage, read_write> radiance_info: array<SurfaceCacheRadianceInfo>;

@compute @workgroup_size(128, 1, 1)
fn cs(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {
    bvh_stack_lane = local_idx;
    if (gid.x >= surface_cache_total_ray_count(counters, surface_cache_params)) {
        return;
    }
    let work = surface_cache_ray_work(
        gid.x,
        arrayLength(&radiance_info),
        counters,
        surface_cache_params
    );
    let ray_data_index = work.data_index;
    let shadow_state = radiance_info[ray_data_index].shadow_radiance.w;
    let shadow_direction = radiance_info[ray_data_index].shadow_direction;
    if (
        shadow_state != 1.0 ||
        shadow_direction.w <= 0.0
    ) {
        return;
    }

    var ray: Ray;
    ray.origin_and_tmin = radiance_info[ray_data_index].shadow_origin;
    ray.direction_and_tmax = shadow_direction;
    ray.inv_direction = vec4<f32>(
        1.0 / max(abs(ray.direction_and_tmax.x), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.x < 0.0),
        1.0 / max(abs(ray.direction_and_tmax.y), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.y < 0.0),
        1.0 / max(abs(ray.direction_and_tmax.z), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.z < 0.0),
        0.0
    );

    if (!trace_ray_any(&ray)) {
        radiance_info[ray_data_index].shadow_radiance.w = 2.0;
    }
}
