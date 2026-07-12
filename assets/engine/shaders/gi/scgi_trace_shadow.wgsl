#define RAY_TRAVERSAL_USE_RAY_INSTANCE_TRANSFORMS

#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "gi/scgi_common.wgsl"
#include "acceleration/bvh_ray_traversal.wgsl"

@group(1) @binding(0) var<uniform> scgi_params: SCGIParams;
@group(1) @binding(1) var<storage, read> counters: SCGICountersReadOnly;
@group(1) @binding(2) var<storage, read_write> hit_info: array<SCGIHitInfo>;
@group(1) @binding(3) var<storage, read> tlas_bvh2_bounds: array<AABB>;
@group(1) @binding(4) var<uniform> tlas_bvh_info: BVHInfo;
@group(1) @binding(5) var<storage, read> blas_bvh2_nodes: array<AABB>;
@group(1) @binding(6) var<storage, read> blas_directory: array<MeshDirectoryEntry>;
@group(1) @binding(7) var<storage, read> ray_instance_transforms: array<RayInstanceTransform>;
@group(1) @binding(8) var<storage, read> index_buffer: array<u32>;
@group(1) @binding(9) var<storage, read> entity_index_lookup: array<u32>;
@group(1) @binding(10) var<storage, read_write> radiance_info: array<SCGIRadianceInfo>;

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell_index = gid.x;
    if (
        cell_index >= counters.active_patch_count ||
        radiance_info[cell_index].shadow_radiance.w != 1.0 ||
        hit_info[cell_index].shadow_direction.w <= 0.0
    ) {
        return;
    }

    var ray: Ray;
    ray.origin_and_tmin = hit_info[cell_index].shadow_origin;
    ray.direction_and_tmax = hit_info[cell_index].shadow_direction;
    ray.inv_direction = vec4<f32>(
        1.0 / max(abs(ray.direction_and_tmax.x), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.x < 0.0),
        1.0 / max(abs(ray.direction_and_tmax.y), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.y < 0.0),
        1.0 / max(abs(ray.direction_and_tmax.z), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.z < 0.0),
        0.0
    );

    if (!trace_ray_any(&ray)) {
        radiance_info[cell_index].shadow_radiance.w = 2.0;
    }
}
