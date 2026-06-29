// =============================================================================
// GI-1.0 World Cache Ray Tracing - Shadow Hit Pass
// - Traces shadow rays from active world cache cells against BVH
// - Writes visibility to path state
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
// HELPER: Process a shadow ray and write result
// =============================================================================

fn process_shadow_ray(active_index: u32) {
    var ray: Ray;
    ray.origin_and_tmin = world_cache_path_state[active_index].shadow_origin;
    ray.direction_and_tmax = world_cache_path_state[active_index].shadow_direction;
    ray.inv_direction = vec4<f32>(
        1.0 / max(abs(ray.direction_and_tmax.x), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.x < 0.0),
        1.0 / max(abs(ray.direction_and_tmax.y), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.y < 0.0),
        1.0 / max(abs(ray.direction_and_tmax.z), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.z < 0.0),
        0.0
    );

    if (!trace_ray_any(&ray)) {
        // No shadow hit - light is visible
        world_cache_path_state[active_index].state_u32.z = 1u;
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
        process_shadow_ray(active_index);
    }
}
