// =============================================================================
// RTAO BVH TRAVERSAL (SIMPLIFIED)
// =============================================================================
//
// Traces rays for Ray Traced Ambient Occlusion only:
//   - Closest-hit TLAS/BLAS traversal (same as full path tracer)
//   - No shadow rays, no NEE, no any-hit
//   - On hit: set state_u32.w = 0 for resolve; on miss: leave 0xffffffff
//
// =============================================================================

#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "gi/gi_common.wgsl"
#include "acceleration/bvh_ray_traversal.wgsl"

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read_write> gi_counters: GICounters;
@group(1) @binding(2) var<storage, read_write> pixel_path_state: array<AOPixelPathState>;
@group(1) @binding(3) var<storage, read_write> ray_work_queue: array<u32>;
@group(1) @binding(4) var<storage, read> tlas_bvh2_bounds: array<AABB>;
@group(1) @binding(5) var<uniform> tlas_bvh_info: BVHInfo;
@group(1) @binding(6) var<storage, read> blas_bvh2_nodes: array<AABB>;
@group(1) @binding(7) var<storage, read> blas_directory: array<MeshDirectoryEntry>;
@group(1) @binding(8) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(9) var<storage, read> index_buffer: array<u32>;
@group(1) @binding(10) var<storage, read> entity_index_lookup: array<u32>;

// =============================================================================
// RTAO: Process primary ray — record hit for AO (state_u32.w only)
// =============================================================================

fn process_rtao_ray(ray_index: u32) {
    var ray: Ray;
    ray.origin_and_tmin = pixel_path_state[ray_index].origin_tmin;
    ray.direction_and_tmax = pixel_path_state[ray_index].direction_tmax;
    ray.inv_direction = vec4<f32>(
        1.0 / max(abs(ray.direction_and_tmax.x), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.x < 0.0),
        1.0 / max(abs(ray.direction_and_tmax.y), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.y < 0.0),
        1.0 / max(abs(ray.direction_and_tmax.z), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.z < 0.0),
        0.0
    );

    let hit_result = trace_ray_closest(&ray);

    if (hit_result.has_hit != 0u) {
        pixel_path_state[ray_index].state_u32.w = 0u;
        pixel_path_state[ray_index].origin_tmin.w = hit_result.t_hit;
    }
}

// =============================================================================
// MAIN COMPUTE
// =============================================================================

@compute @workgroup_size(128, 1, 1)
fn cs(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {
    bvh_stack_lane = local_idx;
    let rays_per_pixel = u32(gi_params.screen_ray_count);
    let gi_resolution = vec2<u32>(u32(gi_params.gi_resolution_x), u32(gi_params.gi_resolution_y));
    let total_pixels = gi_resolution.x * gi_resolution.y;
    let total_rays = total_pixels * rays_per_pixel;

    if (gid.x >= total_rays) {
        return;
    }

    let queue_count = atomicLoad(&gi_counters.ray_queue_count);

    loop {
        let queue_index = atomicAdd(&gi_counters.ray_queue_primary_head, 1u);
        if (queue_index >= queue_count) {
            break;
        }
        let ray_slot = ray_work_queue[queue_index];
        process_rtao_ray(ray_slot);
    }
}
