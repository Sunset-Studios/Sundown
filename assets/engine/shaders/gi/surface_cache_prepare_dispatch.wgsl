#include "common.wgsl"
#include "gi/surface_cache_common.wgsl"

@group(1) @binding(0) var<uniform> surface_cache_params: SurfaceCacheParams;
@group(1) @binding(1) var<storage, read_write> counters: SurfaceCacheCounters;
@group(1) @binding(2) var<storage, read_write> dispatch_args: array<u32>;
@group(1) @binding(3) var<uniform> regular_batch: SurfaceCacheRayBatchParams;
@group(1) @binding(4) var<uniform> bootstrap_batch: SurfaceCacheRayBatchParams;

const SURFACE_CACHE_WORKGROUP_SIZE: u32 = 128u;

fn surface_cache_workgroup_count(item_count: u32) -> u32 {
    return (item_count + SURFACE_CACHE_WORKGROUP_SIZE - 1u) /
        SURFACE_CACHE_WORKGROUP_SIZE;
}

fn write_dispatch_args(offset: u32, workgroup_count: u32) {
    dispatch_args[offset] = workgroup_count;
    dispatch_args[offset + 1u] = 1u;
    dispatch_args[offset + 2u] = 1u;
}

@compute @workgroup_size(1, 1, 1)
fn cs() {
    let patch_capacity = u32(surface_cache_params.total_patch_count);
    let total_ray_capacity = patch_capacity * regular_batch.rays_per_patch;
    let active_patch_count = min(
        atomicLoad(&counters.active_patch_count),
        patch_capacity
    );
    let bootstrap_patch_count = min(
        min(
            atomicLoad(&counters.bootstrap_patch_count),
            bootstrap_batch.patch_capacity
        ),
        total_ray_capacity / max(bootstrap_batch.rays_per_patch, 1u)
    );
    let bootstrap_ray_count = bootstrap_patch_count *
        bootstrap_batch.rays_per_patch;
    let remaining_ray_capacity = total_ray_capacity - bootstrap_ray_count;
    // New cells own the front of both the patch and ray arrays. Regular work
    // is trimmed to the remaining budget and consumes both arrays backwards,
    // so the two batches cannot overlap and no bootstrap working set is needed.
    let update_patch_count = min(
        min(
            atomicLoad(&counters.update_patch_count),
            active_patch_count
        ),
        remaining_ray_capacity / max(regular_batch.rays_per_patch, 1u)
    );
    let ray_count = update_patch_count * regular_batch.rays_per_patch;

    atomicStore(&counters.update_patch_count, update_patch_count);
    atomicStore(&counters.bootstrap_patch_count, bootstrap_patch_count);
    atomicStore(&counters.active_patch_count, active_patch_count);

    write_dispatch_args(0u, surface_cache_workgroup_count(ray_count));
    write_dispatch_args(3u, surface_cache_workgroup_count(update_patch_count));
    write_dispatch_args(6u, surface_cache_workgroup_count(bootstrap_ray_count));
    write_dispatch_args(9u, surface_cache_workgroup_count(bootstrap_patch_count));
}
