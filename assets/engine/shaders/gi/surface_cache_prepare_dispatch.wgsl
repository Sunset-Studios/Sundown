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
        active_patch_count
    );
    let update_patch_count = min(
        atomicLoad(&counters.update_patch_count),
        active_patch_count - bootstrap_patch_count
    );
    let regular_rays_per_patch = max(regular_batch.rays_per_patch, 1u);
    let regular_ray_count = update_patch_count * regular_rays_per_patch;
    let bootstrap_minimum_ray_count = bootstrap_patch_count *
        regular_rays_per_patch;
    let remaining_ray_capacity = total_ray_capacity -
        regular_ray_count - bootstrap_minimum_ray_count;

    // Every new patch first receives the same baseline as a mature patch. The
    // remaining fixed ray budget is then divided evenly across all new patches,
    // avoiding the clean/noisy cell mosaic produced by boosting an arbitrary
    // prefix at a fixed ray count.
    var bootstrap_rays_per_patch = 0u;
    if (bootstrap_patch_count > 0u) {
        let maximum_extra_rays =
            max(bootstrap_batch.rays_per_patch, regular_rays_per_patch) -
            regular_rays_per_patch;
        bootstrap_rays_per_patch = regular_rays_per_patch + min(
            maximum_extra_rays,
            remaining_ray_capacity / bootstrap_patch_count
        );
    }
    let bootstrap_ray_count = bootstrap_patch_count *
        bootstrap_rays_per_patch;

    atomicStore(&counters.update_patch_count, update_patch_count);
    atomicStore(&counters.bootstrap_patch_count, bootstrap_patch_count);
    atomicStore(&counters.active_patch_count, active_patch_count);
    atomicStore(
        &counters.bootstrap_rays_per_patch,
        bootstrap_rays_per_patch
    );

    write_dispatch_args(0u, surface_cache_workgroup_count(regular_ray_count));
    write_dispatch_args(3u, surface_cache_workgroup_count(update_patch_count));
    write_dispatch_args(6u, surface_cache_workgroup_count(bootstrap_ray_count));
    write_dispatch_args(9u, surface_cache_workgroup_count(bootstrap_patch_count));
}
