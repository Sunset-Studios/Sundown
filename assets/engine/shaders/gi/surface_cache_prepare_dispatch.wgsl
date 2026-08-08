#include "common.wgsl"
#include "gi/surface_cache_common.wgsl"

@group(1) @binding(0) var<uniform> surface_cache_params: SurfaceCacheParams;
@group(1) @binding(1) var<storage, read_write> counters: SurfaceCacheCounters;
@group(1) @binding(2) var<storage, read_write> dispatch_args: array<u32>;

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
    let active_patch_count = min(
        atomicLoad(&counters.active_patch_count),
        patch_capacity
    );
    let update_patch_count = min(
        atomicLoad(&counters.update_patch_count),
        active_patch_count
    );
    let ray_count = update_patch_count *
        surface_cache_rays_per_patch(surface_cache_params);

    atomicStore(&counters.update_patch_count, update_patch_count);
    atomicStore(&counters.active_patch_count, active_patch_count);

    write_dispatch_args(0u, surface_cache_workgroup_count(ray_count));
    write_dispatch_args(3u, surface_cache_workgroup_count(update_patch_count));
    write_dispatch_args(6u, surface_cache_workgroup_count(active_patch_count));
}
