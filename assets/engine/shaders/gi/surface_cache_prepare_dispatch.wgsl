#include "common.wgsl"
#include "gi/surface_cache_common.wgsl"

@group(1) @binding(0) var<uniform> surface_cache_params: SurfaceCacheParams;
@group(1) @binding(1) var<storage, read_write> counters: SurfaceCacheCounters;
@group(1) @binding(2) var<storage, read_write> dispatch_args: array<u32>;

const SURFACE_CACHE_WORKGROUP_SIZE: u32 = 128u;
const SURFACE_CACHE_TRACE_DISPATCH_OFFSET: u32 = 0u;
const SURFACE_CACHE_ACCUMULATION_DISPATCH_OFFSET: u32 = 3u;

fn surface_cache_workgroup_count(item_count: u32) -> u32 {
    return (item_count + SURFACE_CACHE_WORKGROUP_SIZE - 1u) /
        SURFACE_CACHE_WORKGROUP_SIZE;
}

fn surface_cache_write_dispatch_args(offset: u32, workgroup_count: u32) {
    dispatch_args[offset] = workgroup_count;
    dispatch_args[offset + 1u] = 1u;
    dispatch_args[offset + 2u] = 1u;
}

@compute @workgroup_size(1, 1, 1)
fn cs() {
    let available_patch_count = min(
        atomicLoad(&counters.update_patch_count),
        u32(surface_cache_params.total_patch_count)
    );
    let maximum_rays_per_patch = max(
        u32(surface_cache_params.rays_per_patch),
        1u
    );
    let frame_ray_budget = max(
        u32(surface_cache_params.maximum_ray_count_per_frame),
        maximum_rays_per_patch
    );
    let scheduled_patch_count = min(
        available_patch_count,
        frame_ray_budget
    );
    var scheduled_rays_per_patch = 0u;
    if (scheduled_patch_count > 0u) {
        scheduled_rays_per_patch = min(
            maximum_rays_per_patch,
            frame_ray_budget / scheduled_patch_count
        );
    }

    var schedule_offset = 0u;
    if (scheduled_patch_count < available_patch_count) {
        schedule_offset = hash(
            (u32(surface_cache_params.frame_index) * 0x9e3779b9u) ^
                available_patch_count
        ) % available_patch_count;
    }

    atomicStore(&counters.update_patch_count, scheduled_patch_count);
    atomicStore(&counters.schedule_offset, schedule_offset);
    atomicStore(&counters.available_update_patch_count, available_patch_count);
    atomicStore(&counters.scheduled_rays_per_patch, scheduled_rays_per_patch);

    surface_cache_write_dispatch_args(
        SURFACE_CACHE_TRACE_DISPATCH_OFFSET,
        surface_cache_workgroup_count(
            scheduled_patch_count * scheduled_rays_per_patch
        )
    );
    surface_cache_write_dispatch_args(
        SURFACE_CACHE_ACCUMULATION_DISPATCH_OFFSET,
        surface_cache_workgroup_count(scheduled_patch_count)
    );
}
