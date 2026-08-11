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
    let regular_rays_per_patch = surface_cache_regular_rays_per_patch(
        surface_cache_params
    );
    let active_patch_count = min(
        atomicLoad(&counters.active_patch_count),
        patch_capacity
    );
    let bootstrap_patch_count = min(
        min(
            atomicLoad(&counters.bootstrap_patch_count),
            u32(surface_cache_params.bootstrap_patch_capacity)
        ),
        active_patch_count
    );
    let available_update_patch_count = min(
        atomicLoad(&counters.update_patch_count),
        active_patch_count - bootstrap_patch_count
    );

    // ────────────────────────────────────────────────────────────────────────
    // The active cache footprint owns one immutable frame budget. Bootstrap
    // patches borrow whole regular-sized batches from mature patches instead
    // of expanding the workload when camera motion exposes new surfaces.
    // ────────────────────────────────────────────────────────────────────────
    var bootstrap_rays_per_patch = 0u;
    var update_patch_count = available_update_patch_count;
    if (bootstrap_patch_count > 0u) {
        let maximum_bootstrap_batch_count = max(
            surface_cache_params.maximum_bootstrap_rays_per_patch /
                regular_rays_per_patch,
            1u
        );
        let bootstrap_ray_budget_patch_count = max(
            bootstrap_patch_count,
            u32(
                f32(active_patch_count) * clamp(
                    surface_cache_params.bootstrap_ray_budget_fraction,
                    0.0,
                    1.0
                )
            )
        );
        let budget_bootstrap_batch_count = max(
            bootstrap_ray_budget_patch_count / bootstrap_patch_count,
            1u
        );
        let bootstrap_batch_count = min(
            maximum_bootstrap_batch_count,
            budget_bootstrap_batch_count
        );
        bootstrap_rays_per_patch =
            bootstrap_batch_count * regular_rays_per_patch;
        update_patch_count = min(
            available_update_patch_count,
            active_patch_count -
                bootstrap_patch_count * bootstrap_batch_count
        );
    }
    let regular_ray_count = update_patch_count * regular_rays_per_patch;
    let bootstrap_ray_count = bootstrap_patch_count *
        bootstrap_rays_per_patch;
    var regular_schedule_offset = 0u;
    if (
        update_patch_count < available_update_patch_count &&
        available_update_patch_count > 0u
    ) {
        regular_schedule_offset = hash(
            (u32(surface_cache_params.frame_index) * 0x9e3779b9u) ^
                available_update_patch_count
        ) % available_update_patch_count;
    }

    atomicStore(&counters.update_patch_count, update_patch_count);
    atomicStore(&counters.bootstrap_patch_count, bootstrap_patch_count);
    atomicStore(&counters.active_patch_count, active_patch_count);
    atomicStore(
        &counters.bootstrap_rays_per_patch,
        bootstrap_rays_per_patch
    );
    atomicStore(
        &counters.regular_schedule_offset,
        regular_schedule_offset
    );

    let parallel_bootstrap_accumulation =
        bootstrap_rays_per_patch > SURFACE_CACHE_PARALLEL_BOOTSTRAP_THRESHOLD;
    let serial_bootstrap_patch_count = select(
        bootstrap_patch_count,
        0u,
        parallel_bootstrap_accumulation
    );
    let parallel_bootstrap_patch_count = select(
        0u,
        bootstrap_patch_count,
        parallel_bootstrap_accumulation
    );

    write_dispatch_args(
        0u,
        surface_cache_workgroup_count(regular_ray_count + bootstrap_ray_count)
    );
    write_dispatch_args(
        3u,
        surface_cache_workgroup_count(
            update_patch_count + serial_bootstrap_patch_count
        )
    );
    write_dispatch_args(
        6u,
        parallel_bootstrap_patch_count
    );
}
