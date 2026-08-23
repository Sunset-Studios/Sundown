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
    let available_bootstrap_patch_count = min(
        min(
            atomicLoad(&counters.bootstrap_patch_count),
            u32(surface_cache_params.bootstrap_patch_capacity)
        ),
        active_patch_count
    );
    let available_update_patch_count = min(
        atomicLoad(&counters.update_patch_count),
        active_patch_count - available_bootstrap_patch_count
    );
    let maximum_ray_count = max(
        min(
            u32(surface_cache_params.maximum_ray_count_per_frame),
            patch_capacity * regular_rays_per_patch
        ),
        regular_rays_per_patch
    );

    // ────────────────────────────────────────────────────────────────────────
    // Bootstrap and regular refresh work share one immutable ray ceiling.
    // Both queues rotate when oversubscribed, spreading camera-motion and
    // invalidation waves over frames instead of allowing a high-water spike.
    // ────────────────────────────────────────────────────────────────────────
    var bootstrap_patch_count = 0u;
    var bootstrap_rays_per_patch = 0u;
    if (available_bootstrap_patch_count > 0u) {
        let bootstrap_ray_budget = select(
            maximum_ray_count,
            max(
                regular_rays_per_patch,
                u32(f32(maximum_ray_count) * clamp(
                    surface_cache_params.bootstrap_ray_budget_fraction,
                    0.0,
                    1.0
                ))
            ),
            available_update_patch_count > 0u
        );
        // Admit new patches breadth-first. A single valid sample is enough for
        // resolve and temporal reconstruction to avoid an invalid black cell;
        // remaining rays then improve every admitted patch uniformly. Under
        // the default budget this covers the entire cache capacity in one
        // frame without increasing the immutable total ray ceiling.
        bootstrap_patch_count = min(
            available_bootstrap_patch_count,
            max(bootstrap_ray_budget, 1u)
        );
        bootstrap_rays_per_patch = min(
            max(
                u32(surface_cache_params.maximum_bootstrap_rays_per_patch),
                1u
            ),
            max(
                bootstrap_ray_budget /
                    bootstrap_patch_count,
                1u
            )
        );
    }
    let bootstrap_ray_count = bootstrap_patch_count *
        bootstrap_rays_per_patch;
    let remaining_ray_count = maximum_ray_count - bootstrap_ray_count;
    var update_patch_count = 0u;
    var scheduled_regular_rays_per_patch = 0u;
    if (available_update_patch_count > 0u && remaining_ray_count > 0u) {
        // Spread the regular budget breadth-first too. Underconverged patches
        // should advance together instead of allowing a raster-ordered subset
        // to consume a full batch before the rest receive their next sample.
        update_patch_count = min(
            available_update_patch_count,
            remaining_ray_count
        );
        scheduled_regular_rays_per_patch = min(
            regular_rays_per_patch,
            max(remaining_ray_count / update_patch_count, 1u)
        );
    }
    let regular_ray_count = update_patch_count *
        scheduled_regular_rays_per_patch;
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
    var bootstrap_schedule_offset = 0u;
    if (
        bootstrap_patch_count < available_bootstrap_patch_count &&
        available_bootstrap_patch_count > 0u
    ) {
        bootstrap_schedule_offset = hash(
            (u32(surface_cache_params.frame_index) * 0x85ebca6bu) ^
                available_bootstrap_patch_count
        ) % available_bootstrap_patch_count;
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
    atomicStore(
        &counters.available_bootstrap_patch_count,
        available_bootstrap_patch_count
    );
    atomicStore(
        &counters.bootstrap_schedule_offset,
        bootstrap_schedule_offset
    );
    atomicStore(
        &counters.available_update_patch_count,
        available_update_patch_count
    );
    atomicStore(
        &counters.regular_rays_per_patch,
        scheduled_regular_rays_per_patch
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
