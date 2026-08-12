#include "common.wgsl"
#include "gi/surface_cache_common.wgsl"

@group(1) @binding(0) var<uniform> surface_cache_params: SurfaceCacheParams;
@group(1) @binding(1) var<storage, read_write> surface_cache: array<SurfacePatch>;
@group(1) @binding(2) var<storage, read_write> surface_cache_sh: array<u32>;
@group(1) @binding(3) var<storage, read> update_indices: array<u32>;
@group(1) @binding(4) var<storage, read> bootstrap_indices: array<u32>;
@group(1) @binding(5) var<storage, read> counters: SurfaceCacheCountersReadOnly;
@group(1) @binding(6) var<storage, read> hit_info: array<SurfaceCacheHitInfo>;
@group(1) @binding(7) var<storage, read> radiance_info: array<SurfaceCacheRadianceInfo>;

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bootstrap_batch = gid.x >= counters.update_patch_count;
    if (
        bootstrap_batch &&
        counters.bootstrap_rays_per_patch >
            SURFACE_CACHE_PARALLEL_BOOTSTRAP_THRESHOLD
    ) {
        return;
    }

    let active_index = select(
        gid.x,
        gid.x - counters.update_patch_count,
        bootstrap_batch
    );
    if (
        bootstrap_batch &&
        active_index >= counters.bootstrap_patch_count
    ) {
        return;
    }

    var patch_index = 0u;
    if (bootstrap_batch) {
        patch_index = bootstrap_indices[
            surface_cache_bootstrap_schedule_index(
                active_index,
                counters
            )
        ];
    } else {
        let source_index = surface_cache_regular_schedule_index(
            active_index,
            counters
        );
        patch_index = update_indices[source_index];
    }
    let rays_per_patch = select(
        surface_cache_regular_rays_per_patch(surface_cache_params),
        max(counters.bootstrap_rays_per_patch, 1u),
        bootstrap_batch
    );
    let local_ray_base = active_index * rays_per_patch;

    var sample_sh_sum = sh_l1_rgb_zero();
    var luminance_sum = 0.0;
    var luminance_squared_sum = 0.0;
    var valid_sample_count = 0.0;

    for (var ray_index = 0u; ray_index < rays_per_patch; ray_index = ray_index + 1u) {
        let ray_data_index = surface_cache_ray_data_index(
            local_ray_base + ray_index,
            arrayLength(&radiance_info),
            bootstrap_batch
        );
        let sample_info = radiance_info[ray_data_index].sample_radiance;
        if (sample_info.w <= 0.0) {
            continue;
        }

        var sample_radiance = sample_info.xyz;
        let shadow_info = radiance_info[ray_data_index].shadow_radiance;
        if (shadow_info.w == 2.0) {
            sample_radiance += shadow_info.xyz;
        }
        sample_radiance = safe_clamp_vec3_max(
            sample_radiance,
            SURFACE_CACHE_MAX_RADIANCE
        );
        // Store the cell's directional signal in world space. Ray origins and
        // hemisphere normals intentionally move across the cell, so a basis
        // tied to the current representative surface sample cannot persist.
        let direction_weight = hit_info[ray_data_index].ray_direction_sampling_weight;
        let sample_direction = safe_normalize(direction_weight.xyz);
        let sampling_weight = max(direction_weight.w, 0.0);
        sample_sh_sum = sh_l1_rgb_add(
            sample_sh_sum,
            sh_project_onto_l1_rgb(
                sample_direction,
                sample_radiance * sampling_weight
            )
        );
        let sample_luminance = luminance(sample_radiance);
        luminance_sum += sample_luminance;
        luminance_squared_sum += sample_luminance * sample_luminance;
        valid_sample_count += 1.0;
    }

    surface_cache_commit_accumulation(
        &surface_cache,
        &surface_cache_sh,
        surface_cache_params,
        patch_index,
        sample_sh_sum,
        luminance_sum,
        luminance_squared_sum,
        valid_sample_count
    );
}
