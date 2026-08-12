#include "common.wgsl"
#include "gi/surface_cache_common.wgsl"

@group(1) @binding(0) var<uniform> surface_cache_params: SurfaceCacheParams;
@group(1) @binding(1) var<storage, read_write> surface_cache: array<SurfacePatch>;
@group(1) @binding(2) var<storage, read_write> surface_cache_sh: array<u32>;
@group(1) @binding(3) var<storage, read> bootstrap_indices: array<u32>;
@group(1) @binding(4) var<storage, read> counters: SurfaceCacheCountersReadOnly;
@group(1) @binding(5) var<storage, read> hit_info: array<SurfaceCacheHitInfo>;
@group(1) @binding(6) var<storage, read> radiance_info: array<SurfaceCacheRadianceInfo>;

const SURFACE_CACHE_BOOTSTRAP_REDUCTION_SIZE: u32 = 32u;
var<workgroup> bootstrap_sh_sum: array<
    SH_L1_RGB,
    SURFACE_CACHE_BOOTSTRAP_REDUCTION_SIZE
>;
var<workgroup> bootstrap_moment_sum: array<
    vec4<f32>,
    SURFACE_CACHE_BOOTSTRAP_REDUCTION_SIZE
>;

// ─────────────────────────────────────────────────────────────────────────────
// Bootstrap batches are intentionally wide. One workgroup now owns one patch,
// projecting its rays in parallel before a workgroup-local tree reduction.
// This replaces the old 256-iteration serial tail that dominated moving frames.
// ─────────────────────────────────────────────────────────────────────────────
@compute @workgroup_size(32, 1, 1)
fn cs(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32
) {
    let bootstrap_index = workgroup_id.x;
    if (bootstrap_index >= counters.bootstrap_patch_count) {
        return;
    }

    let rays_per_patch = max(counters.bootstrap_rays_per_patch, 1u);
    let local_ray_base = bootstrap_index * rays_per_patch;
    var lane_sh_sum = sh_l1_rgb_zero();
    var lane_moment_sum = vec4<f32>(0.0);

    for (
        var ray_index = local_index;
        ray_index < rays_per_patch;
        ray_index = ray_index + SURFACE_CACHE_BOOTSTRAP_REDUCTION_SIZE
    ) {
        let ray_data_index = surface_cache_ray_data_index(
            local_ray_base + ray_index,
            arrayLength(&radiance_info),
            true
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
        let direction_weight =
            hit_info[ray_data_index].ray_direction_sampling_weight;
        let sample_direction = safe_normalize(direction_weight.xyz);
        let sampling_weight = max(direction_weight.w, 0.0);
        lane_sh_sum = sh_l1_rgb_add(
            lane_sh_sum,
            sh_project_onto_l1_rgb(
                sample_direction,
                sample_radiance * sampling_weight
            )
        );
        let sample_luminance = luminance(sample_radiance);
        lane_moment_sum += vec4<f32>(
            sample_luminance,
            sample_luminance * sample_luminance,
            1.0,
            0.0
        );
    }

    bootstrap_sh_sum[local_index] = lane_sh_sum;
    bootstrap_moment_sum[local_index] = lane_moment_sum;
    workgroupBarrier();

    for (
        var reduction_stride = SURFACE_CACHE_BOOTSTRAP_REDUCTION_SIZE / 2u;
        reduction_stride > 0u;
        reduction_stride = reduction_stride / 2u
    ) {
        if (local_index < reduction_stride) {
            bootstrap_sh_sum[local_index] = sh_l1_rgb_add(
                bootstrap_sh_sum[local_index],
                bootstrap_sh_sum[local_index + reduction_stride]
            );
            bootstrap_moment_sum[local_index] +=
                bootstrap_moment_sum[local_index + reduction_stride];
        }
        workgroupBarrier();
    }

    if (local_index == 0u) {
        let patch_index = bootstrap_indices[
            surface_cache_bootstrap_schedule_index(
                bootstrap_index,
                counters
            )
        ];
        let moments = bootstrap_moment_sum[0];
        surface_cache_commit_accumulation(
            &surface_cache,
            &surface_cache_sh,
            surface_cache_params,
            patch_index,
            bootstrap_sh_sum[0],
            moments.x,
            moments.y,
            moments.z
        );
    }
}
