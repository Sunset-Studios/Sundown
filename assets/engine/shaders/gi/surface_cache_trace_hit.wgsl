#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "gi/surface_cache_common.wgsl"
#include "acceleration/bvh_ray_traversal.wgsl"

@group(1) @binding(0) var<uniform> surface_cache_params: SurfaceCacheParams;
@group(1) @binding(1) var<storage, read> surface_cache: array<SurfacePatchReadOnly>;
@group(1) @binding(2) var<storage, read> update_indices: array<u32>;
@group(1) @binding(3) var<storage, read> counters: SurfaceCacheCountersReadOnly;
@group(1) @binding(4) var<storage, read_write> hit_info: array<SurfaceCacheHitInfo>;
@group(1) @binding(5) var<storage, read> tlas_bvh2_bounds: array<AABB>;
@group(1) @binding(6) var<uniform> tlas_bvh_info: BVHInfo;
@group(1) @binding(7) var<storage, read> blas_bvh2_nodes: array<AABB>;
@group(1) @binding(8) var<storage, read> blas_directory: array<MeshDirectoryEntry>;
@group(1) @binding(9) var<storage, read> compact_transforms: array<RayInstanceTransform>;
@group(1) @binding(10) var<storage, read> index_buffer: array<u32>;
@group(1) @binding(11) var<storage, read> entity_index_lookup: array<u32>;

struct SurfaceCacheRaySample {
    direction: vec3<f32>,
    sampling_weight: f32,
};

fn sample_uniform_hemisphere_surface_cache(normal: vec3<f32>, r1: f32, r2: f32) -> vec3<f32> {
    let phi = 2.0 * PI * r1;
    let cos_theta = r2;
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    return surface_cache_hemisphere_frame(normal) * vec3<f32>(
        sin_theta * cos(phi),
        sin_theta * sin(phi),
        cos_theta
    );
}

fn generate_ray_sample(
    seed: u32,
    normal: vec3<f32>,
    sample_index: u32
) -> SurfaceCacheRaySample {
    // A Cranley-Patterson rotated R2 sequence uniformly covers the hemisphere.
    // Unlike history-guided RIS it remains unbiased when lighting changes and
    // cannot reinforce a noisy lobe already present in this cache entry.
    var rng = random_seed(seed);
    let rotation_u = rand_float(rng);
    rng = random_seed(rng);
    let rotation_v = rand_float(rng);
    let sequence_value = f32(sample_index);
    let r1 = fract(rotation_u + sequence_value * 0.7548776662466927);
    let r2 = fract(rotation_v + sequence_value * 0.5698402909980532);
    var result: SurfaceCacheRaySample;
    result.direction = sample_uniform_hemisphere_surface_cache(normal, r1, r2);
    result.sampling_weight = 2.0 * PI;
    return result;
}

fn trace_surface_cache_ray(
    active_index: u32,
    ray_index_in_patch: u32,
    ray_data_index: u32
) {
    let patch_index = update_indices[active_index];
    let surface_patch = surface_cache[patch_index];
    let normal = safe_normalize(surface_patch.normal_cell_exponent.xyz);
    let seed = surface_cache_patch_rng(patch_index, surface_patch.grid_key);
    let ray_sample = generate_ray_sample(
        seed,
        normal,
        u32(surface_patch.history.y) + ray_index_in_patch
    );
    let direction = ray_sample.direction;

    var ray: Ray;
    let cell_exponent = surface_cache_grid_key_cell_exponent(surface_patch.grid_key);
    let origin_offset = max(
        0.001,
        surface_cache_cell_size(cell_exponent) * 0.002
    );
    ray.origin_and_tmin = vec4<f32>(
        surface_patch.position_frame.xyz + normal * origin_offset,
        origin_offset * 0.25
    );
    ray.direction_and_tmax = vec4<f32>(direction, surface_cache_params.max_ray_length);
    ray.inv_direction = vec4<f32>(
        1.0 / max(abs(direction.x), 1e-8) * select(1.0, -1.0, direction.x < 0.0),
        1.0 / max(abs(direction.y), 1e-8) * select(1.0, -1.0, direction.y < 0.0),
        1.0 / max(abs(direction.z), 1e-8) * select(1.0, -1.0, direction.z < 0.0),
        0.0
    );

    hit_info[ray_data_index].ray_direction_sampling_weight = vec4<f32>(
        direction,
        ray_sample.sampling_weight
    );
    hit_info[ray_data_index].hit_identity = vec4<u32>(INVALID_IDX, 0u, 0u, 0u);
    hit_info[ray_data_index].hit_barycentrics_t = vec4<f32>(0.0);

    let hit_result = trace_ray_closest(&ray);
    if (hit_result.has_hit == 0u) {
        return;
    }

    hit_info[ray_data_index].hit_identity = vec4<u32>(
        entity_index_lookup[hit_result.prim_store],
        hit_result.tri_indices.xyz
    );
    hit_info[ray_data_index].hit_barycentrics_t = vec4<f32>(
        hit_result.barycentrics,
        hit_result.t_hit,
        0.0
    );
}

@compute @workgroup_size(128, 1, 1)
fn cs(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {
    bvh_stack_lane = local_idx;
    let rays_per_patch = surface_cache_rays_per_patch(surface_cache_params);
    let ray_data_index = gid.x;
    let active_index = ray_data_index / rays_per_patch;
    if (active_index >= counters.update_patch_count) {
        return;
    }
    trace_surface_cache_ray(
        active_index,
        ray_data_index % rays_per_patch,
        ray_data_index
    );
}
