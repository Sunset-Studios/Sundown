#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "gi/surface_cache_common.wgsl"
#include "acceleration/bvh_ray_traversal.wgsl"

@group(1) @binding(0) var<uniform> surface_cache_params: SurfaceCacheParams;
@group(1) @binding(1) var<storage, read> surface_cache: array<SurfacePatchReadOnly>;
@group(1) @binding(2) var<storage, read> update_indices: array<u32>;
@group(1) @binding(3) var<storage, read> bootstrap_indices: array<u32>;
@group(1) @binding(4) var<storage, read> counters: SurfaceCacheCountersReadOnly;
@group(1) @binding(5) var<storage, read_write> hit_info: array<SurfaceCacheHitInfo>;
@group(1) @binding(6) var<storage, read> tlas_bvh2_bounds: array<AABB>;
@group(1) @binding(7) var<uniform> tlas_bvh_info: BVHInfo;
@group(1) @binding(8) var<storage, read> blas_bvh2_nodes: array<AABB>;
@group(1) @binding(9) var<storage, read> blas_directory: array<MeshDirectoryEntry>;
@group(1) @binding(10) var<storage, read> compact_transforms: array<RayInstanceTransform>;
@group(1) @binding(11) var<storage, read> index_buffer: array<u32>;
@group(1) @binding(12) var<storage, read> entity_index_lookup: array<u32>;

// The BVH traversal stack consumes 12 KiB of workgroup storage. Sixty-six
// patch slots cover every patch intersecting a 128-ray workgroup when sharing
// is active (at least two rays per patch) while keeping the combined allocation
// below WebGPU's guaranteed 16 KiB limit.
const SURFACE_CACHE_SHARED_PATCH_CAPACITY: u32 = 66u;
var<workgroup> shared_patch_origin_tmin: array<
    vec4<f32>,
    SURFACE_CACHE_SHARED_PATCH_CAPACITY
>;
var<workgroup> shared_patch_normal_history: array<
    vec4<f32>,
    SURFACE_CACHE_SHARED_PATCH_CAPACITY
>;
var<workgroup> shared_patch_rotation: array<
    vec2<f32>,
    SURFACE_CACHE_SHARED_PATCH_CAPACITY
>;

fn surface_cache_shared_patch_slot(
    work_index: u32,
    work: SurfaceCacheRayWork
) -> u32 {
    let workgroup_begin = (work_index / 128u) * 128u;
    let workgroup_end = workgroup_begin + 128u;
    let regular_rays_per_patch = surface_cache_regular_rays_per_patch(
        surface_cache_params
    );
    let regular_ray_count = surface_cache_regular_ray_count(
        counters,
        surface_cache_params
    );
    let regular_begin = min(workgroup_begin, regular_ray_count);
    let regular_end = min(workgroup_end, regular_ray_count);
    var regular_patch_span = 0u;
    if (regular_end > regular_begin) {
        regular_patch_span =
            (regular_end - 1u) / regular_rays_per_patch -
            regular_begin / regular_rays_per_patch + 1u;
    }

    if (work.bootstrap_batch == 0u) {
        return work.active_index - regular_begin / regular_rays_per_patch;
    }

    let bootstrap_rays_per_patch = max(counters.bootstrap_rays_per_patch, 1u);
    let bootstrap_begin = max(workgroup_begin, regular_ray_count) - regular_ray_count;
    return regular_patch_span + work.active_index -
        bootstrap_begin / bootstrap_rays_per_patch;
}

fn trace_surface_cache_prepared_ray(
    direction: vec3<f32>,
    origin_tmin: vec4<f32>,
    ray_data_index: u32
) {
    var ray: Ray;
    ray.origin_and_tmin = origin_tmin;
    ray.direction_and_tmax = vec4<f32>(direction, surface_cache_params.max_ray_length);
    ray.inv_direction = vec4<f32>(
        1.0 / max(abs(direction.x), 1e-8) * select(1.0, -1.0, direction.x < 0.0),
        1.0 / max(abs(direction.y), 1e-8) * select(1.0, -1.0, direction.y < 0.0),
        1.0 / max(abs(direction.z), 1e-8) * select(1.0, -1.0, direction.z < 0.0),
        0.0
    );

    hit_info[ray_data_index].ray_direction_sampling_weight = vec4<f32>(
        direction,
        2.0 * PI
    );
    hit_info[ray_data_index].hit_identity.x = INVALID_IDX;

    let hit_result = trace_ray_closest(&ray);
    if (hit_result.has_hit != 0u) {
        // A backface is still an opaque boundary. Preserve it as a blocker so
        // thin walls and creases cannot expose radiance from behind them, but
        // do not pass its attributes to the shading recurrence.
        if (hit_result.tri_indices.w == 0u) {
            hit_info[ray_data_index].hit_identity.x =
                SURFACE_CACHE_BACKFACE_HIT;
        } else {
            hit_info[ray_data_index].hit_identity = vec4<u32>(
                entity_index_lookup[hit_result.prim_store],
                hit_result.tri_indices.xyz
            );
        }
        hit_info[ray_data_index].hit_barycentrics_t = vec4<f32>(
            hit_result.barycentrics,
            hit_result.t_hit,
            0.0
        );
    }
}

fn trace_surface_cache_ray(
    patch_index: u32,
    ray_index_in_patch: u32,
    ray_data_index: u32
) {
    let surface_patch = surface_cache[patch_index];
    let normal = safe_normalize(surface_patch.normal_cell_exponent.xyz);
    let seed = surface_cache_patch_rng(patch_index, surface_patch.grid_key);
    let ray_sample = generate_ray_sample(
        seed,
        normal,
        u32(surface_patch.history.y) + ray_index_in_patch
    );
    let direction = ray_sample.direction;
    let cell_exponent = surface_cache_grid_key_cell_exponent(surface_patch.grid_key);
    let origin_offset = max(
        0.001,
        surface_cache_cell_size(cell_exponent) * 0.002
    );

    let origin_tmin = vec4<f32>(
        surface_patch.position_frame.xyz + normal * origin_offset,
        origin_offset * 0.25
    );
    trace_surface_cache_prepared_ray(direction, origin_tmin, ray_data_index);
}

@compute @workgroup_size(128, 1, 1)
fn cs(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {
    bvh_stack_lane = local_idx;
    let ray_count = surface_cache_total_ray_count(counters, surface_cache_params);
    let valid_work = gid.x < ray_count;
    let work = surface_cache_ray_work(
        gid.x,
        arrayLength(&hit_info),
        counters,
        surface_cache_params
    );

    // The shared array is sized for at least two rays per patch. Breadth-first
    // bootstrap can intentionally assign one ray to each patch during a large
    // admission wave, so use the direct path for that exceptional frame.
    let use_shared_patch_setup =
        surface_cache_regular_rays_per_patch(surface_cache_params) >= 2u &&
        (
            counters.bootstrap_patch_count == 0u ||
            counters.bootstrap_rays_per_patch >= 2u
        );
    if (use_shared_patch_setup) {
        var shared_slot = 0u;
        if (valid_work) {
            shared_slot = surface_cache_shared_patch_slot(gid.x, work);
            let first_patch_lane = local_idx == 0u || work.ray_index_in_patch == 0u;
            if (first_patch_lane) {
                var shared_patch_index = 0u;
                if (work.bootstrap_batch != 0u) {
                    shared_patch_index = bootstrap_indices[
                        surface_cache_bootstrap_schedule_index(
                            work.active_index,
                            counters
                        )
                    ];
                } else {
                    shared_patch_index = update_indices[
                        surface_cache_regular_schedule_index(
                            work.active_index,
                            counters
                        )
                    ];
                }
                let shared_patch = surface_cache[shared_patch_index];
                let normal = safe_normalize(shared_patch.normal_cell_exponent.xyz);
                let cell_exponent = surface_cache_grid_key_cell_exponent(
                    shared_patch.grid_key
                );
                let origin_offset = max(
                    0.001,
                    surface_cache_cell_size(cell_exponent) * 0.002
                );
                shared_patch_origin_tmin[shared_slot] = vec4<f32>(
                    shared_patch.position_frame.xyz + normal * origin_offset,
                    origin_offset * 0.25
                );
                shared_patch_normal_history[shared_slot] = vec4<f32>(
                    normal,
                    shared_patch.history.y
                );

                let seed = surface_cache_patch_rng(
                    shared_patch_index,
                    shared_patch.grid_key
                );
                var rng = random_seed(seed);
                let rotation_u = rand_float(rng);
                rng = random_seed(rng);
                shared_patch_rotation[shared_slot] = vec2<f32>(
                    rotation_u,
                    rand_float(rng)
                );
            }
        }
        workgroupBarrier();

        if (!valid_work) {
            return;
        }
        let normal_history = shared_patch_normal_history[shared_slot];
        let rotation = shared_patch_rotation[shared_slot];
        let sequence_value = f32(
            u32(normal_history.w) + work.ray_index_in_patch
        );
        let r1 = fract(rotation.x + sequence_value * 0.7548776662466927);
        let r2 = fract(rotation.y + sequence_value * 0.5698402909980532);
        let direction = sample_uniform_hemisphere_surface_cache(
            normal_history.xyz,
            r1,
            r2
        );
        trace_surface_cache_prepared_ray(
            direction,
            shared_patch_origin_tmin[shared_slot],
            work.data_index
        );
        return;
    }

    if (!valid_work) {
        return;
    }
    var patch_index = 0u;
    if (work.bootstrap_batch != 0u) {
        patch_index = bootstrap_indices[
            surface_cache_bootstrap_schedule_index(
                work.active_index,
                counters
            )
        ];
    } else {
        patch_index = update_indices[surface_cache_regular_schedule_index(
            work.active_index,
            counters
        )];
    }
    trace_surface_cache_ray(
        patch_index,
        work.ray_index_in_patch,
        work.data_index
    );
}
