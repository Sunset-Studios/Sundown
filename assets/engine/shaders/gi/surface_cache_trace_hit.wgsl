#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "gi/surface_cache_common.wgsl"
#include "acceleration/bvh_ray_traversal.wgsl"

@group(1) @binding(0) var<uniform> surface_cache_params: SurfaceCacheParams;
@group(1) @binding(1) var<storage, read_write> surface_cache: array<SurfacePatchReadOnly>;
@group(1) @binding(2) var<storage, read_write> surface_cache_sh: array<u32>;
@group(1) @binding(3) var<storage, read> active_indices: array<u32>;
@group(1) @binding(4) var<storage, read> counters: SurfaceCacheCountersReadOnly;
@group(1) @binding(5) var<storage, read_write> hit_info: array<SurfaceCacheHitInfo>;
@group(1) @binding(6) var<storage, read> tlas_bvh2_bounds: array<AABB>;
@group(1) @binding(7) var<uniform> tlas_bvh_info: BVHInfo;
@group(1) @binding(8) var<storage, read> blas_bvh2_nodes: array<AABB>;
@group(1) @binding(9) var<storage, read> blas_directory: array<MeshDirectoryEntry>;
@group(1) @binding(10) var<storage, read> compact_transforms: array<RayInstanceTransform>;
@group(1) @binding(11) var<storage, read> index_buffer: array<u32>;
@group(1) @binding(12) var<storage, read> entity_index_lookup: array<u32>;

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
    let patch_index = active_indices[active_index];
    let surface_patch = surface_cache[patch_index];
    let normal = safe_normalize(surface_patch.normal_lod.xyz);
    let seed = surface_cache_patch_rng(patch_index, surface_patch.grid_key);
    let ray_sample = generate_ray_sample(
        seed,
        normal,
        u32(surface_patch.history.y) + ray_index_in_patch
    );
    let direction = ray_sample.direction;

    var ray: Ray;
    let lod = surface_cache_grid_key_lod(surface_patch.grid_key);
    let origin_offset = max(
        0.001,
        surface_cache_lod_cell_size(lod, surface_cache_params) * 0.002
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

    // A negative primitive lane is the miss marker consumed by shade.
    hit_info[ray_data_index].hit_position_sampling_weight = vec4<f32>(
        0.0,
        0.0,
        0.0,
        ray_sample.sampling_weight
    );
    hit_info[ray_data_index].ray_direction_primitive = vec4<f32>(direction, -1.0);
    hit_info[ray_data_index].normal_section_index = vec4<f32>(normal, 0.0);
    hit_info[ray_data_index].hit_attr0 = vec4<f32>(0.0);
    hit_info[ray_data_index].hit_attr1 = vec4<f32>(0.0);
    hit_info[ray_data_index].shadow_origin = vec4<f32>(0.0);
    hit_info[ray_data_index].shadow_direction = vec4<f32>(0.0);

    let hit_result = trace_ray_closest(&ray);
    if (hit_result.has_hit == 0u) {
        return;
    }

    let prim_store = hit_result.prim_store;
    let entity_resolved = entity_index_lookup[prim_store];
    let instance_transform = compact_transforms[entity_resolved];
    var ray_local = build_local_ray_from_instance(&ray, instance_transform);
    let hit_position_local = ray_local.origin_and_tmin.xyz
        + ray_local.direction_and_tmax.xyz * hit_result.t_hit;
    let hit_position_world = transform_local_point_from_instance(
        instance_transform,
        hit_position_local
    );

    let vertex0 = decode_vertex(vertex_buffer[hit_result.tri_indices.x]);
    let vertex1 = decode_vertex(vertex_buffer[hit_result.tri_indices.y]);
    let vertex2 = decode_vertex(vertex_buffer[hit_result.tri_indices.z]);
    let edge0 = vertex1.position.xyz - vertex0.position.xyz;
    let edge1 = vertex2.position.xyz - vertex0.position.xyz;
    let vertex_delta = hit_position_local - vertex0.position.xyz;
    let d00 = dot(edge0, edge0);
    let d01 = dot(edge0, edge1);
    let d11 = dot(edge1, edge1);
    let d20 = dot(vertex_delta, edge0);
    let d21 = dot(vertex_delta, edge1);
    let denominator = max(d00 * d11 - d01 * d01, 1e-8);
    let bary_v = (d00 * d21 - d01 * d20) / denominator;
    let bary_u = (d11 * d20 - d01 * d21) / denominator;
    let bary_w = 1.0 - bary_u - bary_v;

    let uv = vertex0.uv * bary_w + vertex1.uv * bary_u + vertex2.uv * bary_v;
    let normal_local = vertex0.normal.xyz * bary_w
        + vertex1.normal.xyz * bary_u
        + vertex2.normal.xyz * bary_v;
    let tangent_local = vertex0.tangent.xyz * bary_w
        + vertex1.tangent.xyz * bary_u
        + vertex2.tangent.xyz * bary_v;
    let bitangent_local = vertex0.bitangent.xyz * bary_w
        + vertex1.bitangent.xyz * bary_u
        + vertex2.bitangent.xyz * bary_v;

    var normal_world = safe_normalize(transform_local_direction_from_instance(
        instance_transform,
        normal_local
    ));
    var tangent_world = safe_normalize(transform_local_direction_from_instance(
        instance_transform,
        tangent_local
    ));
    var bitangent_world = safe_normalize(transform_local_direction_from_instance(
        instance_transform,
        bitangent_local
    ));
    let backfacing = dot(normal_world, direction) > 0.0;
    normal_world = select(normal_world, -normal_world, backfacing);
    tangent_world = select(tangent_world, -tangent_world, backfacing);
    bitangent_world = select(bitangent_world, -bitangent_world, backfacing);

    hit_info[ray_data_index].hit_position_sampling_weight = vec4<f32>(
        hit_position_world,
        ray_sample.sampling_weight
    );
    hit_info[ray_data_index].ray_direction_primitive = vec4<f32>(
        direction,
        f32(prim_store)
    );
    hit_info[ray_data_index].normal_section_index = vec4<f32>(
        normal_world,
        vertex0.section_index
    );
    hit_info[ray_data_index].hit_attr0 = vec4<f32>(tangent_world, uv.x);
    hit_info[ray_data_index].hit_attr1 = vec4<f32>(bitangent_world, uv.y);
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
    if (active_index >= counters.active_patch_count) {
        return;
    }
    trace_surface_cache_ray(
        active_index,
        ray_data_index % rays_per_patch,
        ray_data_index
    );
}
