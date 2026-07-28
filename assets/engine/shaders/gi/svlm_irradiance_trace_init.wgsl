#include "gi/svlm_common.wgsl"

// Initializes one bounded batch of probe rays. Probe positions remain implicit
// in the leaf-brick 4x4x4 lattice; only transient ray records are materialized.

@group(1) @binding(0) var<storage, read_write> svlm_params: SVLMParams;
@group(1) @binding(1) var<storage, read_write> svlm_counters: SVLMCounters;
@group(1) @binding(2) var<storage, read> leaf_bricks: array<SVLMLeafBrick>;
@group(1) @binding(3) var<storage, read_write> ray_data: SVLMProbeRayDataBuffer;

const SVLM_GOLDEN_RATIO_CONJUGATE = 0.6180339887498948;

fn svlm_fibonacci_sphere_direction(
    ray_index: u32,
    ray_count: u32,
    rotation_01: f32
) -> vec3<f32> {
    let count = max(ray_count, 1u);
    let sample_index = min(ray_index, count - 1u);
    let u = (f32(sample_index) + 0.5) / f32(count);
    let cos_theta = 1.0 - 2.0 * u;
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let phi = 2.0 * PI * fract(
        f32(sample_index) * SVLM_GOLDEN_RATIO_CONJUGATE + rotation_01
    );
    return vec3<f32>(
        cos(phi) * sin_theta,
        sin(phi) * sin_theta,
        cos_theta
    );
}

fn svlm_probe_ray_direction(
    probe_position: vec3<f32>,
    ray_index: u32,
    ray_count: u32,
    sample_index: u32
) -> vec3<f32> {
    let position_seed =
        bitcast<u32>(probe_position.x) ^
        (bitcast<u32>(probe_position.y) * 0x9e3779b9u) ^
        (bitcast<u32>(probe_position.z) * 0x85ebca6bu);
    var rng = hash(
        position_seed ^
        (sample_index * 0x9e3779b9u) ^
        (u32(svlm_params.bake_serial) * 0xa511e9b3u)
    );
    let rotation_01 = rand_float(rng);
    rng = random_seed(rng);
    let r1 = rand_float(rng);
    rng = random_seed(rng);
    let r2 = rand_float(rng);

    let z = 1.0 - 2.0 * r1;
    let phi = 2.0 * PI * r2;
    let radius_xy = sqrt(max(1.0 - z * z, 0.0));
    let rotation_axis = vec3<f32>(
        cos(phi) * radius_xy,
        sin(phi) * radius_xy,
        z
    );
    let local_direction = svlm_fibonacci_sphere_direction(
        ray_index,
        ray_count,
        rotation_01
    );
    return orthonormalize(rotation_axis) * local_direction;
}

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rays_per_probe = max(1u, u32(svlm_params.irradiance_rays_per_probe));
    let probes_per_batch = max(
        1u,
        u32(svlm_params.irradiance_probes_per_batch)
    );
    let probe_count = min(
        atomicLoad(&svlm_counters.probe_count),
        arrayLength(&leaf_bricks) * SVLM_PROBES_PER_BRICK
    );
    let cursor = min(
        atomicLoad(&svlm_counters.irradiance_cursor),
        probe_count
    );
    let sample_index = atomicLoad(&svlm_counters.irradiance_sample_index);
    let target_sample_count = max(
        1u,
        u32(svlm_params.irradiance_sample_count)
    );
    let active_probe_count = select(
        min(probes_per_batch, probe_count - cursor),
        0u,
        sample_index >= target_sample_count
    );

    if (gid.x == 0u) {
        atomicStore(
            &ray_data.header.active_ray_count,
            active_probe_count * rays_per_probe
        );
    }

    let ray_index = gid.x;
    if (ray_index >= active_probe_count * rays_per_probe) {
        return;
    }

    let probe_slot = ray_index / rays_per_probe;
    let ray_index_in_probe = ray_index % rays_per_probe;
    let probe_index = cursor + probe_slot;
    let leaf_index = probe_index / SVLM_PROBES_PER_BRICK;
    let local_probe_index = probe_index % SVLM_PROBES_PER_BRICK;
    if (leaf_index >= arrayLength(&leaf_bricks)) {
        return;
    }

    if (ray_index >= arrayLength(&ray_data.rays)) {
        return;
    }

    let probe_position = svlm_probe_position(
        leaf_bricks[leaf_index],
        local_probe_index
    );
    let direction = svlm_probe_ray_direction(
        probe_position,
        ray_index_in_probe,
        rays_per_probe,
        sample_index
    );

    ray_data.rays[ray_index].hit_payload_t = vec4<f32>(probe_position, 0.0);
    ray_data.rays[ray_index].ray_direction = vec4<f32>(direction, 0.0);
    ray_data.rays[ray_index].nee_light_radiance = vec4<f32>(0.0);
    ray_data.rays[ray_index].state_u32 = vec4<u32>(
        INVALID_IDX,
        1u,
        0u,
        INVALID_IDX
    );
    ray_data.rays[ray_index].radiance = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    ray_data.rays[ray_index].meta_u32 = vec4<u32>(
        probe_index,
        ray_index_in_probe,
        sample_index,
        bitcast<u32>(max(svlm_params.irradiance_max_ray_distance, 1.0))
    );
}
