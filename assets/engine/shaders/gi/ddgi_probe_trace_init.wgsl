// =============================================================================
// DDGI Probe Ray Trace - Init Pass
// - Generates per-ray directions (guided by last frame's probe SH)
// - Writes per-ray direction-space PDF for unbiased accumulation
// - Initializes the per-ray hit buffer to a known default state
// =============================================================================
#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read> probe_update_indices: array<u32>;
@group(1) @binding(2) var<storage, read> probe_ray_allocations: array<vec2<u32>>; // x=ray_base, y=ray_count
@group(1) @binding(3) var<storage, read_write> probe_ray_data: DDGIProbeRayDataBuffer;
@group(1) @binding(4) var<storage, read> gi_counters: GICountersReadOnly;

// =============================================================================
// Ray direction sampling
// =============================================================================
const ddgi_uniform_sphere_pdf: f32 = 0.07957747154594767; // 1 / (4 * PI)

fn ddgi_fibonacci_sphere_direction(ray_index: u32, ray_count: u32, rotation_01: f32) -> vec3<f32> {
    let n = max(ray_count, 1u);
    let i = min(ray_index, n - 1u);

    // Stratified latitude, uniform in cos(theta) for uniform area on the sphere.
    let u = (f32(i) + 0.5) / f32(n);           // (0,1)
    let cos_theta = 1.0 - 2.0 * u;             // [-1,1]
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));

    // Fibonacci spiral azimuth with a per-probe Cranley-Patterson rotation.
    let phi = 2.0 * PI * fract(f32(i) * GOLDEN_RATIO_CONJUGATE + rotation_01);
    return vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

fn ddgi_probe_ray_direction_spherical_fibonacci(
    probe_index: u32,
    ray_index_in_probe: u32,
    rays_per_probe: u32
) -> vec3<f32> {
    // One stochastic rotation per probe per frame (shared across all rays in the probe).
    var probe_rng = hash(probe_index ^ (u32(ddgi_params.frame_index) * 0x9E3779B9u));
    let rotation_01 = rand_float(probe_rng);

    // Randomly rotate the entire point set in 3D (avoid locking the pattern to world axes).
    probe_rng = random_seed(probe_rng);
    let r1 = rand_float(probe_rng);
    probe_rng = random_seed(probe_rng);
    let r2 = rand_float(probe_rng);

    let z = 1.0 - 2.0 * r1;
    let rot_phi = 2.0 * PI * r2;
    let r_xy = sqrt(max(1.0 - z * z, 0.0));
    let z_axis = vec3<f32>(cos(rot_phi) * r_xy, sin(rot_phi) * r_xy, z);

    let dir_local = ddgi_fibonacci_sphere_direction(ray_index_in_probe, rays_per_probe, rotation_01);
    return orthonormalize(z_axis) * dir_local;
}

// =============================================================================
// Main
// =============================================================================
@compute @workgroup_size(16, 16, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let probe_slot = gid.y;
    let ray_index_in_probe = gid.x;
    let active_probe_count = gi_counters.probe_update_count;
    let max_probes_per_frame = u32(ddgi_params.probe_counts.z);

    if (probe_slot >= max_probes_per_frame) {
        return;
    }

    if (probe_slot >= active_probe_count) {
        return;
    }

    let allocation = probe_ray_allocations[probe_slot];
    let ray_base = allocation.x;
    let rays_per_probe = allocation.y;

    if (ray_index_in_probe >= rays_per_probe) {
        return;
    }

    let probe_index = probe_update_indices[probe_slot];
    let ray_index = ray_base + ray_index_in_probe;

    probe_ray_data.rays[ray_index].state_u32 = vec4<u32>(INVALID_IDX, 1u, 0u, INVALID_IDX);
    probe_ray_data.rays[ray_index].hit_pos_t = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    probe_ray_data.rays[ray_index].nee_light_dir_type = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    probe_ray_data.rays[ray_index].nee_light_radiance = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    probe_ray_data.rays[ray_index].radiance = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    probe_ray_data.rays[ray_index].meta_u32 = vec4<u32>(probe_index, ray_index_in_probe, probe_slot, 0u);

    let uniform_ray_dir = ddgi_probe_ray_direction_spherical_fibonacci(
        probe_index,
        ray_index_in_probe,
        rays_per_probe
    );

    probe_ray_data.rays[ray_index].ray_dir_prim = vec4<f32>(uniform_ray_dir, ddgi_uniform_sphere_pdf);
}
