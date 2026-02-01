// =============================================================================
// DDGI Probe Ray Trace - Init Pass
// - Generates per-ray directions (guided by last frame's probe SH)
// - Writes per-ray direction-space PDF for unbiased accumulation
// - Initializes the per-ray hit buffer to a known default state
//
// NOTE:
// We split this out of ddgi_probe_trace_hit.wgsl to stay under WebGPU's
// per-stage storage buffer binding limits.
// =============================================================================
#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read> probe_update_indices: array<u32>;
@group(1) @binding(2) var<storage, read_write> probe_ray_data: DDGIProbeRayDataBuffer;
@group(1) @binding(3) var<storage, read> sh_probes: array<u32>;
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
@compute @workgroup_size(256, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rays_per_probe = u32(ddgi_params.probe_counts.y);
    let total_rays = gi_counters.probe_update_count * rays_per_probe;
    let max_rays = u32(ddgi_params.probe_counts.z) * rays_per_probe;

    if (gid.x >= max_rays) {
        return;
    }

    if (gid.x >= total_rays) {
        probe_ray_data.rays[gid.x].state_u32 = vec4<u32>(INVALID_IDX, 0u, 0u, INVALID_IDX);
        probe_ray_data.rays[gid.x].meta_u32 = vec4<u32>(INVALID_IDX, 0u, 0u, 0u);
        return;
    }

    let probe_slot = gid.x / rays_per_probe;
    let ray_index_in_probe = gid.x - probe_slot * rays_per_probe;
    let probe_index = probe_update_indices[probe_slot];

    // -------------------------------------------------------------------------
    // Initialize hit data to a known default state
    // -------------------------------------------------------------------------
    // state_u32:
    // - x = prim_store (filled by hit pass, 0xffffffff if miss/uninitialized)
    // - y = alive
    // - z = shadow_visible (set by hit pass)
    // - w = tri_id_local (filled by hit pass, 0xffffffff if miss)
    probe_ray_data.rays[gid.x].state_u32 = vec4<u32>(INVALID_IDX, 1u, 0u, INVALID_IDX);
    probe_ray_data.rays[gid.x].hit_pos_t = vec4f(0.0, 0.0, 0.0, 0.0);
    probe_ray_data.rays[gid.x].radiance = vec4f(0.0, 0.0, 0.0, 1.0);
    probe_ray_data.rays[gid.x].meta_u32 = vec4<u32>(probe_index, 0u, 0u, 0u);

    // -------------------------------------------------------------------------
    // Uniform directions (base proposal)
    // -------------------------------------------------------------------------
    let uniform_ray_dir = ddgi_probe_ray_direction_spherical_fibonacci(
        probe_index,
        ray_index_in_probe,
        rays_per_probe
    );

    // Store direction + per-ray PDF in ray_dir_prim.
    // ray_dir_prim.w holds the PDF until the hit pass fills other attributes.
    probe_ray_data.rays[gid.x].ray_dir_prim = vec4f(uniform_ray_dir, ddgi_uniform_sphere_pdf);

    // Track active rays in the header (reset in ddgi_probe_ray_data_header_reset.wgsl).
    atomicAdd(&probe_ray_data.header.active_ray_count, 1u);
}

