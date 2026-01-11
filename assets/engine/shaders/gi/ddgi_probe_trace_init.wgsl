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
@group(1) @binding(2) var<storage, read_write> probe_ray_hits: array<DDGIProbeRayHit>;
@group(1) @binding(3) var<storage, read> sh_probes: array<u32>;
@group(1) @binding(4) var<storage, read> sample_counts: array<u32>;

// =============================================================================
// Ray direction sampling
// =============================================================================
const ddgi_uniform_sphere_pdf: f32 = 0.07957747154594767; // 1 / (4 * PI)

// Guidance tuning (stable defaults for low sample counts).
const ddgi_guided_p_max: f32 = 0.25;
const ddgi_guided_anisotropy_to_p: f32 = 0.25;
const ddgi_guided_kappa_scale: f32 = 4.0;
const ddgi_guided_kappa_max: f32 = 8.0;

// History gating: scale guidance up only when history is stable.
const ddgi_guided_history_full_batches: u32 = 8u;

fn ddgi_vmf_pdf(mu: vec3<f32>, kappa: f32, w: vec3<f32>) -> f32 {
    // vMF on S^2: p(w) = kappa / (4π sinh(kappa)) * exp(kappa * dot(mu, w))
    // For kappa→0, this approaches the uniform distribution.
    let mu_dot_w = clamp(dot(mu, w), -1.0, 1.0);
    let exp_kappa = exp(kappa);
    let exp_neg_kappa = exp(-kappa);
    let sinh_kappa = 0.5 * (exp_kappa - exp_neg_kappa);
    let denom = 4.0 * PI * max(sinh_kappa, 1e-6);
    let base = (kappa / denom) * exp(kappa * mu_dot_w);
    return select(base, ddgi_uniform_sphere_pdf, kappa < 1e-3);
}

fn ddgi_vmf_sample_direction(mu: vec3<f32>, kappa: f32, u1: f32, u2: f32) -> vec3<f32> {
    // Sample the vMF distribution by sampling t = dot(mu, w) and a uniform azimuth.
    let exp_kappa = exp(kappa);
    let exp_neg_kappa = exp(-kappa);
    let t = (1.0 / max(kappa, 1e-6)) * log(exp_neg_kappa + u1 * (exp_kappa - exp_neg_kappa));
    let sin_theta = sqrt(max(1.0 - t * t, 0.0));
    let phi = 2.0 * PI * u2;

    // Local frame assumes +Z is the lobe axis.
    let dir_local = vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, t);
    return orthonormalize(mu) * dir_local;
}

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
@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rays_per_probe = u32(ddgi_params.probe_counts.y);
    let probes_per_frame = u32(ddgi_params.probe_counts.z);
    let total_rays = probes_per_frame * rays_per_probe;

    if (gid.x >= total_rays) {
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
    probe_ray_hits[gid.x].state_u32 = vec4<u32>(0xffffffffu, 1u, 0u, 0xffffffffu);
    probe_ray_hits[gid.x].hit_pos_t = vec4f(0.0, 0.0, 0.0, -1.0);

    // -------------------------------------------------------------------------
    // Uniform directions (base proposal)
    // -------------------------------------------------------------------------
    let uniform_ray_dir = ddgi_probe_ray_direction_spherical_fibonacci(
        probe_index,
        ray_index_in_probe,
        rays_per_probe
    );

    // -------------------------------------------------------------------------
    // Guided directions from last frame's probe SH (mixture importance sampling)
    // -------------------------------------------------------------------------
    let probe_sh = ddgi_sh_probe_read(&sh_probes, probe_index);
    let lum = vec3<f32>(0.2126, 0.7152, 0.0722);
    let l0 = dot(probe_sh.c[0], lum);
    let l0_abs = max(abs(l0), 1e-4);
    let l1_vec = vec3<f32>(
        dot(probe_sh.c[3], lum),
        dot(probe_sh.c[1], lum),
        dot(probe_sh.c[2], lum)
    );
    let l1_len = length(l1_vec);
    let anisotropy = l1_len / l0_abs;

    let guided_valid = l1_len > 1e-4;
    let anisotropy_clamped = min(anisotropy, 1.0);

    // Confidence ramp (prevents guidance from thrashing when history is noisy/uninitialized).
    // We don't have direct access to the reprojection mapping here, so this is a simple,
    // robust heuristic: scale by per-probe history batch count stored in sample_counts.
    // (bound by the caller)
    let prev_batches = min(sample_counts[probe_index], ddgi_guided_history_full_batches);
    var history_confidence = f32(prev_batches) / f32(ddgi_guided_history_full_batches);
    // If the probe grid snapped this frame, history at this index corresponds to a different
    // world-space location. Disable guidance to avoid temporal instability.
    let snap_active = ddgi_params.probe_grid_snap_delta.w != 0.0;
    history_confidence = history_confidence * select(1.0, 0.0, snap_active);

    let p_guided_raw = clamp(anisotropy_clamped * ddgi_guided_anisotropy_to_p, 0.0, ddgi_guided_p_max);
    let kappa_raw = clamp(anisotropy_clamped * ddgi_guided_kappa_scale, 0.0, ddgi_guided_kappa_max);

    let p_guided = select(0.0, p_guided_raw * history_confidence, guided_valid);
    let kappa = select(0.0, kappa_raw * history_confidence, guided_valid);
    let mu = safe_normalize(l1_vec);

    // Stratify the mixture selection (avoid per-ray random "mode flips" with low ray counts).
    let guided_ray_count = u32(round(p_guided * f32(rays_per_probe)));
    let do_guided = ray_index_in_probe < min(guided_ray_count, rays_per_probe);

    // Per-ray sample point in [0,1)^2 for the guided lobe (stable hash per ray).
    var ray_rng = hash(
        probe_index
            ^ (ray_index_in_probe * 0xA24BAEDDu)
            ^ (u32(ddgi_params.frame_index) * 0x9E3779B9u)
    );
    ray_rng = random_seed(ray_rng);
    let u1 = rand_float(ray_rng);
    ray_rng = random_seed(ray_rng);
    let u2 = rand_float(ray_rng);
    let guided_ray_dir = ddgi_vmf_sample_direction(mu, kappa, u1, u2);
    let ray_dir = select(uniform_ray_dir, guided_ray_dir, do_guided);

    // Mixture PDF for the chosen direction.
    let guided_pdf = ddgi_vmf_pdf(mu, kappa, ray_dir);
    let pdf = (1.0 - p_guided) * ddgi_uniform_sphere_pdf + p_guided * guided_pdf;

    // Store direction + per-ray PDF in ray_dir_prim.
    // ray_dir_prim.w holds the PDF until the hit pass fills other attributes.
    probe_ray_hits[gid.x].ray_dir_prim = vec4f(ray_dir, pdf);
}

