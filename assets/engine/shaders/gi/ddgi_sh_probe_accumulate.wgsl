// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║              DDGI SPHERICAL HARMONICS PROBE ACCUMULATION                  ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Projects shaded ray samples onto L1 spherical harmonics per probe.       ║
// ║  This provides a compact, smooth representation of probe irradiance       ║
// ║  that interpolates naturally and is efficient for real-time sampling.     ║
// ║                                                                           ║
// ║  Key features:                                                            ║
// ║  • Monte Carlo integration of ray radiance onto SH basis                  ║
// ║  • Sample-count weighted temporal accumulation for stable convergence     ║
// ║  • Probe grid snapping support (history reprojection)                     ║
// ║  • L1 RGB representation (12 floats packed to 6 u32)                      ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read> probe_update_indices: array<u32>;
@group(1) @binding(2) var<storage, read> probe_ray_allocations: array<vec2<u32>>;
@group(1) @binding(3) var<storage, read_write> probe_ray_data: DDGIProbeRayDataBuffer;
@group(1) @binding(4) var<storage, read_write> probe_alpha: array<f32>;
@group(1) @binding(5) var<storage, read_write> sh_probes: array<u32>;
@group(1) @binding(6) var<storage, read_write> probe_states: array<ProbeStateData>;
@group(1) @binding(7) var<storage, read> gi_counters: GICountersReadOnly;

// =============================================================================
// CONSTANTS
// =============================================================================

// Monte Carlo normalization for uniform sphere sampling
// For uniform sphere: PDF = 1 / (4 * PI), so weight = 4 * PI / N
const SPHERE_AREA = 12.566370614359172; // 4 * PI

// -----------------------------------------------------------------------------
// Adaptive temporal hysteresis (no fixed MAX_ACCUMULATED_SAMPLES)
//
// We treat each probe update as ONE temporal sample (the probe already integrates
// many rays into a single Monte Carlo estimate each frame).
//
// - If probe luminance changes a lot: aggressively discard history -> fast adapt.
// - If probe luminance changes a little: keep a long effective history -> stable.
// -----------------------------------------------------------------------------
const DDGI_HISTORY_CAP_FRAMES_MIN = 1.0;    // big change -> behave like "replace"
const DDGI_HISTORY_CAP_FRAMES_MAX = 128.0;  // small change -> stable long history
const DDGI_LUMA_FAST_START = 0.60;          // relative delta where we start speeding up
const DDGI_LUMA_FAST_END = 0.90;            // relative delta where we fully speed up
const DDGI_LUMA_EPS = 1e-6;

// Variance gate for the fast-adapt path:
// High-variance probes can flicker frame-to-frame; we suppress "fast change" when
// the detected change is not statistically significant relative to ray noise.
const DDGI_NOISE_RATIO_START = 0.10; // standard_error / mean where we start suppressing fast adapt
const DDGI_NOISE_RATIO_END = 0.35;   // standard_error / mean where fast adapt is mostly suppressed
const DDGI_NOISE_SIGMA_MULTIPLIER = 2.0; // require delta > k * standard_error to be considered "real"

fn ddgi_sh_average_radiance_luma(sh: SH_L1_RGB) -> f32 {
    // For L1 SH, coefficient c0 is the projection onto Y00 (constant basis).
    // For a constant radiance field k: c0 = k * ∫Y00 dω = k * (4π * SH_BASIS_L0).
    // So average radiance ≈ c0 / (4π * SH_BASIS_L0).
    let avg_radiance = max(sh.c[0], vec3<f32>(0.0)) / (SPHERE_AREA * SH_BASIS_L0);
    return luminance(avg_radiance);
}

// =============================================================================
// MAIN COMPUTE SHADER
// One workgroup per probe, 256 threads per workgroup. Each thread handles one ray;
// subgroup (warp) reduce within each warp, then warp leaders write to shared;
// thread 0 sums the 8 warp partial sums and does hysteresis/writes.
// Depth moment updates run in ddgi_depth_update with 1 thread per ray.
// =============================================================================

const NUM_WARPS: u32 = 8u; // 256 / 32

var<workgroup> sh_c0: array<vec3<f32>, NUM_WARPS>;
var<workgroup> sh_c1: array<vec3<f32>, NUM_WARPS>;
var<workgroup> sh_c2: array<vec3<f32>, NUM_WARPS>;
var<workgroup> sh_c3: array<vec3<f32>, NUM_WARPS>;
var<workgroup> luma_vals: array<vec2<f32>, NUM_WARPS>;

@compute @workgroup_size(256, 1, 1)
fn cs(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(subgroup_size) warp_size: u32,
    @builtin(subgroup_invocation_id) lane_id: u32
) {
    let active_probe_count = gi_counters.probe_update_count;
    if (wg_id.x >= active_probe_count) {
        return;
    }

    let probe_index = probe_update_indices[wg_id.x];
    let allocation = probe_ray_allocations[wg_id.x];
    let ray_base = allocation.x;
    let rays_per_probe = max(1u, allocation.y);
    let n = f32(rays_per_probe);
    let sample_weight = SPHERE_AREA / n;

    let i = local_id.x;
    var sample_sh: SH_L1_RGB;
    var sample_luma = 0.0;
    var sample_luma_sq = 0.0;

    if (i < rays_per_probe) {
        let ray_index = ray_base + i;
        let radiance = probe_ray_data.rays[ray_index].radiance.xyz;
        let ray_dir = probe_ray_data.rays[ray_index].ray_dir_prim.xyz;
        sample_sh = ddgi_sh_project_sample(ray_dir, radiance, sample_weight);
        sample_luma = luminance(radiance);
        sample_luma_sq = sample_luma * sample_luma;
    } else {
        sample_sh = sh_l1_rgb_zero();
    }

    let warp_id = local_id.x / warp_size;
    let warp_ctx = make_warp_ctx(local_id.x, lane_id, warp_size);

    let reduced_c0 = vec3<f32>(
        warp_reduce_add_f32(warp_ctx, sample_sh.c[0].x),
        warp_reduce_add_f32(warp_ctx, sample_sh.c[0].y),
        warp_reduce_add_f32(warp_ctx, sample_sh.c[0].z)
    );
    let reduced_c1 = vec3<f32>(
        warp_reduce_add_f32(warp_ctx, sample_sh.c[1].x),
        warp_reduce_add_f32(warp_ctx, sample_sh.c[1].y),
        warp_reduce_add_f32(warp_ctx, sample_sh.c[1].z)
    );
    let reduced_c2 = vec3<f32>(
        warp_reduce_add_f32(warp_ctx, sample_sh.c[2].x),
        warp_reduce_add_f32(warp_ctx, sample_sh.c[2].y),
        warp_reduce_add_f32(warp_ctx, sample_sh.c[2].z)
    );
    let reduced_c3 = vec3<f32>(
        warp_reduce_add_f32(warp_ctx, sample_sh.c[3].x),
        warp_reduce_add_f32(warp_ctx, sample_sh.c[3].y),
        warp_reduce_add_f32(warp_ctx, sample_sh.c[3].z)
    );
    let reduced_luma = vec2<f32>(
        warp_reduce_add_f32(warp_ctx, sample_luma),
        warp_reduce_add_f32(warp_ctx, sample_luma_sq)
    );

    if (is_warp_leader(warp_ctx)) {
        sh_c0[warp_id] = reduced_c0;
        sh_c1[warp_id] = reduced_c1;
        sh_c2[warp_id] = reduced_c2;
        sh_c3[warp_id] = reduced_c3;
        luma_vals[warp_id] = reduced_luma;
    }

    workgroupBarrier();

    if (local_index != 0u) {
        return;
    }

    var sh_new: SH_L1_RGB;
    sh_new.c[0] = vec3<f32>(0.0);
    sh_new.c[1] = vec3<f32>(0.0);
    sh_new.c[2] = vec3<f32>(0.0);
    sh_new.c[3] = vec3<f32>(0.0);
    var luma_sum = 0.0;
    var luma_sum_sq = 0.0;
    for (var w = 0u; w < NUM_WARPS; w = w + 1u) {
        sh_new.c[0] += sh_c0[w];
        sh_new.c[1] += sh_c1[w];
        sh_new.c[2] += sh_c2[w];
        sh_new.c[3] += sh_c3[w];
        luma_sum += luma_vals[w].x;
        luma_sum_sq += luma_vals[w].y;
    }

    let sh_prev = ddgi_sh_probe_read(&sh_probes, probe_index);
    let prev_sample_count = ddgi_probe_state_get_sample_count(&probe_states[probe_index]);

    let probe_state = probe_state_get_state(probe_states[probe_index].packed_state);
    let is_newly_state = probe_state_is_newly(probe_state);

    let luma_prev = ddgi_sh_average_radiance_luma(sh_prev);
    let luma_new = ddgi_sh_average_radiance_luma(sh_new);
    let luma_ref = max(max(luma_prev, luma_new), DDGI_LUMA_EPS);
    let relative_luma_delta = abs(luma_new - luma_prev) / luma_ref;

    let luma_mean = luma_sum / n;
    let luma_var = max(luma_sum_sq / n - luma_mean * luma_mean, 0.0);
    let inv_sqrt_n = inverseSqrt(n);
    let luma_std_err = sqrt(luma_var) * inv_sqrt_n;
    let luma_mean_ref = max(luma_mean, DDGI_LUMA_EPS);
    let noise_ratio = luma_std_err / luma_mean_ref;

    let noise_suppression = 1.0 - smoothstep(DDGI_NOISE_RATIO_START, DDGI_NOISE_RATIO_END, noise_ratio);
    let sigma_threshold = (DDGI_NOISE_SIGMA_MULTIPLIER * luma_std_err) / luma_ref;
    let significant_delta = max(relative_luma_delta - sigma_threshold, 0.0);

    var change_factor = smoothstep(DDGI_LUMA_FAST_START, DDGI_LUMA_FAST_END, significant_delta) * noise_suppression;
    change_factor = select(change_factor, 1.0, is_newly_state);

    let prev_frames = min(f32(prev_sample_count), DDGI_HISTORY_CAP_FRAMES_MAX);
    let history_cap_frames = mix(DDGI_HISTORY_CAP_FRAMES_MAX, DDGI_HISTORY_CAP_FRAMES_MIN, change_factor);
    let retained_frames = min(prev_frames * (1.0 - change_factor), history_cap_frames - 1.0);
    let accumulated_frames = retained_frames + 1.0;
    let alpha = 1.0 / accumulated_frames;

    let sh_result = sh_l1_rgb_lerp(sh_prev, sh_new, alpha);

    probe_alpha[probe_index] = alpha;
    ddgi_sh_probe_write(&sh_probes, probe_index, sh_result);
    ddgi_probe_state_set_sample_count(&probe_states[probe_index], u32(clamp(accumulated_frames, 1.0, DDGI_HISTORY_CAP_FRAMES_MAX)));
}
