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
@group(1) @binding(2) var<storage, read_write> probe_ray_data: DDGIProbeRayDataBuffer;
@group(1) @binding(3) var<storage, read_write> sh_probes: array<u32>;
@group(1) @binding(4) var<storage, read_write> sample_counts: array<u32>;
@group(1) @binding(5) var<storage, read_write> probe_depth_moments: array<vec4<f32>>;
@group(1) @binding(6) var<storage, read> probe_states: array<ProbeStateData>;
@group(1) @binding(7) var<storage, read_write> gi_counters: GICounters;

// =============================================================================
// CONSTANTS
// =============================================================================

// Monte Carlo normalization for uniform sphere sampling
// For uniform sphere: PDF = 1 / (4 * PI), so weight = 4 * PI / N
const SPHERE_AREA = 12.566370614359172; // 4 * PI

// Depth moments update
const DDGI_DEPTH_SAMPLE_COUNT_CAP = 64.0;

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
// =============================================================================

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    // ─────────────────────────────────────────────────────────────────────────
    // Early exit if we're beyond the number of probes to update this frame
    // ─────────────────────────────────────────────────────────────────────────
    let max_probes_per_frame = u32(ddgi_params.probe_counts.z);
    let active_probe_count = atomicLoad(&gi_counters.probe_update_count);
    let rays_per_probe = u32(ddgi_params.probe_counts.y);
    
    if (gid.x >= active_probe_count) {
        return;
    }
    
    let probe_index = probe_update_indices[gid.x];
    let ray_base = gid.x * rays_per_probe;
    
    // ─────────────────────────────────────────────────────────────────────────
    // Compute probe grid reprojection for snapped grids
    // When the camera moves, we need to reproject history from shifted coords
    // ─────────────────────────────────────────────────────────────────────────
    let dims = vec3<i32>(
        i32(ddgi_params.probe_grid_dims.x),
        i32(ddgi_params.probe_grid_dims.y),
        i32(ddgi_params.probe_grid_dims.z)
    );
    
    let dst_coord = ddgi_probe_coord_from_index(&ddgi_params, probe_index);

    let depth_base = probe_index * DDGI_DEPTH_TEXEL_COUNT;

    // -------------------------------------------------------------------------
    // Warm start (copy/reproject history into current for THIS probe)
    // -------------------------------------------------------------------------
    var sh_prev = ddgi_sh_probe_read(&sh_probes, probe_index);
    var prev_sample_count = sample_counts[probe_index];
    
    // ─────────────────────────────────────────────────────────────────────────
    // Project all ray samples onto SH basis
    // Monte Carlo integration: E[L(ω)] ≈ (1/N) Σ L(ω_i) * Y(ω_i) * (4π)
    // where 4π is the sphere surface area (normalization for uniform sampling)
    // ─────────────────────────────────────────────────────────────────────────
    var sh_new = sh_l1_rgb_zero();
    var luma_sum = 0.0;
    var luma_sum_sq = 0.0;

    for (var i = 0u; i < rays_per_probe; i = i + 1u) {
        let ray_index = ray_base + i;
        let hit_data = probe_ray_data.rays[ray_index];
        let radiance = probe_ray_data.rays[ray_index].radiance.xyz;
        let ray_dir = hit_data.ray_dir_prim.xyz;
        
        // Project onto SH basis
        // Weight: 4π/N for uniform sphere sampling Monte Carlo integration
        let sample_weight = SPHERE_AREA / f32(rays_per_probe);
        let sample_sh = ddgi_sh_project_sample(ray_dir, radiance, sample_weight);
        
        sh_new = sh_l1_rgb_add(sh_new, sample_sh);

        let spacing = ddgi_probe_spacing_from_index(&ddgi_params, probe_index);
        let max_dim = max(
            ddgi_params.probe_grid_dims.x,
            max(ddgi_params.probe_grid_dims.y, ddgi_params.probe_grid_dims.z)
        );
        let miss_distance = max(1.0, spacing * max_dim * 2.0);

        let t_raw = hit_data.hit_pos_t.w;
        let is_valid_hit = hit_data.state_u32.w != INVALID_IDX;
        let t = min(select(miss_distance, abs(t_raw), is_valid_hit && t_raw > 0.0), miss_distance);
        let t2 = t * t;

        let uv = encode_octahedral(safe_normalize(ray_dir));
        let tx = min(u32(uv.x * f32(DDGI_PROBE_DEPTH_RES)), DDGI_PROBE_DEPTH_RES - 1u);
        let ty = min(u32(uv.y * f32(DDGI_PROBE_DEPTH_RES)), DDGI_PROBE_DEPTH_RES - 1u);
        let texel_id = tx + ty * DDGI_PROBE_DEPTH_RES;

        let depth_idx = depth_base + texel_id;
        var moments = probe_depth_moments[depth_idx];

        let new_count = min(moments.w + 1.0, DDGI_DEPTH_SAMPLE_COUNT_CAP);

        // Online update: mean <- mean + (x - mean) / n
        let alpha = 1.0 / max(new_count, 1.0);
        moments.x += (t - moments.x) * alpha;
        moments.y += (t2 - moments.y) * alpha;

        let prev_min_d = select(t, moments.z, moments.w > 0.0);
        moments.z = min(prev_min_d, t);
        moments.w = new_count;

        probe_depth_moments[depth_idx] = moments;

        // ---------------------------------------------------------------------
        // Luminance statistics for variance gating
        // ---------------------------------------------------------------------
        let sample_luma = luminance(radiance);
        luma_sum += sample_luma;
        luma_sum_sq += sample_luma * sample_luma;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Probe state-based hysteresis override
    // "Newly" states (NEWLY_AWAKE, NEWLY_VIGILANT) use fast convergence
    // ─────────────────────────────────────────────────────────────────────────
    let probe_state = probe_state_get_state(probe_states[probe_index].packed_state);
    let is_newly_state = probe_state_is_newly(probe_state);

    // Luminance-driven hysteresis:
    // - change_factor = 0 -> keep long history (slow updates)
    // - change_factor = 1 -> discard history (fast updates)
    let luma_prev = ddgi_sh_average_radiance_luma(sh_prev);
    let luma_new = ddgi_sh_average_radiance_luma(sh_new);
    let luma_ref = max(max(luma_prev, luma_new), DDGI_LUMA_EPS);
    let relative_luma_delta = abs(luma_new - luma_prev) / luma_ref;

    // Estimate standard error of the *new* radiance luminance from per-ray variance.
    // This is a cheap proxy for "how noisy is this probe update?"
    let n = f32(rays_per_probe);
    let luma_mean = luma_sum / n;
    let luma_var = max(luma_sum_sq / n - luma_mean * luma_mean, 0.0);
    let luma_std_err = sqrt(luma_var) / sqrt(n);
    let luma_mean_ref = max(luma_mean, DDGI_LUMA_EPS);
    let noise_ratio = luma_std_err / luma_mean_ref;

    // 1) Suppress fast-adapt when variance is high.
    let noise_suppression = 1.0 - smoothstep(DDGI_NOISE_RATIO_START, DDGI_NOISE_RATIO_END, noise_ratio);

    // 2) Also require the delta to exceed a sigma-based noise floor.
    let sigma_threshold = (DDGI_NOISE_SIGMA_MULTIPLIER * luma_std_err) / luma_ref;
    let significant_delta = max(relative_luma_delta - sigma_threshold, 0.0);

    var change_factor = smoothstep(DDGI_LUMA_FAST_START, DDGI_LUMA_FAST_END, significant_delta) * noise_suppression;

    // ─────────────────────────────────────────────────────────────────────────
    // For "Newly" states, force fast convergence (high change factor)
    // This effectively sets hysteresis to 0, discarding history
    // ─────────────────────────────────────────────────────────────────────────
    change_factor = select(change_factor, 1.0, is_newly_state);

    let prev_frames = min(f32(prev_sample_count), DDGI_HISTORY_CAP_FRAMES_MAX);
    let history_cap_frames = mix(DDGI_HISTORY_CAP_FRAMES_MAX, DDGI_HISTORY_CAP_FRAMES_MIN, change_factor);
    let retained_frames = min(prev_frames * (1.0 - change_factor), history_cap_frames - 1.0);

    // alpha = 1 / (retained_frames + 1) because the new estimate is one temporal sample.
    let accumulated_frames = retained_frames + 1.0;
    let alpha = 1.0 / accumulated_frames;
    
    // Blend new SH with history using weighted average
    let sh_result = sh_l1_rgb_lerp(sh_prev, sh_new, alpha);
    
    // ─────────────────────────────────────────────────────────────────────────
    // Write results to output buffers
    // ─────────────────────────────────────────────────────────────────────────
    ddgi_sh_probe_write(&sh_probes, probe_index, sh_result);
    sample_counts[probe_index] = u32(clamp(accumulated_frames, 1.0, DDGI_HISTORY_CAP_FRAMES_MAX));
}
