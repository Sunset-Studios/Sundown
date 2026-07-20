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
@group(1) @binding(3) var<storage, read_write> probe_history_valid: array<f32>;
@group(1) @binding(4) var<storage, read_write> sh_probes: array<u32>;
@group(1) @binding(5) var<storage, read_write> probe_states: array<ProbeStateData>;
@group(1) @binding(6) var<storage, read_write> probe_msme_stats: array<DDGIMSMEProbeStats>;
@group(1) @binding(7) var<storage, read> gi_counters: GICountersReadOnly;

// =============================================================================
// CONSTANTS
// =============================================================================

// Monte Carlo normalization for uniform sphere sampling
// For uniform sphere: PDF = 1 / (4 * PI), so weight = 4 * PI / N
const SPHERE_AREA = 12.566370614359172; // 4 * PI

// -----------------------------------------------------------------------------
// MSME-style temporal irradiance accumulation. Visibility uses a separate,
// more conservative policy in ddgi_depth_update.
// -----------------------------------------------------------------------------
// Keep the long and short estimators far enough apart for the short mean to
// detect changes without imposing its noise floor on the stable result.
const DDGI_HISTORY_CAP_FRAMES_MAX = 64.0;
const DDGI_MSME_SHORT_WINDOW_FRAMES = 8.0;
const DDGI_MSME_MIN_SIGNAL_ENERGY = 1e-5;
const DDGI_MSME_VARIANCE_FORGIVENESS = 2.0;
const DDGI_MSME_VARIANCE_BLEND_REDUCTION = 12.0;
const DDGI_MSME_INCONSISTENCY_LOW = 0.025;
const DDGI_MSME_INCONSISTENCY_HIGH = 0.18;
const DDGI_MSME_INCONSISTENCY_RISE_ALPHA = 0.45;
const DDGI_MSME_INCONSISTENCY_FALL_ALPHA = 0.08;
const DDGI_MSME_CATCHUP_ALPHA_MIN = 0.04;
const DDGI_MSME_CATCHUP_ALPHA_MAX = 0.40;
const DDGI_MSME_INSTANT_CATCHUP_BOOST_MAX = 0.34;
const DDGI_MSME_INSTANT_SHORT_ALPHA_MAX = 0.34;
const DDGI_MSME_INSTANT_INCONSISTENCY_LOW = 0.10;
const DDGI_MSME_INSTANT_INCONSISTENCY_HIGH = 0.40;
const DDGI_MSME_NOISY_CATCHUP_SCALE_MIN = 0.35;

fn ddgi_sh_l1_rgb_mean_square(sh: SH_L1_RGB) -> f32 {
    var sum = 0.0;
    for (var c = 0u; c < 4u; c = c + 1u) {
        sum += dot(sh.c[c], sh.c[c]);
    }
    return max(sum / 12.0, 0.0);
}

fn ddgi_sh_l1_rgb_distance_square(a: SH_L1_RGB, b: SH_L1_RGB) -> f32 {
    var sum = 0.0;
    for (var c = 0u; c < 4u; c = c + 1u) {
        let d = a.c[c] - b.c[c];
        sum += dot(d, d);
    }
    return max(sum / 12.0, 0.0);
}

fn ddgi_sh_l1_rgb_component_mean(sh: SH_L1_RGB) -> f32 {
    var sum = 0.0;
    for (var c = 0u; c < 4u; c = c + 1u) {
        sum += sh.c[c].x + sh.c[c].y + sh.c[c].z;
    }
    return max(sum / 12.0, 0.0);
}

fn ddgi_sh_l1_rgb_square_delta(a: SH_L1_RGB, b: SH_L1_RGB) -> SH_L1_RGB {
    var result: SH_L1_RGB;
    for (var c = 0u; c < 4u; c = c + 1u) {
        let d = a.c[c] - b.c[c];
        result.c[c] = d * d;
    }
    return result;
}

fn ddgi_msme_stats_short_mean(stats: DDGIMSMEProbeStats) -> SH_L1_RGB {
    var sh: SH_L1_RGB;
    sh.c[0] = stats.short_mean_c0.xyz;
    sh.c[1] = stats.short_mean_c1.xyz;
    sh.c[2] = stats.short_mean_c2.xyz;
    sh.c[3] = stats.short_mean_c3.xyz;
    return sh;
}

fn ddgi_msme_stats_variance(stats: DDGIMSMEProbeStats) -> SH_L1_RGB {
    var sh: SH_L1_RGB;
    sh.c[0] = max(stats.variance_c0.xyz, vec3<f32>(0.0));
    sh.c[1] = max(stats.variance_c1.xyz, vec3<f32>(0.0));
    sh.c[2] = max(stats.variance_c2.xyz, vec3<f32>(0.0));
    sh.c[3] = max(stats.variance_c3.xyz, vec3<f32>(0.0));
    return sh;
}

fn ddgi_msme_stats_write(
    probe_index: u32,
    short_mean: SH_L1_RGB,
    variance: SH_L1_RGB,
    scalars: vec4<f32>
) {
    probe_msme_stats[probe_index].short_mean_c0 = vec4<f32>(short_mean.c[0], 0.0);
    probe_msme_stats[probe_index].short_mean_c1 = vec4<f32>(short_mean.c[1], 0.0);
    probe_msme_stats[probe_index].short_mean_c2 = vec4<f32>(short_mean.c[2], 0.0);
    probe_msme_stats[probe_index].short_mean_c3 = vec4<f32>(short_mean.c[3], 0.0);
    probe_msme_stats[probe_index].variance_c0 = vec4<f32>(variance.c[0], 0.0);
    probe_msme_stats[probe_index].variance_c1 = vec4<f32>(variance.c[1], 0.0);
    probe_msme_stats[probe_index].variance_c2 = vec4<f32>(variance.c[2], 0.0);
    probe_msme_stats[probe_index].variance_c3 = vec4<f32>(variance.c[3], 0.0);
    probe_msme_stats[probe_index].scalars = scalars;
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
    let rays_per_probe = ddgi_max_rays_per_probe(&ddgi_params);
    let ray_base = wg_id.x * rays_per_probe;
    let n = f32(rays_per_probe);
    let sample_weight = SPHERE_AREA / n;

    let i = local_id.x;
    var sample_sh: SH_L1_RGB;
    if (i < rays_per_probe) {
        let ray_index = ray_base + i;
        let radiance = probe_ray_data.rays[ray_index].radiance.xyz;
        let ray_dir = probe_ray_data.rays[ray_index].ray_dir_prim.xyz;
        sample_sh = ddgi_sh_project_sample(ray_dir, radiance, sample_weight);
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
    if (is_warp_leader(warp_ctx)) {
        sh_c0[warp_id] = reduced_c0;
        sh_c1[warp_id] = reduced_c1;
        sh_c2[warp_id] = reduced_c2;
        sh_c3[warp_id] = reduced_c3;
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
    for (var w = 0u; w < NUM_WARPS; w = w + 1u) {
        sh_new.c[0] += sh_c0[w];
        sh_new.c[1] += sh_c1[w];
        sh_new.c[2] += sh_c2[w];
        sh_new.c[3] += sh_c3[w];
    }

    let sh_prev = ddgi_sh_probe_read(&sh_probes, probe_index);
    let prev_sample_count = ddgi_probe_state_get_sample_count(&probe_states[probe_index]);

    if (prev_sample_count == 0u) {
        let zero_variance = sh_l1_rgb_zero();
        ddgi_msme_stats_write(
            probe_index,
            sh_new,
            zero_variance,
            vec4<f32>(1.0, 0.0, 0.0, 1.0)
        );
        probe_history_valid[probe_index] = 0.0;
        ddgi_sh_probe_write(&sh_probes, probe_index, sh_new);
        ddgi_probe_state_set_sample_count(&probe_states[probe_index], 1u);
        return;
    }

    let stats_prev = probe_msme_stats[probe_index];
    let has_msme_history = stats_prev.scalars.x > 0.0;
    var short_prev = ddgi_msme_stats_short_mean(stats_prev);
    if (!has_msme_history) {
        short_prev = sh_prev;
    }

    let prev_short_frames = select(
        1.0,
        min(stats_prev.scalars.x, DDGI_MSME_SHORT_WINDOW_FRAMES),
        has_msme_history
    );
    let short_frames = min(prev_short_frames + 1.0, DDGI_MSME_SHORT_WINDOW_FRAMES);
    let short_alpha = 1.0 / max(short_frames, 1.0);
    let variance_alpha = min(short_alpha * 1.5, 1.0);

    let prev_variance = ddgi_msme_stats_variance(stats_prev);
    let pre_signal_energy = max(
        max(ddgi_sh_l1_rgb_mean_square(sh_prev), ddgi_sh_l1_rgb_mean_square(short_prev)),
        max(ddgi_sh_l1_rgb_mean_square(sh_new), DDGI_MSME_MIN_SIGNAL_ENERGY)
    );
    let prev_normalized_variance = ddgi_sh_l1_rgb_component_mean(prev_variance) / pre_signal_energy;
    let normalized_sample_long_delta = ddgi_sh_l1_rgb_distance_square(sh_new, sh_prev) / pre_signal_energy;
    let instant_inconsistency = max(
        normalized_sample_long_delta - prev_normalized_variance * DDGI_MSME_VARIANCE_FORGIVENESS,
        0.0
    );
    let instant_change_weight = smoothstep(
        DDGI_MSME_INSTANT_INCONSISTENCY_LOW,
        DDGI_MSME_INSTANT_INCONSISTENCY_HIGH,
        instant_inconsistency
    );
    let adaptive_short_alpha = mix(
        short_alpha,
        max(short_alpha, DDGI_MSME_INSTANT_SHORT_ALPHA_MAX),
        instant_change_weight
    );

    let short_mean = sh_l1_rgb_lerp(short_prev, sh_new, adaptive_short_alpha);
    let sample_variance = ddgi_sh_l1_rgb_square_delta(sh_new, short_prev);
    let variance = sh_l1_rgb_lerp(prev_variance, sample_variance, variance_alpha);

    let signal_energy = max(
        max(ddgi_sh_l1_rgb_mean_square(sh_prev), ddgi_sh_l1_rgb_mean_square(short_mean)),
        max(ddgi_sh_l1_rgb_mean_square(sh_new), DDGI_MSME_MIN_SIGNAL_ENERGY)
    );
    let normalized_variance = ddgi_sh_l1_rgb_component_mean(variance) / signal_energy;
    let normalized_short_long_delta = ddgi_sh_l1_rgb_distance_square(short_mean, sh_prev) / signal_energy;
    let raw_inconsistency = max(
        normalized_short_long_delta - normalized_variance * DDGI_MSME_VARIANCE_FORGIVENESS,
        0.0
    );
    let inconsistency_alpha = select(
        DDGI_MSME_INCONSISTENCY_FALL_ALPHA,
        DDGI_MSME_INCONSISTENCY_RISE_ALPHA,
        raw_inconsistency > stats_prev.scalars.z
    );
    let inconsistency = mix(stats_prev.scalars.z, raw_inconsistency, inconsistency_alpha);
    let change_weight = smoothstep(
        DDGI_MSME_INCONSISTENCY_LOW,
        DDGI_MSME_INCONSISTENCY_HIGH,
        inconsistency
    );
    let vbbr = 1.0 / (1.0 + normalized_variance * DDGI_MSME_VARIANCE_BLEND_REDUCTION);
    let instant_catchup_boost = instant_change_weight * (1.0 - change_weight);
    let resolved_change_weight = clamp(
        change_weight + instant_catchup_boost * DDGI_MSME_INSTANT_CATCHUP_BOOST_MAX,
        0.0,
        1.0
    );
    let noise_guard = mix(DDGI_MSME_NOISY_CATCHUP_SCALE_MIN, 1.0, clamp(vbbr, 0.0, 1.0));

    let prev_frames = min(f32(prev_sample_count), DDGI_HISTORY_CAP_FRAMES_MAX);
    let stable_frames = min(prev_frames + 1.0, DDGI_HISTORY_CAP_FRAMES_MAX);
    let stable_alpha = 1.0 / max(stable_frames, 1.0);
    let catchup_alpha = mix(
        DDGI_MSME_CATCHUP_ALPHA_MIN,
        DDGI_MSME_CATCHUP_ALPHA_MAX,
        resolved_change_weight
    ) * noise_guard;

    // Preserve an unbiased, sample-count-weighted mean while lighting is
    // stable. Reducing this alpha based on variance makes the first noisy
    // estimate dominate even though sample_count continues to advance. MSME's
    // variance estimate instead belongs on the change-response path: when the
    // short and long means disagree beyond expected noise, converge rapidly
    // toward the short mean; otherwise keep integrating the new observation.
    let stable_result = sh_l1_rgb_lerp(sh_prev, sh_new, stable_alpha);
    let catchup_result = sh_l1_rgb_lerp(sh_prev, short_mean, catchup_alpha);
    let sh_result = sh_l1_rgb_lerp(stable_result, catchup_result, resolved_change_weight);

    ddgi_msme_stats_write(
        probe_index,
        short_mean,
        variance,
        vec4<f32>(short_frames, normalized_variance, inconsistency, vbbr)
    );
    probe_history_valid[probe_index] = 1.0;

    ddgi_sh_probe_write(&sh_probes, probe_index, sh_result);
    ddgi_probe_state_set_sample_count(&probe_states[probe_index], u32(clamp(stable_frames, 1.0, DDGI_HISTORY_CAP_FRAMES_MAX)));
}
