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
@group(1) @binding(2) var<storage, read> probe_ray_hits: array<DDGIProbeRayHit>;
@group(1) @binding(3) var<storage, read> probe_ray_radiance: array<vec4<f32>>;
@group(1) @binding(4) var<storage, read> sh_probes_prev: array<u32>;
@group(1) @binding(5) var<storage, read_write> sh_probes_curr: array<u32>;
@group(1) @binding(6) var<storage, read> sample_counts_prev: array<u32>;
@group(1) @binding(7) var<storage, read_write> sample_counts_curr: array<u32>;

// =============================================================================
// CONSTANTS
// =============================================================================

// Maximum accumulated sample count before capping
// This allows gradual adaptation to lighting changes while maintaining stability
// After reaching this cap, the effective alpha stabilizes at new_samples / max_samples
const MAX_ACCUMULATED_SAMPLES: u32 = 128u;

// Monte Carlo normalization for uniform sphere sampling
// For uniform sphere: PDF = 1 / (4 * PI), so weight = 4 * PI / N
const SPHERE_AREA: f32 = 12.566370614359172; // 4 * PI

// =============================================================================
// MAIN COMPUTE SHADER
// =============================================================================

@compute @workgroup_size(64, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    // ─────────────────────────────────────────────────────────────────────────
    // Early exit if we're beyond the number of probes to update this frame
    // ─────────────────────────────────────────────────────────────────────────
    let probes_per_frame = u32(ddgi_params.probe_counts.z);
    let rays_per_probe = u32(ddgi_params.probe_counts.y);
    
    if (gid.x >= probes_per_frame) {
        return;
    }
    
    let probe_index = probe_update_indices[gid.x];
    let shadow_ray_index = gid.x;
    let ray_base = probes_per_frame + gid.x * rays_per_probe;
    
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
    let dst_coord_i32 = vec3<i32>(dst_coord);
    let src_coord_i32 = dst_coord_i32 + vec3<i32>(ddgi_params.probe_grid_snap_delta.xyz);
    
    // Check if the source probe is within grid bounds
    let in_bounds =
        src_coord_i32.x >= 0 && src_coord_i32.x < dims.x &&
        src_coord_i32.y >= 0 && src_coord_i32.y < dims.y &&
        src_coord_i32.z >= 0 && src_coord_i32.z < dims.z;
    
    let src_coord = vec3<u32>(src_coord_i32);
    let src_probe_index = ddgi_probe_index_from_coord(&ddgi_params, src_coord);
    
    // ─────────────────────────────────────────────────────────────────────────
    // Project all ray samples onto SH basis
    // Monte Carlo integration: E[L(ω)] ≈ (1/N) Σ L(ω_i) * Y(ω_i) * (4π)
    // where 4π is the sphere surface area (normalization for uniform sampling)
    // ─────────────────────────────────────────────────────────────────────────
    var sh_new = sh_l1_rgb_zero();
    var valid_sample_count = 0u;

    // ─────────────────────────────────────────────────────────────────────────
    // 1x Shadow ray sample (NEE) per probe
    // Stored at [0..probes_per_frame) in probe_ray_hits/probe_ray_radiance.
    // ─────────────────────────────────────────────────────────────────────────
    {
        let shadow_hit_data = probe_ray_hits[shadow_ray_index];
        let shadow_radiance = probe_ray_radiance[shadow_ray_index].xyz;
        let shadow_dir = shadow_hit_data.ray_dir_prim.xyz;

        if (shadow_hit_data.state_u32.y != 0u && shadow_hit_data.state_u32.z == 1u && length(shadow_dir) >= 0.5) {
            let max_luminance = MAX_RADIANCE_LUMINANCE * 2.0;
            let lum = dot(shadow_radiance, vec3<f32>(0.2126, 0.7152, 0.0722));
            let clamped_radiance = select(
                shadow_radiance,
                shadow_radiance * (max_luminance / max(lum, 0.001)),
                lum > max_luminance
            );

            let sample_weight = SPHERE_AREA / f32(rays_per_probe + 1u);
            let sample_sh = ddgi_sh_project_sample(shadow_dir, clamped_radiance, sample_weight);
            sh_new = sh_l1_rgb_add(sh_new, sample_sh);
            valid_sample_count = valid_sample_count + 1u;
        }
    }
    
    for (var i = 0u; i < rays_per_probe; i = i + 1u) {
        let ray_index = ray_base + i;
        let hit_data = probe_ray_hits[ray_index];
        let radiance = probe_ray_radiance[ray_index].xyz;
        
        // Extract ray direction (normalized)
        let ray_dir = hit_data.ray_dir_prim.xyz;
        
        // Skip invalid rays (should be normalized)
        if (length(ray_dir) < 0.5) {
            continue;
        }
        
        // Clamp radiance to prevent fireflies in SH
        let max_luminance = MAX_RADIANCE_LUMINANCE * 2.0;
        let lum = dot(radiance, vec3<f32>(0.2126, 0.7152, 0.0722));
        let clamped_radiance = select(
            radiance,
            radiance * (max_luminance / max(lum, 0.001)),
            lum > max_luminance
        );
        
        // Project onto SH basis
        // Weight: 4π/N for uniform sphere sampling Monte Carlo integration
        let sample_weight = SPHERE_AREA / f32(rays_per_probe + 1u);
        let sample_sh = ddgi_sh_project_sample(ray_dir, clamped_radiance, sample_weight);
        
        sh_new = sh_l1_rgb_add(sh_new, sample_sh);
        valid_sample_count = valid_sample_count + 1u;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Sample-count weighted temporal accumulation
    // Uses proper weighted average: new_avg = (old_avg * old_n + new_val * new_n) / (old_n + new_n)
    // Which simplifies to: alpha = new_n / (old_n + new_n)
    // ─────────────────────────────────────────────────────────────────────────
    var sh_prev = sh_l1_rgb_zero();
    var prev_sample_count = 0u;
    
    if (in_bounds) {
        // Read history from reprojected position
        sh_prev = ddgi_sh_probe_read(&sh_probes_prev, src_probe_index);
        prev_sample_count = sample_counts_prev[src_probe_index];
    }
    
    // Compute weighted average alpha based on sample counts
    // alpha = new_samples / (old_samples + new_samples)
    let total_samples = min(prev_sample_count + valid_sample_count, MAX_ACCUMULATED_SAMPLES);
    let alpha = 1.0 / (1.0 + f32(total_samples));
    
    // Blend new SH with history using weighted average
    let sh_result = sh_l1_rgb_lerp(sh_prev, sh_new, alpha);
    
    // ─────────────────────────────────────────────────────────────────────────
    // Write results to output buffers
    // ─────────────────────────────────────────────────────────────────────────
    ddgi_sh_probe_write(&sh_probes_curr, probe_index, sh_result);
    sample_counts_curr[probe_index] = total_samples;
}

