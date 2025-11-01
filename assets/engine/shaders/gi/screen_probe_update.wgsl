// =============================================================================
// GI-1.0 Screen Probe Update
// - Accumulates radiance from traced rays back to screen probes
// - Updates world cache with secondary bounce radiance
// - Performs temporal filtering with exponential moving average
// - Only updates probes marked active this frame
// =============================================================================
#include "common.wgsl"
#include "gi/gi_common.wgsl"
#include "gi/world_cache_common.wgsl"

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read_write> gi_counters: GICounters;
@group(1) @binding(2) var<storage, read_write> screen_probes: array<ScreenProbe>;
@group(1) @binding(3) var<storage, read> probe_path_state: array<ProbePathState>;
@group(1) @binding(4) var<storage, read> probe_path_shade: array<ProbePathShade>;
@group(1) @binding(5) var<storage, read_write> world_cache: array<WorldCacheCell>;

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Process all grid probes (derived from resolution)
    let probe_count = u32(gi_params.total_screen_probes);
    
    if (gid.x >= probe_count) {
        return;
    }
    
    // Only update probes that are active (were spawned/updated this frame)
    let probe = screen_probes[gid.x];
    if (probe.state.x == 0.0 || probe.state.w == 0.0) {
        return; // Skip inactive probes
    }
    
    let rays_per_probe = u32(gi_params.screen_ray_count);
    
    // Accumulate radiance from all rays for this probe
    var accumulated_radiance = vec3<f32>(0.0);
    
    // World cache parameters
    let world_cache_size = u32(gi_params.world_cache_size);
    let cell_size = 1.0;
    
    for (var i = 0u; i < rays_per_probe; i = i + 1u) {
        let ray_id = gid.x * rays_per_probe + i;
        let path = probe_path_state[ray_id];
        let shade = probe_path_shade[ray_id];

        let sample_count = max(path.rng_sample_count_frame_stamp.y, 1.0);
        let accumulated_avg = path.throughput.xyz / sample_count;
        let radiance = safe_clamp_vec3(accumulated_avg);
        
        // Accumulate ALL rays (even if zero) to increment M properly
        accumulated_radiance += radiance;
        
        // === Update World Cache for Secondary Bounces ===
        // Insert radiance at secondary hit points to enable reuse across probes
        let bounce = path.state_u32.x;
        let tri_id = path.state_u32.w;
        
        if (bounce > 0u && tri_id != 0xffffffffu) {
            let hit_pos = path.origin_tmin.xyz;
            let hit_normal = path.normal_section_index.xyz;
            
            // Compute outgoing radiance at this point
            // Divide by path weight PDF to get unbiased radiance estimate
            let outgoing_radiance = radiance / max(path.path_weight.w, 0.001);
            
            // Validate before inserting
            if (!isinf(outgoing_radiance.x) && !isinf(outgoing_radiance.y) && !isinf(outgoing_radiance.z)) {
                let inserted = insert_world_cache(
                    hit_pos,
                    hit_normal,
                    outgoing_radiance,
                    &world_cache,
                    world_cache_size,
                    cell_size,
                    u32(gi_params.frame_index)
                );
            }
        }
    }
    
    // =========================================================================
    // Temporal blend with biased hysteresis (GI-1.0 Algorithm 3)
    // - Adapts based on luminance difference
    // - Preserves shadows and removes fireflies
    // =========================================================================
    let prev_radiance = max(screen_probes[gid.x].radiance_m.xyz, vec3<f32>(0.0));
    let prev_sample_count = max(screen_probes[gid.x].radiance_m.w, 1.0);
    
    // Apply temporal blend for stable, shadow-preserving accumulation
    let blended_radiance = temporal_blend(accumulated_radiance + prev_radiance, prev_radiance);
    
    // Update sample count to track total accumulated samples
    screen_probes[gid.x].radiance_m = vec4<f32>(blended_radiance, prev_sample_count + f32(rays_per_probe));
    screen_probes[gid.x].normal_frame.w = gi_params.frame_index;
}