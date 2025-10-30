// =============================================================================
// GI-1.0 Screen Probe Update
// - Accumulates radiance from traced rays back to screen probes
// - Updates world cache with secondary bounce radiance
// - Performs temporal filtering with exponential moving average
// - Only updates probes marked active this frame
// =============================================================================
#include "common.wgsl"
#include "lighting_common.wgsl"
#include "gi/gi_common.wgsl"
#include "gi/world_cache_common.wgsl"

struct ProbePathState {
    origin_tmin: vec4<f32>,
    direction_tmax: vec4<f32>,
    normal_section_index: vec4<f32>,
    state_u32: vec4<u32>,      // x=bounce, y=alive, z=unused, w=tri_id
    hit_attr0: vec4<f32>,
    hit_attr1: vec4<f32>,
    shadow_origin: vec4<f32>,
    shadow_direction: vec4<f32>,
    shadow_radiance: vec4<f32>,
};

struct ProbePathShade {
    path_weight: vec4<f32>,
    rng_sample_count_frame_stamp: vec4<f32>,
    throughput: vec4<f32>,
    reservoir_radiance_m: vec4<f32>,
    reservoir_direction_w: vec4<f32>,
};

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read_write> gi_counters: GICounters;
@group(1) @binding(2) var<storage, read_write> screen_probes: array<ScreenProbe>;
@group(1) @binding(3) var<storage, read> probe_path_state: array<ProbePathState>;
@group(1) @binding(4) var<storage, read> probe_path_shade: array<ProbePathShade>;
@group(1) @binding(5) var<storage, read_write> world_cache: array<WorldCacheCell>;

// =============================================================================
// Biased Temporal Hysteresis (GI-1.0 Algorithm 3)
// - Adapts blend factor based on luminance difference
// - Preserves shadows and occlusion better than exponential moving average
// - Acts as firefly removal by filtering out transient bright signals
// =============================================================================
fn temporal_blend(curr_radiance: vec3<f32>, prev_radiance: vec3<f32>) -> vec3<f32> {
    // Compute luminance using equal weighting (1/3, 1/3, 1/3)
    let l1 = dot(curr_radiance, vec3<f32>(1.0 / 3.0));
    let l2 = dot(prev_radiance, vec3<f32>(1.0 / 3.0));
    
    // Compute adaptive alpha based on normalized difference
    // Bias towards darker values to preserve shadows
    let numerator = max(l1 - l2 - min(l1, l2), 0.0);
    let denominator = max(max(l1, l2), 1e-4);
    var alpha = numerator / denominator;
    
    // Clamp and remap with squared falloff
    alpha = clamp(alpha, 0.0, 0.95);
    alpha = alpha * alpha;
    
    // Blend: higher alpha = more previous radiance (temporal stability)
    return mix(curr_radiance, prev_radiance, alpha);
}

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Process all grid probes (derived from resolution)
    let probe_count = u32(gi_params.total_screen_probes);
    
    if (gid.x >= probe_count) {
        return;
    }
    
    // Only update probes that are active (were spawned/updated this frame)
    let probe = screen_probes[gid.x];
    if (probe.state.x == 0.0) {
        return; // Skip inactive probes
    }
    
    let rays_per_probe = u32(gi_params.screen_ray_count);
    
    // Accumulate radiance from all rays for this probe
    var accumulated_radiance = vec3<f32>(0.0);
    var valid_ray_count = 0u;
    
    // World cache parameters
    let world_cache_size = u32(gi_params.world_cache_size);
    let cell_size = 1.0;
    
    for (var i = 0u; i < rays_per_probe; i = i + 1u) {
        let ray_id = gid.x * rays_per_probe + i;
        let path = probe_path_state[ray_id];
        let shade = probe_path_shade[ray_id];
        
        let radiance = shade.throughput.rgb;
        
        // Accumulate ALL rays (even if zero) to increment M properly
        // This ensures probes show as "blue" in debug even if radiance is zero
        accumulated_radiance += radiance;
        valid_ray_count += 1u;
        
        // === Update World Cache for Secondary Bounces ===
        // Insert radiance at secondary hit points to enable reuse across probes
        let bounce = path.state_u32.x;
        let tri_id = path.state_u32.w;
        
        if (bounce > 0u && tri_id != 0xffffffffu && length(radiance) > 0.001) {
            let hit_pos = path.origin_tmin.xyz;
            let hit_normal = path.normal_section_index.xyz;
            
            // Compute outgoing radiance at this point
            // Divide by path weight PDF to get unbiased radiance estimate
            let outgoing_radiance = radiance / max(shade.path_weight.w, 0.001);
            
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
    
    // Average radiance across valid rays (protect against divide by zero)
    if (valid_ray_count > 0u) {
        accumulated_radiance /= f32(valid_ray_count);
    }
    
    // Retrieve previous radiance for temporal blending
    let prev_radiance = screen_probes[gid.x].radiance_m.xyz;
    let prev_sample_count = screen_probes[gid.x].radiance_m.w;
    
    // Apply biased temporal hysteresis
    // For newly spawned probes (M=0), prev_radiance is 0, so blend naturally starts at current
    // For reprojected probes, blend with accumulated history
    var final_radiance = accumulated_radiance;
    // if (prev_sample_count > 0.0 && valid_ray_count > 0u) {
    //     final_radiance = temporal_blend(accumulated_radiance, prev_radiance);
    // }
    
    // Update probe with blended radiance
    screen_probes[gid.x].radiance_m = vec4<f32>(
        final_radiance,
        prev_sample_count + f32(valid_ray_count)
    );
    screen_probes[gid.x].normal_frame.w = gi_params.frame_index;
}

