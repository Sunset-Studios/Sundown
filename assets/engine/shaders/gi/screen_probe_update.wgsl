// =============================================================================
// GI-1.0 Screen Probe Update
// - Accumulates radiance from traced rays back to screen probes
// - Updates world cache with secondary bounce radiance
// - Performs temporal filtering with exponential moving average
// =============================================================================
#include "common.wgsl"
#include "gi/world_cache_common.wgsl"

struct GIParams {
    screen_probe_spawn_rate: u32,
    screen_probe_size: u32,
    screen_ray_count: u32,
    world_cache_size: u32,
    max_screen_probes: u32,
    frame_index: u32,
    reset_caches: u32,
    indirect_boost: u32,
    upscale_x: u32,
    upscale_y: u32,
    cell_size_heuristic: u32,
    padding: u32,
};

struct ScreenProbe {
    position_radius: vec4<f32>,
    normal_frame: vec4<f32>,
    radiance_m: vec4<f32>,
    albedo_roughness: vec4<f32>,
    state: vec4<u32>,
};

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
@group(1) @binding(1) var<storage, read_write> screen_probes: array<ScreenProbe>;
@group(1) @binding(2) var<storage, read> screen_probe_counter: array<u32>;
@group(1) @binding(3) var<storage, read> probe_path_state: array<ProbePathState>;
@group(1) @binding(4) var<storage, read> probe_path_shade: array<ProbePathShade>;
@group(1) @binding(5) var<storage, read_write> world_cache: array<WorldCacheCell>;

@compute @workgroup_size(64, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let probe_count = screen_probe_counter[0];
    
    if (gid.x >= probe_count) {
        return;
    }
    
    let probe_index = gid.x;
    let rays_per_probe = gi_params.screen_ray_count;
    
    // Accumulate radiance from all rays for this probe
    var accumulated_radiance = vec3<f32>(0.0);
    var valid_ray_count = 0u;
    
    // World cache parameters
    let world_cache_size = gi_params.world_cache_size;
    let cell_size = 1.0;
    
    for (var i = 0u; i < rays_per_probe; i = i + 1u) {
        let ray_id = probe_index * rays_per_probe + i;
        let path = probe_path_state[ray_id];
        let shade = probe_path_shade[ray_id];
        
        let radiance = shade.throughput.rgb;
        
        // Only accumulate if radiance is valid (not NaN or inf)
        if (!any(isnan(radiance)) && !any(isinf(radiance)) && length(radiance) < 1000.0) {
            accumulated_radiance += radiance;
            valid_ray_count += 1u;
        }
        
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
            if (!any(isnan(outgoing_radiance)) && !any(isinf(outgoing_radiance)) && length(outgoing_radiance) < 100.0) {
                let inserted = insert_world_cache(
                    hit_pos,
                    hit_normal,
                    outgoing_radiance,
                    &world_cache,
                    world_cache_size,
                    cell_size,
                    gi_params.frame_index
                );
            }
        }
    }
    
    // Average radiance across valid rays
    if (valid_ray_count > 0u) {
        accumulated_radiance /= f32(valid_ray_count);
    }
    
    // Read current probe state
    var probe = screen_probes[probe_index];
    
    // Temporal filtering with exponential moving average
    let alpha = 0.1; // Blend factor (higher = faster response, lower = more stable)
    let old_radiance = probe.radiance_m.xyz;
    let old_m = probe.radiance_m.w;
    
    // Blend with previous frame
    let new_radiance = mix(old_radiance, accumulated_radiance, alpha);
    let new_m = old_m + f32(valid_ray_count);
    
    // Clamp to reasonable range
    let clamped_radiance = clamp(new_radiance, vec3<f32>(0.0), vec3<f32>(100.0));
    
    // Update probe
    probe.radiance_m = vec4<f32>(clamped_radiance, new_m);
    probe.normal_frame.w = f32(gi_params.frame_index);
    
    screen_probes[probe_index] = probe;
}

