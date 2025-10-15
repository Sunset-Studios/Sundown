// =============================================================================
// GI-1.0 Screen Probe Reconstruction
// - Interpolates radiance from screen probes to final pixels
// - Uses distance and normal-based weighting similar to GI-1.0 paper
// - Outputs final indirect lighting texture
// =============================================================================
#include "common.wgsl"

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

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read> screen_probes: array<ScreenProbe>;
@group(1) @binding(2) var<storage, read> screen_probe_counter: array<u32>;
@group(1) @binding(3) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(4) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(5) var gbuffer_albedo: texture_2d<f32>;
@group(1) @binding(6) var output_gi: texture_storage_2d<rgba16float, write>;

// Compute weight for a probe contribution
fn compute_probe_weight(
    pixel_pos: vec3<f32>,
    pixel_normal: vec3<f32>,
    probe_pos: vec3<f32>,
    probe_normal: vec3<f32>,
    probe_radius: f32
) -> f32 {
    // Distance-based weight
    let dist = length(pixel_pos - probe_pos);
    let dist_weight = max(0.0, 1.0 - dist / max(0.001, probe_radius));
    
    // Normal similarity weight
    let normal_dot = max(0.0, dot(pixel_normal, probe_normal));
    let normal_weight = pow(normal_dot, 4.0); // Higher power for sharper falloff
    
    // View direction weight (prefer probes in front of surface)
    let to_probe = normalize(probe_pos - pixel_pos);
    let view_weight = max(0.0, dot(pixel_normal, to_probe));
    
    return dist_weight * normal_weight * view_weight;
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(output_gi);
    
    if (gid.x >= res.x || gid.y >= res.y) {
        return;
    }
    
    let pixel_coord = vec2<i32>(i32(gid.x), i32(gid.y));
    
    // Read G-buffer for this pixel
    let position = textureLoad(gbuffer_position, pixel_coord, 0).xyz;
    let normal_data = textureLoad(gbuffer_normal, pixel_coord, 0);
    let normal = safe_normalize(normal_data.xyz);
    let normal_length = length(normal_data.xyz);
    
    // If no geometry, output black
    if (normal_length < 0.01) {
        textureStore(output_gi, pixel_coord, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }
    
    let albedo = textureLoad(gbuffer_albedo, pixel_coord, 0).rgb;
    
    // Interpolate radiance from nearby screen probes
    let probe_count = screen_probe_counter[0];
    var total_radiance = vec3<f32>(0.0);
    var total_weight = 0.0;
    
    // Search for nearby probes
    // For now, do a linear search (could be optimized with spatial acceleration)
    let max_search_radius = 50.0; // World units
    let max_probes_to_check = min(probe_count, 64u); // Limit for performance
    
    for (var i = 0u; i < max_probes_to_check; i = i + 1u) {
        let probe = screen_probes[i];
        
        // Check if probe is active
        if (probe.state.x == 0u) {
            continue;
        }
        
        let probe_pos = probe.position_radius.xyz;
        let probe_radius = probe.position_radius.w;
        let probe_normal = probe.normal_frame.xyz;
        let probe_radiance = probe.radiance_m.xyz;
        
        // Quick distance cull
        let dist = length(position - probe_pos);
        if (dist > max_search_radius) {
            continue;
        }
        
        // Compute probe weight
        let weight = compute_probe_weight(
            position,
            normal,
            probe_pos,
            probe_normal,
            probe_radius * 10.0 // Scale radius for wider influence
        );
        
        if (weight > 0.001) {
            total_radiance += probe_radiance * weight;
            total_weight += weight;
        }
    }
    
    // Normalize and apply albedo
    var final_radiance = vec3<f32>(0.0);
    if (total_weight > 0.001) {
        let interpolated_radiance = total_radiance / total_weight;
        
        // Apply indirect boost
        let indirect_boost = bitcast<f32>(gi_params.indirect_boost);
        final_radiance = interpolated_radiance * albedo * indirect_boost;
    }
    
    // Clamp to reasonable range
    final_radiance = clamp(final_radiance, vec3<f32>(0.0), vec3<f32>(10.0));
    
    textureStore(output_gi, pixel_coord, vec4<f32>(final_radiance, 1.0));
}

