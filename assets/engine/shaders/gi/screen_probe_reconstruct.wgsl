// =============================================================================
// GI-1.0 Screen Probe Reconstruction
// - Interpolates radiance from screen probes to final pixels
// - Uses distance and normal-based weighting similar to GI-1.0 paper
// - Outputs final indirect lighting texture
// =============================================================================
#include "common.wgsl"
#include "gi/gi_common.wgsl"

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read_write> gi_counters: GICounters;
@group(1) @binding(2) var<storage, read> screen_probe_metadata: array<ScreenProbe>;
@group(1) @binding(3) var probe_radiance_curr: texture_2d<f32>;
@group(1) @binding(4) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(5) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(6) var gbuffer_albedo: texture_2d<f32>;
@group(1) @binding(7) var output_gi: texture_storage_2d<rgba16float, write>;

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
    let probe_count = u32(gi_params.total_screen_probes);
    var total_radiance = vec3<f32>(0.0);
    var total_weight = 0.0;
    
    // Search for nearby probes in screen-space neighborhood
    // For temporal upscale, probes are roughly uniformly distributed at probe_size intervals
    let search_radius_screen = gi_params.screen_probe_size * 1.5; // Search within 1.5 probe tiles
    
    // Probe grid layout (matches debug visualization and update pass)
    let probe_size = u32(gi_params.screen_probe_size);
    let grid_width = (res.x + probe_size - 1u) / probe_size;
    
    // Linear search through all probes, but early exit based on screen distance
    var probes_checked = 0u;
    let max_probes_to_interpolate = 9u; // Use up to 9 nearest probes
    
    for (var i = 0u; i < probe_count && probes_checked < max_probes_to_interpolate; i = i + 1u) {
        let probe = screen_probe_metadata[i];
        
        // Check if probe is active
        if (probe.state.x == 0.0) {
            continue;
        }
        
        let probe_pixel = vec2<u32>(probe.state.yz);
        let pixel_dist = length(vec2<f32>(gid.xy) - vec2<f32>(probe_pixel));
        
        // Screen-space distance cull - only consider nearby probes
        if (pixel_dist > search_radius_screen) {
            continue;
        }
        
        // Look up this probe's radiance from the atlas (center texel)
        let probe_tile_x = i % grid_width;
        let probe_tile_y = i / grid_width;
        let probe_tile = vec2<u32>(probe_tile_x, probe_tile_y);
        let atlas_center_offset = probe_size / 2u;
        let atlas_coord = probe_tile * probe_size + atlas_center_offset;
        let radiance_sample = textureLoad(probe_radiance_curr, vec2<i32>(atlas_coord), 0);
        let probe_radiance = radiance_sample.rgb;
        
        let weight = 1.0 / (1.0 + pixel_dist * pixel_dist);
        total_radiance += probe_radiance * weight;
        total_weight += weight;
        probes_checked += 1u;
    }
    
    // Normalize and apply albedo
    var final_radiance = vec3<f32>(0.0);
    if (total_weight > 0.001) {
        let interpolated_radiance = total_radiance / total_weight;
        
        // Apply indirect boost
        final_radiance = interpolated_radiance * albedo * gi_params.indirect_boost;
    }
    
    // Clamp to reasonable range
    final_radiance = clamp(final_radiance, vec3<f32>(0.0), vec3<f32>(10.0));
    
    textureStore(output_gi, pixel_coord, vec4<f32>(final_radiance, 1.0));
}

