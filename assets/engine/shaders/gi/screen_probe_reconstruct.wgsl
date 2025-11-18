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
    
    // -------------------------------------------------------------------------
    // Interpolate radiance from nearby screen probes using grid structure
    // -------------------------------------------------------------------------
    let probe_count = u32(gi_params.total_screen_probes);
    var total_radiance = vec3<f32>(0.0);
    var total_weight = 0.0;
    
    // Probe grid layout (matches debug visualization and update pass)
    let probe_size = u32(gi_params.screen_probe_size);
    let grid_width = (res.x + probe_size - 1u) / probe_size;
    let grid_height = (res.y + probe_size - 1u) / probe_size;
    
    // Calculate which grid cell this pixel is in
    let pixel_grid_x = gid.x / probe_size;
    let pixel_grid_y = gid.y / probe_size;
    
    // Search radius in grid cells (1 means check 3x3 neighborhood)
    let grid_search_radius = 1i;
    
    // Iterate through nearby grid cells only
    for (var dy = -grid_search_radius; dy <= grid_search_radius; dy = dy + 1) {
        for (var dx = -grid_search_radius; dx <= grid_search_radius; dx = dx + 1) {
            // Calculate neighbor grid position
            let neighbor_grid_x = i32(pixel_grid_x) + dx;
            let neighbor_grid_y = i32(pixel_grid_y) + dy;
            
            // Bounds check
            if (neighbor_grid_x < 0 || neighbor_grid_y < 0 || 
                neighbor_grid_x >= i32(grid_width) || neighbor_grid_y >= i32(grid_height)) {
                continue;
            }
            
            // Compute probe index directly from grid position
            let probe_index = u32(neighbor_grid_y) * grid_width + u32(neighbor_grid_x);
            
            // Bounds check against actual probe count
            if (probe_index >= probe_count) {
                continue;
            }
            
            // Fetch probe metadata
            let probe = screen_probe_metadata[probe_index];
            
            // Check if probe is active
            if (probe.state.x == 0.0) {
                continue;
            }
            
            // Get actual probe pixel position (may differ slightly from grid center)
            let probe_pixel = vec2<u32>(probe.state.yz);
            let pixel_dist = length(vec2<f32>(gid.xy) - vec2<f32>(probe_pixel));
            
            // Look up this probe's radiance from the atlas (center texel)
            let probe_tile = vec2<u32>(u32(neighbor_grid_x), u32(neighbor_grid_y));
            let atlas_center_offset = probe_size / 2u;
            let atlas_coord = probe_tile * probe_size + atlas_center_offset;
            let radiance_sample = textureLoad(probe_radiance_curr, vec2<i32>(atlas_coord), 0);
            let probe_radiance = radiance_sample.rgb / max(radiance_sample.w, 1.0);
            
            // Distance-based weight (inverse square falloff)
            let weight = 1.0 / (1.0 + pixel_dist * pixel_dist);
            total_radiance += probe_radiance * weight;
            total_weight += weight;
        }
    }
    
    // Normalize and apply albedo
    var final_radiance = vec3<f32>(0.0);
    if (total_weight > 0.001) {
        final_radiance = total_radiance / total_weight;
    }
    
    textureStore(output_gi, pixel_coord, vec4<f32>(final_radiance, 1.0));
}

