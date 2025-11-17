// =============================================================================
// GI-1.0 Screen Probe Debug Visualization
// - Renders a visual representation of the screen probe grid
// - Shows probe irradiance, visibility, and coverage
// - Dispatches one thread per probe, each probe writes to its influenced pixels
// =============================================================================
#include "common.wgsl"
#include "gi/gi_common.wgsl"

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read_write> gi_counters: GICounters;
@group(1) @binding(2) var<storage, read> screen_probe_metadata: array<ScreenProbe>;
@group(1) @binding(3) var probe_radiance_curr: texture_2d<f32>;
@group(1) @binding(4) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(5) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(6) var scene_color: texture_2d<f32>;
@group(1) @binding(7) var output_debug: texture_storage_2d<rgba16float, write>;

const DEBUG_PROBE_RADIUS: f32 = 2.5;

// =============================================================================
// Probe Grid Visualization
// =============================================================================

@compute @workgroup_size(64, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let probe_count = u32(gi_params.total_screen_probes);
    
    if (gid.x >= probe_count) {
        return;
    }
    
    let probe_index = gid.x;
    let probe = screen_probe_metadata[probe_index];
    
    // Get texture dimensions
    let res = textureDimensions(output_debug);
    let probe_size = u32(gi_params.screen_probe_size);
    let grid_width = (res.x + probe_size - 1u) / probe_size;
    
    // Calculate probe tile coordinates
    let probe_tile_x = probe_index % grid_width;
    let probe_tile_y = probe_index / grid_width;
    let probe_tile = vec2<u32>(probe_tile_x, probe_tile_y);
    
    // Calculate probe tile center for visualization
    let tile_corner = vec2<u32>(probe_tile_x * probe_size, probe_tile_y * probe_size);
    let probe_center = vec2<f32>(tile_corner) + vec2<f32>(f32(probe_size) * 0.5);
    
    // Calculate bounding box for pixels to write (small circle)
    let min_pixel = vec2<i32>(
        max(0, i32(probe_center.x - gi_params.screen_probe_size * 0.5)),
        max(0, i32(probe_center.y - gi_params.screen_probe_size * 0.5))
    );
    let max_pixel = vec2<i32>(
        min(i32(res.x), i32(probe_center.x + gi_params.screen_probe_size * 0.5) + 1),
        min(i32(res.y), i32(probe_center.y + gi_params.screen_probe_size * 0.5) + 1)
    );

    // Determine probe visualization color
    let probe_valid = probe.state.x > 0.0;
    var probe_radiance = vec3<f32>(0.0);
    
    // Write to pixels within debug radius, compositing on top of scene
    for (var py = min_pixel.y; py < max_pixel.y; py = py + 1) {
        for (var px = min_pixel.x; px < max_pixel.x; px = px + 1) {
            let pixel = vec2<f32>(f32(px), f32(py));
            let pixel_i32 = vec2<i32>(px, py);

            if (probe_valid) {
                // Read radiance from probe atlas center
                let probe_data = textureLoad(probe_radiance_curr, pixel_i32, 0);
                probe_radiance = probe_data.rgb / max(probe_data.w, 1.0);
            } else {
                probe_radiance = vec3<f32>(1.0, 0.0, 0.0);
            }
            
            // Optionally load existing scene color for compositing
            // let scene = textureLoad(scene_color, pixel_i32, 0).rgb;
            // let final_color = mix(scene, probe_radiance, 0.8);
            
            let final_color = probe_radiance * f32(gi_params.indirect_boost);
             
            textureStore(output_debug, pixel_i32, vec4<f32>(final_color, 1.0));
        }
    }
}
