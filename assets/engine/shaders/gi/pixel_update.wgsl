// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║               PER-PIXEL PATH TRACING - TEMPORAL ACCUMULATION              ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Accumulates per-pixel radiance samples over time:                        ║
// ║  • Temporal reprojection using motion vectors                             ║
// ║  • Geometry-aware blending (depth + normal validation)                    ║
// ║  • Adaptive sample accumulation for noise reduction                       ║
// ║  • Outputs final GI radiance                                              ║
// ║                                                                           ║
// ║  Algorithm:                                                               ║
// ║  1. Read current frame's traced radiance from path state                  ║
// ║  2. Reproject to find previous frame's accumulated value                  ║
// ║  3. Validate reprojection with depth/normal similarity                    ║
// ║  4. Blend current sample with history (adaptive alpha)                    ║
// ║  5. Output to both accumulation buffer and final GI texture               ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

#include "common.wgsl"
#include "gi/gi_common.wgsl"
#include "gi/world_cache_common.wgsl"

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read_write> gi_counters: GICounters;
@group(1) @binding(2) var<storage, read> pixel_path_state: array<PixelPathState>;
@group(1) @binding(3) var pixel_radiance_prev: texture_2d<f32>;
@group(1) @binding(4) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(5) var gbuffer_position_prev: texture_2d<f32>;
@group(1) @binding(6) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(7) var gbuffer_normal_prev: texture_2d<f32>;
@group(1) @binding(8) var gbuffer_motion: texture_2d<f32>;
@group(1) @binding(9) var<storage, read_write> world_cache: array<WorldCacheCell>;
@group(1) @binding(10) var pixel_radiance_curr: texture_storage_2d<rgba16float, write>;
@group(1) @binding(11) var output_gi: texture_storage_2d<rgba16float, write>;

// =============================================================================
// CONSTANTS
// =============================================================================

// Reprojection validation thresholds
const MIN_NORMAL_SIMILARITY = 0.95;
const MAX_DEPTH_RATIO = 0.1;

// Temporal blending parameters
const MIN_BLEND_ALPHA = 0.05;     // Minimum blend for stability (5%)
const MAX_BLEND_ALPHA = 0.8;      // Maximum blend for responsiveness (80%)
const VARIANCE_BOOST = 0.5;       // How much luminance difference boosts alpha

// =============================================================================
// MAIN COMPUTE SHADER
// =============================================================================

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(gbuffer_position);
    
    if (gid.x >= res.x || gid.y >= res.y) {
        return;
    }
    
    let pixel_coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let rays_per_tile = u32(gi_params.screen_ray_count);
    
    // ─────────────────────────────────────────────────────────────────────────
    // Compute tile index for this pixel (path state is indexed by tile, not pixel)
    // ─────────────────────────────────────────────────────────────────────────
    let upscale = vec2<u32>(u32(gi_params.upscale_x), u32(gi_params.upscale_y));
    let tile_x = gid.x / upscale.x;
    let tile_y = gid.y / upscale.y;
    let tile_grid_width = res.x / upscale.x;
    let tile_index = tile_y * tile_grid_width + tile_x;
    
    // ─────────────────────────────────────────────────────────────────────────
    // Read G-buffer for current pixel
    // ─────────────────────────────────────────────────────────────────────────
    let position = textureLoad(gbuffer_position, pixel_coord, 0).xyz;
    let normal_data = textureLoad(gbuffer_normal, pixel_coord, 0);
    let normal = safe_normalize(normal_data.xyz);
    let normal_length = length(normal_data.xyz);
    
    // Skip sky pixels (no geometry)
    if (normal_length < 0.01) {
        textureStore(output_gi, pixel_coord, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        textureStore(pixel_radiance_curr, pixel_coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Check if this pixel was traced this frame
    // Path state is indexed by tile - check if the traced pixel matches this one
    // ─────────────────────────────────────────────────────────────────────────
    var current_radiance = vec3<f32>(0.0);
    var current_count = 0.0;
    
    for (var i = 0u; i < rays_per_tile; i = i + 1u) {
        // Index into path state by tile, not by pixel
        let ray_id = tile_index * rays_per_tile + i;
        let path = pixel_path_state[ray_id];
        
        // Check if the traced pixel coordinates match THIS pixel
        let ray_pixel_x = u32(path.pixel_coords.x);
        let ray_pixel_y = u32(path.pixel_coords.y);
        let coords_match = ray_pixel_x == gid.x && ray_pixel_y == gid.y;
        
        if (coords_match) {
            // This tile's ray was traced at our pixel location
            let ray_was_alive = path.state_u32.y != 0u || 
                                path.state_u32.w != 0xffffffffu ||
                                length(path.throughput.xyz) > 0.0;
            
            if (ray_was_alive) {
                let sample_count = max(path.rng_sample_count_frame_stamp.y, 1.0);
                let accumulated_avg = path.throughput.xyz / sample_count;
                let radiance = safe_clamp_vec3(accumulated_avg);
                current_radiance += radiance;
                current_count += 1.0;
            }
        }
    }
    
    // Average if multiple rays per tile
    if (current_count > 0.0) {
        current_radiance /= current_count;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Temporal Reprojection
    // ─────────────────────────────────────────────────────────────────────────
    let motion_sample = textureLoad(gbuffer_motion, pixel_coord, 0);
    let pixel_velocity = motion_sample.xy * vec2<f32>(f32(res.x), f32(res.y)) * vec2<f32>(0.5, -0.5);
    
    let pixel_center = vec2<f32>(gid.xy) + 0.5;
    let pixel_prev_center = pixel_center + -pixel_velocity;
    let pixel_prev = vec2<i32>(floor(pixel_prev_center));
    
    // Check if reprojected pixel is within bounds
    var reprojection_valid = pixel_prev.x >= 0 && pixel_prev.y >= 0 && 
                             pixel_prev.x < i32(res.x) && pixel_prev.y < i32(res.y);
    
    // ─────────────────────────────────────────────────────────────────────────
    // Validate Reprojection with Geometry Checks
    // ─────────────────────────────────────────────────────────────────────────
    var prev_radiance = vec3<f32>(0.0);
    var prev_count = 0.0;
    
    if (reprojection_valid) {
        // Sample previous frame's G-buffer
        let prev_position = textureLoad(gbuffer_position_prev, pixel_prev, 0).xyz;
        let prev_normal_data = textureLoad(gbuffer_normal_prev, pixel_prev, 0);
        let prev_normal = safe_normalize(prev_normal_data.xyz);
        
        // Depth similarity check
        let depth_current = length(position);
        let depth_prev = length(prev_position);
        let depth_ratio = abs(depth_current - depth_prev) / max(depth_current, 0.001);
        
        // Normal similarity check
        let normal_similarity = dot(normal, prev_normal);
        
        // Validate reprojection
        reprojection_valid = reprojection_valid && 
                            depth_ratio < MAX_DEPTH_RATIO && 
                            normal_similarity > MIN_NORMAL_SIMILARITY;
        
        if (reprojection_valid) {
            let prev_data = textureLoad(pixel_radiance_prev, pixel_prev, 0);
            prev_radiance = prev_data.rgb;
            prev_count = prev_data.w;
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Temporal Accumulation (Upscale-Aware)
    // ─────────────────────────────────────────────────────────────────────────
    // With upscale N×M, each pixel is traced every ~N*M frames on average.
    // When a pixel IS traced, we give it significant weight for responsiveness.
    // The blend alpha scales with upscale factor to maintain consistent response.
    
    var final_radiance: vec3<f32>;
    var final_count: f32;
    
    // Compute upscale-aware base blend factor
    // Higher upscale = fewer samples per pixel = each sample MORE valuable = higher alpha
    // Using sqrt for smooth scaling: 1×1→0.2, 2×2→0.4, 3×3→0.6, 4×4→0.8
    let upscale_factor = f32(upscale.x * upscale.y);
    let base_alpha = clamp(sqrt(upscale_factor) * MIN_BLEND_ALPHA, MIN_BLEND_ALPHA, MAX_BLEND_ALPHA);
    
    if (current_count > 0.0) {
        if (reprojection_valid && prev_count > 0.0) {
            // ─────────────────────────────────────────────────────────────────
            // Upscale-aware adaptive blending
            // ─────────────────────────────────────────────────────────────────
            
            // Compute luminance difference for variance-based adaptation
            let curr_luma = dot(current_radiance, vec3<f32>(0.2126, 0.7152, 0.0722));
            let prev_luma = dot(prev_radiance, vec3<f32>(0.2126, 0.7152, 0.0722));
            let luma_diff = abs(curr_luma - prev_luma) / max(max(curr_luma, prev_luma), 0.001);
            
            // Boost alpha when there's significant luminance change (lighting changed)
            // This makes the system react quickly to light switches, time-of-day, etc.
            let variance_alpha = base_alpha + luma_diff * VARIANCE_BOOST;
            let adaptive_alpha = clamp(variance_alpha, MIN_BLEND_ALPHA, MAX_BLEND_ALPHA);
            
            final_radiance = mix(prev_radiance, current_radiance, adaptive_alpha);
            final_count = prev_count + 1.0;
        } else {
            // No valid history - use current sample directly
            final_radiance = current_radiance;
            final_count = 1.0;
        }
    } else {
        if (reprojection_valid && prev_count > 0.0) {
            // Carry forward reprojected history
            // No decay needed - we'll blend properly when this pixel is traced
            final_radiance = prev_radiance;
            final_count = prev_count;
        } else {
            // No valid data - output zero
            final_radiance = vec3<f32>(0.0);
            final_count = 0.0;
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Output
    // ─────────────────────────────────────────────────────────────────────────
    // Store accumulated radiance for next frame
    textureStore(pixel_radiance_curr, pixel_coord, vec4<f32>(final_radiance, final_count));
    
    // Output final GI radiance
    textureStore(output_gi, pixel_coord, vec4<f32>(final_radiance, 1.0));
}
