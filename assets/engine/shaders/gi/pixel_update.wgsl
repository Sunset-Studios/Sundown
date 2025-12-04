// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║               PER-PIXEL PATH TRACING - TEMPORAL ACCUMULATION              ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Accumulates per-pixel radiance samples over time:                        ║
// ║  • Temporal reprojection using motion vectors                             ║
// ║  • Geometry-aware blending (depth + normal validation)                    ║
// ║  • Adaptive sample accumulation for noise reduction                       ║
// ║                                                                           ║
// ║  Note: pixel_radiance_prev contains the BLURRED output from last frame    ║
// ║  (output of recurrent blur), not raw accumulation. This ensures the       ║
// ║  temporal history is the "clean background" from the denoiser.            ║
// ║                                                                           ║
// ║  Algorithm:                                                               ║
// ║  1. Read current frame's traced radiance from path state                  ║
// ║  2. Reproject to find previous frame's BLURRED value                      ║
// ║  3. Validate reprojection with depth/normal similarity                    ║
// ║  4. Blend current sample with history (adaptive alpha)                    ║
// ║  5. Output raw accumulation (recurrent blur will process this)            ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

#include "common.wgsl"
#include "gi/gi_common.wgsl"
#include "gi/world_cache_common.wgsl"
#include "postprocess_common.wgsl"

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read_write> gi_counters: GICounters;
@group(1) @binding(2) var<storage, read> pixel_path_state: array<PixelPathState>;
// Previous frame's BLURRED radiance (output of recurrent blur from last frame)
@group(1) @binding(3) var pixel_radiance_prev: texture_2d<f32>;
@group(1) @binding(4) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(5) var gbuffer_position_prev: texture_2d<f32>;
@group(1) @binding(6) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(7) var gbuffer_normal_prev: texture_2d<f32>;
@group(1) @binding(8) var gbuffer_motion: texture_2d<f32>;
@group(1) @binding(9) var gbuffer_smra: texture_2d<f32>;
@group(1) @binding(10) var<storage, read_write> world_cache: array<WorldCacheCell>;
// Raw accumulation output (will be processed by recurrent blur)
@group(1) @binding(11) var raw_accumulation: texture_storage_2d<rgba16float, write>;
@group(1) @binding(12) var output_gi: texture_storage_2d<rgba16float, write>;

// =============================================================================
// CONSTANTS
// =============================================================================

// Reprojection validation thresholds
const MIN_NORMAL_SIMILARITY = 0.95;
const MAX_DEPTH_RATIO = 0.1;

// Temporal blending parameters
const MIN_BLEND_ALPHA = 0.02;     // Minimum blend for stability (2%)
const MAX_BLEND_ALPHA = 1.0;      // Maximum blend for responsiveness (100% = instant)

// Lighting change detection (shadow borders, moving lights)
// Luminance is remapped via Reinhard (L / (1 + L)) to normalize the range
// and suppress fireflies. This maps [0, ∞) → [0, 1) while preserving
// relative differences. The ratio threshold applies to remapped values.
const MIN_LUMINANCE_RATIO = 0.1;        // Ratio threshold in remapped space
const MAX_LUMINANCE_RATIO = 1.0;        // Ratio threshold in remapped space

// ─────────────────────────────────────────────────────────────────────────────
// Firefly Prevention: Maximum output luminance
// This is the final safety clamp for all GI output
// ─────────────────────────────────────────────────────────────────────────────
const MAX_OUTPUT_LUMINANCE = 10.0;

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
    
    let view = view_buffer[u32(frame_info.view_index)];
    let camera_position = view.view_position.xyz;

    // ─────────────────────────────────────────────────────────────────────────
    // Read G-buffer for current pixel
    // ─────────────────────────────────────────────────────────────────────────
    let position = textureLoad(gbuffer_position, pixel_coord, 0).xyz;
    let normal_data = textureLoad(gbuffer_normal, pixel_coord, 0);
    let normal = safe_normalize(normal_data.xyz);
    let normal_length = length(normal_data.xyz);
    
    // Read material properties for roughness-aware accumulation
    let smra = textureLoad(gbuffer_smra, pixel_coord, 0);
    let reflectance = smra.r;
    let roughness = smra.g;
    let metallic = smra.b;
    
    // ─────────────────────────────────────────────────────────────────────────
    // Compute specularity factor for relaxing variance checks
    // Low roughness + high reflectivity = very specular = need relaxed variance checks
    // Specular surfaces have legitimately high variance in their reflections
    //
    // Effective reflectivity considers:
    // - Metals (metallic ≈ 1): always highly reflective
    // - Dielectrics: reflectivity depends on reflectance parameter (F0)
    //   High reflectance dielectrics (glass, polished surfaces) also need
    //   relaxed checks to properly gather reflections
    // ─────────────────────────────────────────────────────────────────────────
    let effective_reflectivity = mix(reflectance, 1.0, metallic);
    let specularity = (1.0 - roughness) * effective_reflectivity;
    
    // Skip sky pixels (no geometry)
    if (normal_length < 0.01) {
        textureStore(output_gi, pixel_coord, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        textureStore(raw_accumulation, pixel_coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
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
    // Pre-clamp current radiance to prevent fireflies from entering accumulation
    // ─────────────────────────────────────────────────────────────────────────
    current_radiance = safe_clamp_vec3_max(current_radiance, MAX_OUTPUT_LUMINANCE);
    
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
        let depth_current = length(position - camera_position);
        let depth_prev = length(prev_position - camera_position);
        let depth_ratio = abs(depth_current - depth_prev) / max(depth_current, 0.001);
        
        // Normal similarity check
        let normal_similarity = dot(normal, prev_normal);
        
        // Validate reprojection (geometry tests first)
        reprojection_valid = reprojection_valid && 
                            depth_ratio < MAX_DEPTH_RATIO && 
                            normal_similarity > MIN_NORMAL_SIMILARITY;
        
        if (reprojection_valid) {
            // Load previous frame's accumulated radiance
            let prev_data = textureLoad(pixel_radiance_prev, pixel_prev, 0);
            prev_radiance = prev_data.rgb;
            prev_count = prev_data.w;

            // ─────────────────────────────────────────────────────────────────
            // Firefly Prevention: Clamp current sample relative to history
            // For specular surfaces, we moderately relax these checks since
            // they have higher natural variance in their reflections.
            // ─────────────────────────────────────────────────────────────────
            if (current_count > 0.0 && prev_count > 0.0) {
                let curr_lum_raw = dot(current_radiance, vec3<f32>(0.2126, 0.7152, 0.0722));
                let prev_lum_raw = dot(prev_radiance, vec3<f32>(0.2126, 0.7152, 0.0722));
                
                // ───────────────────────────────────────────────────────────────
                // Variance-based firefly rejection: if current sample is much
                // brighter than history, clamp it to prevent single outliers.
                //
                // For specular surfaces, we use relaxed thresholds but still
                // apply some clamping for stability. The key is that specular
                // surfaces will converge via higher blend alpha, not by
                // completely disabling clamping.
                // ───────────────────────────────────────────────────────────────
                if (prev_count >= 1.0 && prev_lum_raw > 0.001) {
                    // Base tolerance scales with specularity
                    // Diffuse: 2x, Specular: 8x (moderate relaxation)
                    let base_ratio = mix(2.0, 8.0, specularity);
                    let max_ratio = base_ratio / sqrt(max(prev_count, 1.0));
                    let max_allowed_lum = prev_lum_raw * (1.0 + max_ratio);
                    
                    if (curr_lum_raw > max_allowed_lum) {
                        let clamp_scale = max_allowed_lum / curr_lum_raw;
                        current_radiance = current_radiance * clamp_scale;
                    }
                }
                
                // ───────────────────────────────────────────────────────────────
                // Shadow border detection: if current is MUCH darker/brighter
                // than history, we might be at a shadow edge or reflection change.
                // For specular surfaces, we use relaxed thresholds.
                // ───────────────────────────────────────────────────────────────
                let adjusted_min_ratio = mix(MIN_LUMINANCE_RATIO, 0.01, specularity);
                let adjusted_max_ratio = mix(MAX_LUMINANCE_RATIO, 10.0, specularity);
                
                let luminance_difference = curr_lum_raw < prev_lum_raw * adjusted_min_ratio
                    || curr_lum_raw > prev_lum_raw * adjusted_max_ratio;
                reprojection_valid = reprojection_valid && !luminance_difference;
            }
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
    
    // ─────────────────────────────────────────────────────────────────────────
    // Specularity-Aware Temporal Accumulation
    // ─────────────────────────────────────────────────────────────────────────
    // For specular surfaces:
    // - Use higher minimum blend alpha (faster updates)
    // - Cap effective history count (prevent over-accumulation)
    // This ensures specular reflections converge quickly while staying stable
    // ─────────────────────────────────────────────────────────────────────────
    
    // First sample luminance limit scales with specularity
    let max_first_sample_luminance = mix(3.0, MAX_OUTPUT_LUMINANCE, specularity);
    
    // Minimum blend alpha for specular surfaces (faster response)
    // Diffuse: 2% minimum, Highly specular: 15% minimum
    let specular_min_alpha = mix(MIN_BLEND_ALPHA, 0.15, specularity);

    // Maximum effective history count
    let max_effective_count = 32.0;
    
    if (current_count > 0.0) {
        if (reprojection_valid && prev_count > 0.0) {
            // Cap history count for specular surfaces to ensure faster convergence
            let sample_alpha = 1.0 / min(1.0 + prev_count, max_effective_count);
            let adaptive_alpha = clamp(sample_alpha, specular_min_alpha, MAX_BLEND_ALPHA);
            final_radiance = mix(prev_radiance, current_radiance, adaptive_alpha);
            
            // Track sample count for adaptive blur radius in recurrent blur pass
            // Sample count enables self-stabilizing blur: more samples → smaller radius
            final_count = min(prev_count + 1.0, max_effective_count);
        } else {
            // No valid history - use current sample with firefly clamping
            // A single sample with no history is high-variance, so clamp more tightly
            // For specular surfaces, we allow brighter samples
            final_radiance = safe_clamp_vec3_max(current_radiance, max_first_sample_luminance);
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
    // Final firefly clamp on output radiance
    // ─────────────────────────────────────────────────────────────────────────
    final_radiance = safe_clamp_vec3_max(final_radiance, MAX_OUTPUT_LUMINANCE);
    
    // ─────────────────────────────────────────────────────────────────────────
    // Output
    // ─────────────────────────────────────────────────────────────────────────
    // Store accumulated radiance with sample count in alpha channel
    // Sample count is used by recurrent blur for adaptive radius:
    // blur_radius = BASE_RADIUS / (1 + sample_count)
    textureStore(raw_accumulation, pixel_coord, vec4<f32>(final_radiance, final_count));
    
    // Output final GI radiance (will be refined by recurrent blur pass)
    textureStore(output_gi, pixel_coord, vec4<f32>(final_radiance, 1.0));
}
