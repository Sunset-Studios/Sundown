// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║          PER-PIXEL PATH TRACING - VARIANCE-GUIDED MOTION BLUR             ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Applies adaptive spatial blur guided by motion vectors and variance:    ║
// ║  • High-variance areas receive stronger blurring along motion direction  ║
// ║  • Low-variance (converged) areas remain sharp                           ║
// ║  • Bilateral weighting preserves geometric edges                         ║
// ║                                                                           ║
// ║  Algorithm:                                                               ║
// ║  1. Read variance for current pixel                                       ║
// ║  2. If variance below threshold, skip blur (already converged)           ║
// ║  3. Sample neighbors along motion vector direction                        ║
// ║  4. Weight samples by depth/normal similarity and neighbor variance      ║
// ║  5. Blend based on variance - higher variance = more blur                ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

#include "common.wgsl"
#include "gi/gi_common.wgsl"

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var input_radiance: texture_2d<f32>;
@group(1) @binding(2) var<storage, read> input_variance: array<f32>;
@group(1) @binding(3) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(4) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(5) var gbuffer_motion: texture_2d<f32>;
@group(1) @binding(6) var output_radiance: texture_storage_2d<rgba16float, write>;

// =============================================================================
// CONSTANTS
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// Variance thresholds for blur activation
// Below MIN_VARIANCE, no blur is applied (area is converged)
// Above MAX_VARIANCE, maximum blur strength is applied
// ─────────────────────────────────────────────────────────────────────────────
const MIN_VARIANCE_THRESHOLD = 0.5;   // Below this, skip blur entirely
const MAX_VARIANCE_THRESHOLD = 0.95;     // Above this, use maximum blur

// ─────────────────────────────────────────────────────────────────────────────
// Blur kernel configuration
// KERNEL_RADIUS determines how far along motion vector to sample
// KERNEL_SAMPLES is the number of samples in each direction
// ─────────────────────────────────────────────────────────────────────────────
const KERNEL_RADIUS = 8.0;              // Maximum blur radius in pixels
const KERNEL_SAMPLES = 4u;              // Samples per direction (total = 2*SAMPLES + 1)

// ─────────────────────────────────────────────────────────────────────────────
// Bilateral weight thresholds for edge preservation
// ─────────────────────────────────────────────────────────────────────────────
const DEPTH_SIGMA = 0.1;                // Depth similarity falloff
const NORMAL_SIGMA = 0.5;               // Normal similarity falloff (cos angle)

// ─────────────────────────────────────────────────────────────────────────────
// Helper: Convert 2D pixel coordinates to 1D buffer index
// ─────────────────────────────────────────────────────────────────────────────
fn pixel_to_variance_index(coord: vec2<i32>, width: u32) -> u32 {
    return u32(coord.y) * width + u32(coord.x);
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// Compute bilateral weight based on depth and normal similarity
// Returns weight in [0, 1] where 1 = perfect match, 0 = no contribution
// ─────────────────────────────────────────────────────────────────────────────
fn compute_bilateral_weight(
    center_pos: vec3<f32>,
    center_normal: vec3<f32>,
    sample_pos: vec3<f32>,
    sample_normal: vec3<f32>,
    camera_pos: vec3<f32>
) -> f32 {
    // Depth similarity: compare view-space depths
    let center_depth = length(center_pos - camera_pos);
    let sample_depth = length(sample_pos - camera_pos);
    let depth_diff = abs(center_depth - sample_depth) / max(center_depth, 0.001);
    let depth_weight = exp(-depth_diff * depth_diff / (2.0 * DEPTH_SIGMA * DEPTH_SIGMA));
    
    // Normal similarity: compare surface orientations
    let normal_dot = max(dot(center_normal, sample_normal), 0.0);
    let normal_weight = pow(normal_dot, 1.0 / NORMAL_SIGMA);
    
    return depth_weight * normal_weight;
}

// ─────────────────────────────────────────────────────────────────────────────
// Map variance to blur strength
// Uses smooth mapping from [MIN, MAX] variance to [0, 1] blur strength
// ─────────────────────────────────────────────────────────────────────────────
fn variance_to_blur_strength(variance: f32) -> f32 {
    // Smoothstep mapping for gradual transition
    let t = saturate((variance - MIN_VARIANCE_THRESHOLD) / 
                     (MAX_VARIANCE_THRESHOLD - MIN_VARIANCE_THRESHOLD));
    // Use squared curve for more aggressive blur at high variance
    return t * t;
}

// =============================================================================
// MAIN COMPUTE SHADER
// =============================================================================

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(input_radiance);
    
    if (gid.x >= res.x || gid.y >= res.y) {
        return;
    }
    
    let pixel_coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let pixel_uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(res);
    
    // ─────────────────────────────────────────────────────────────────────────
    // Load center pixel data
    // ─────────────────────────────────────────────────────────────────────────
    let center_radiance_data = textureLoad(input_radiance, pixel_coord, 0);
    let center_radiance = center_radiance_data.rgb;
    let sample_count = center_radiance_data.a;
    
    let center_variance_idx = pixel_to_variance_index(pixel_coord, res.x);
    let center_variance = input_variance[center_variance_idx];
    
    // ─────────────────────────────────────────────────────────────────────────
    // Early out: skip blur for converged pixels
    // ─────────────────────────────────────────────────────────────────────────
    if (center_variance < MIN_VARIANCE_THRESHOLD) {
        textureStore(output_radiance, pixel_coord, vec4<f32>(center_radiance, sample_count));
        return;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Load G-buffer data for center pixel
    // ─────────────────────────────────────────────────────────────────────────
    let center_position = textureLoad(gbuffer_position, pixel_coord, 0).xyz;
    let center_normal_data = textureLoad(gbuffer_normal, pixel_coord, 0);
    let center_normal = safe_normalize(center_normal_data.xyz);
    
    // Skip sky pixels
    if (length(center_normal_data.xyz) < 0.01) {
        textureStore(output_radiance, pixel_coord, vec4<f32>(center_radiance, sample_count));
        return;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Get motion vector and compute blur direction
    // Motion vector guides the blur direction for temporal coherence
    // ─────────────────────────────────────────────────────────────────────────
    let motion_sample = textureLoad(gbuffer_motion, pixel_coord, 0);
    var motion_vec = motion_sample.xy * vec2<f32>(f32(res.x), f32(res.y)) * vec2<f32>(0.5, -0.5);
    
    // Clamp motion vector magnitude to prevent excessive blur
    let motion_length = length(motion_vec);
    let max_motion = KERNEL_RADIUS * 2.0;
    if (motion_length > max_motion) {
        motion_vec = motion_vec * (max_motion / motion_length);
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Compute blur direction: prioritize motion, fallback to perpendicular
    // For static pixels, use a circular blur pattern instead
    // ─────────────────────────────────────────────────────────────────────────
    var blur_direction: vec2<f32>;
    var use_motion_blur = motion_length > 0.5;
    
    if (use_motion_blur) {
        blur_direction = normalize(motion_vec);
    } else {
        // Static pixel: use screen-space gradient direction or default
        blur_direction = vec2<f32>(1.0, 0.0);
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Compute blur strength from variance
    // ─────────────────────────────────────────────────────────────────────────
    let blur_strength = variance_to_blur_strength(center_variance);
    let effective_radius = KERNEL_RADIUS * blur_strength;
    
    // If effective radius is too small, skip blur
    if (effective_radius < 0.5) {
        textureStore(output_radiance, pixel_coord, vec4<f32>(center_radiance, sample_count));
        return;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Camera position for bilateral depth weighting
    // ─────────────────────────────────────────────────────────────────────────
    let view = view_buffer[u32(frame_info.view_index)];
    let camera_position = view.view_position.xyz;
    
    // ─────────────────────────────────────────────────────────────────────────
    // Gather samples along blur direction
    // ─────────────────────────────────────────────────────────────────────────
    var accumulated_radiance = center_radiance;
    var total_weight = 1.0;
    
    // Sample in both directions along the blur axis
    for (var i = 1u; i <= KERNEL_SAMPLES; i = i + 1u) {
        let offset_scale = f32(i) / f32(KERNEL_SAMPLES) * effective_radius;
        
        // Sample in positive direction
        let offset_pos = blur_direction * offset_scale;
        let sample_coord_pos = pixel_coord + vec2<i32>(i32(round(offset_pos.x)), i32(round(offset_pos.y)));
        
        // Sample in negative direction
        let offset_neg = -blur_direction * offset_scale;
        let sample_coord_neg = pixel_coord + vec2<i32>(i32(round(offset_neg.x)), i32(round(offset_neg.y)));
        
        // Process positive direction sample
        if (sample_coord_pos.x >= 0 && sample_coord_pos.x < i32(res.x) &&
            sample_coord_pos.y >= 0 && sample_coord_pos.y < i32(res.y)) {
            
            let sample_position = textureLoad(gbuffer_position, sample_coord_pos, 0).xyz;
            let sample_normal_data = textureLoad(gbuffer_normal, sample_coord_pos, 0);
            let sample_normal = safe_normalize(sample_normal_data.xyz);
            
            // Skip invalid samples (sky, etc.)
            if (length(sample_normal_data.xyz) > 0.01) {
                let sample_radiance_data = textureLoad(input_radiance, sample_coord_pos, 0);
                let sample_radiance = sample_radiance_data.rgb;
                let sample_variance_idx = pixel_to_variance_index(sample_coord_pos, res.x);
                let sample_variance = input_variance[sample_variance_idx];
                
                // Compute bilateral weight
                let bilateral = compute_bilateral_weight(
                    center_position, center_normal,
                    sample_position, sample_normal,
                    camera_position
                );
                
                // Spatial weight: Gaussian falloff with distance
                let dist_factor = f32(i) / f32(KERNEL_SAMPLES);
                let spatial_weight = exp(-dist_factor * dist_factor * 2.0);
                
                // Variance-based weight: prefer low-variance samples
                // High variance neighbors contribute less to avoid spreading noise
                let variance_weight = 1.0 / (1.0 + sample_variance * 10.0);
                
                let weight = bilateral * spatial_weight * variance_weight;
                
                accumulated_radiance += sample_radiance * weight;
                total_weight += weight;
            }
        }
        
        // Process negative direction sample
        if (sample_coord_neg.x >= 0 && sample_coord_neg.x < i32(res.x) &&
            sample_coord_neg.y >= 0 && sample_coord_neg.y < i32(res.y)) {
            
            let sample_position = textureLoad(gbuffer_position, sample_coord_neg, 0).xyz;
            let sample_normal_data = textureLoad(gbuffer_normal, sample_coord_neg, 0);
            let sample_normal = safe_normalize(sample_normal_data.xyz);
            
            // Skip invalid samples
            if (length(sample_normal_data.xyz) > 0.01) {
                let sample_radiance_data = textureLoad(input_radiance, sample_coord_neg, 0);
                let sample_radiance = sample_radiance_data.rgb;
                let sample_variance_idx = pixel_to_variance_index(sample_coord_neg, res.x);
                let sample_variance = input_variance[sample_variance_idx];
                
                // Compute bilateral weight
                let bilateral = compute_bilateral_weight(
                    center_position, center_normal,
                    sample_position, sample_normal,
                    camera_position
                );
                
                // Spatial weight
                let dist_factor = f32(i) / f32(KERNEL_SAMPLES);
                let spatial_weight = exp(-dist_factor * dist_factor * 2.0);
                
                // Variance-based weight
                let variance_weight = 1.0 / (1.0 + sample_variance * 10.0);
                
                let weight = bilateral * spatial_weight * variance_weight;
                
                accumulated_radiance += sample_radiance * weight;
                total_weight += weight;
            }
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // For static pixels with high variance, also sample perpendicular direction
    // This creates a cross-shaped kernel for better noise reduction
    // ─────────────────────────────────────────────────────────────────────────
    if (!use_motion_blur && blur_strength > 0.3) {
        let perp_direction = vec2<f32>(-blur_direction.y, blur_direction.x);
        
        for (var i = 1u; i <= KERNEL_SAMPLES / 2u; i = i + 1u) {
            let offset_scale = f32(i) / f32(KERNEL_SAMPLES) * effective_radius * 0.5;
            
            // Sample perpendicular positive
            let offset_perp_pos = perp_direction * offset_scale;
            let sample_coord_perp_pos = pixel_coord + vec2<i32>(i32(round(offset_perp_pos.x)), i32(round(offset_perp_pos.y)));
            
            // Sample perpendicular negative
            let offset_perp_neg = -perp_direction * offset_scale;
            let sample_coord_perp_neg = pixel_coord + vec2<i32>(i32(round(offset_perp_neg.x)), i32(round(offset_perp_neg.y)));
            
            // Process perpendicular positive sample
            if (sample_coord_perp_pos.x >= 0 && sample_coord_perp_pos.x < i32(res.x) &&
                sample_coord_perp_pos.y >= 0 && sample_coord_perp_pos.y < i32(res.y)) {
                
                let sample_position = textureLoad(gbuffer_position, sample_coord_perp_pos, 0).xyz;
                let sample_normal_data = textureLoad(gbuffer_normal, sample_coord_perp_pos, 0);
                let sample_normal = safe_normalize(sample_normal_data.xyz);
                
                if (length(sample_normal_data.xyz) > 0.01) {
                    let sample_radiance_data = textureLoad(input_radiance, sample_coord_perp_pos, 0);
                    let sample_radiance = sample_radiance_data.rgb;
                    let sample_variance_idx = pixel_to_variance_index(sample_coord_perp_pos, res.x);
                    let sample_variance = input_variance[sample_variance_idx];
                    
                    let bilateral = compute_bilateral_weight(
                        center_position, center_normal,
                        sample_position, sample_normal,
                        camera_position
                    );
                    
                    let dist_factor = f32(i) / f32(KERNEL_SAMPLES);
                    let spatial_weight = exp(-dist_factor * dist_factor * 2.0) * 0.5;
                    let variance_weight = 1.0 / (1.0 + sample_variance * 10.0);
                    let weight = bilateral * spatial_weight * variance_weight;
                    
                    accumulated_radiance += sample_radiance * weight;
                    total_weight += weight;
                }
            }
            
            // Process perpendicular negative sample
            if (sample_coord_perp_neg.x >= 0 && sample_coord_perp_neg.x < i32(res.x) &&
                sample_coord_perp_neg.y >= 0 && sample_coord_perp_neg.y < i32(res.y)) {
                
                let sample_position = textureLoad(gbuffer_position, sample_coord_perp_neg, 0).xyz;
                let sample_normal_data = textureLoad(gbuffer_normal, sample_coord_perp_neg, 0);
                let sample_normal = safe_normalize(sample_normal_data.xyz);
                
                if (length(sample_normal_data.xyz) > 0.01) {
                    let sample_radiance_data = textureLoad(input_radiance, sample_coord_perp_neg, 0);
                    let sample_radiance = sample_radiance_data.rgb;
                    let sample_variance_idx = pixel_to_variance_index(sample_coord_perp_neg, res.x);
                    let sample_variance = input_variance[sample_variance_idx];
                    
                    let bilateral = compute_bilateral_weight(
                        center_position, center_normal,
                        sample_position, sample_normal,
                        camera_position
                    );
                    
                    let dist_factor = f32(i) / f32(KERNEL_SAMPLES);
                    let spatial_weight = exp(-dist_factor * dist_factor * 2.0) * 0.5;
                    let variance_weight = 1.0 / (1.0 + sample_variance * 10.0);
                    let weight = bilateral * spatial_weight * variance_weight;
                    
                    accumulated_radiance += sample_radiance * weight;
                    total_weight += weight;
                }
            }
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Normalize and output
    // Variance is not updated here - it's tracked in the accumulation pass
    // The blur pass only uses variance for guidance, doesn't modify it
    // ─────────────────────────────────────────────────────────────────────────
    let final_radiance = accumulated_radiance / total_weight;
    
    textureStore(output_radiance, pixel_coord, vec4<f32>(final_radiance, sample_count));
}

