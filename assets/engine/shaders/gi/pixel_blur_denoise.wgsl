// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║        STABILIZED RECURRENT BLUR - ADAPTIVE RADIUS SPATIAL FILTER         ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Implements "Ingredient #1: Recurrent Blur" and "Ingredient #4: Choosing  ║
// ║  Sampling Space" from NVIDIA's "Fast Denoising with Self-Stabilizing      ║
// ║  Recurrent Blurs" (GTC 2020)                                              ║
// ║                                                                           ║
// ║  Key Insight - Recurrent Blur:                                            ║
// ║  ──────────────────────────────                                           ║
// ║  Unlike standard blur that only uses current frame's noisy input,         ║
// ║  recurrent blur samples NEIGHBORS from previous frame's BLURRED output    ║
// ║  (the "clean background"). This redistributes spatial sampling over       ║
// ║  time: 30 FPS × 8 samples = 240 cumulative samples/sec.                   ║
// ║                                                                           ║
// ║  Stabilization via Adaptive Radius:                                       ║
// ║  ───────────────────────────────────                                      ║
// ║  • blur_radius = BASE_RADIUS / (1.0 + sample_count)                       ║
// ║  • Fresh pixels (low sample count) → Large radius → Aggressive blur       ║
// ║  • Converged pixels (high sample count) → Small radius → Sharp output     ║
// ║                                                                           ║
// ║  This naturally prevents over-blurring: as accumulation converges,        ║
// ║  the blur radius shrinks to preserve detail.                              ║
// ║                                                                           ║
// ║  Ingredient #4 - Anisotropic Screen-Space Kernel:                         ║
// ║  ─────────────────────────────────────────────────                        ║
// ║  At grazing viewing angles, the kernel is stretched along the surface     ║
// ║  direction in screen space. This prevents blur across depth               ║
// ║  discontinuities while maintaining edge-aligned filtering.                ║
// ║                                                                           ║
// ║  Inputs:                                                                   ║
// ║  • current_radiance: This frame's accumulated radiance (sample count in α)║
// ║  • blur_prev: Previous frame's blurred output (clean background)          ║
// ║                                                                           ║
// ║  Outputs:                                                                  ║
// ║  • blur_curr: This frame's blurred output (becomes next frame's blur_prev)║
// ║  • gi_output: Final GI for deferred lighting passes                       ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

#include "common.wgsl"
#include "gi/gi_common.wgsl"

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> gi_params: GIParams;
// Current frame's raw temporal accumulation (from pixel_accumulate)
// Contains fresh accumulated radiance with sample count in .w
@group(1) @binding(1) var raw_accumulation: texture_2d<f32>;
// Previous frame's BLURRED output - the "clean background" for recurrent blur
// Neighbors are sampled from here, enabling temporal redistribution of spatial sampling
@group(1) @binding(2) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(3) var gbuffer_position_prev: texture_2d<f32>;
@group(1) @binding(4) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(5) var gbuffer_normal_prev: texture_2d<f32>;
// Current frame's blurred output - becomes pixel_radiance_prev next frame
// This is what pixel_accumulate will read as history in the next frame
@group(1) @binding(6) var pixel_radiance_curr: texture_storage_2d<rgba16float, write>;
// Final GI output for deferred lighting passes
@group(1) @binding(7) var gi_output: texture_storage_2d<rgba16float, write>;

// =============================================================================
// CONSTANTS
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// Blur Configuration
// BASE_RADIUS: Maximum blur radius in PIXELS when sample_count = 0
// As samples accumulate, effective radius = BASE_RADIUS / (1 + sample_count)
// ─────────────────────────────────────────────────────────────────────────────
const BASE_BLUR_RADIUS: f32 = 8.0;

// ─────────────────────────────────────────────────────────────────────────────
// Number of samples per blur pass
// Using 8 samples with Poisson disk distribution works well (paper recommendation)
// 30 FPS × 8 samples = 240 cumulative samples/sec due to recurrent nature
// ─────────────────────────────────────────────────────────────────────────────
const NUM_SAMPLES: u32 = 0u;

// ─────────────────────────────────────────────────────────────────────────────
// Bilateral weight parameters for edge preservation
// ─────────────────────────────────────────────────────────────────────────────
const DEPTH_SIGMA: f32 = 0.05;          // Depth similarity falloff (relative)
const NORMAL_POWER: f32 = 64.0;         // Normal similarity sharpness

// ─────────────────────────────────────────────────────────────────────────────
// Anisotropic Kernel Configuration (Ingredient #4)
// MIN_ANISO_SCALE: Minimum scale at grazing angles (0.25 = 4x compression)
// MAX_ACCUMULATED_FRAMES: Sample count at which kernel becomes fully isotropic
// ─────────────────────────────────────────────────────────────────────────────
const MIN_ANISO_SCALE: f32 = 0.25;
const MAX_ACCUMULATED_FRAMES: f32 = 8.0;

// ─────────────────────────────────────────────────────────────────────────────
// Poisson Disk Sample Offsets
// Pre-computed 8-sample Poisson disk for quality spatial distribution
// ─────────────────────────────────────────────────────────────────────────────
const POISSON_DISK: array<vec2<f32>, 8> = array<vec2<f32>, 8>(
    vec2<f32>( 0.0,       1.0      ),
    vec2<f32>( 0.7071,    0.7071   ),
    vec2<f32>( 1.0,       0.0      ),
    vec2<f32>( 0.7071,   -0.7071   ),
    vec2<f32>( 0.0,      -1.0      ),
    vec2<f32>(-0.7071,   -0.7071   ),
    vec2<f32>(-1.0,       0.0      ),
    vec2<f32>(-0.7071,    0.7071   )
);

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// Compute adaptive blur radius based on sample count (in pixels)
// ─────────────────────────────────────────────────────────────────────────────
fn compute_adaptive_radius(sample_count: f32) -> f32 {
    return BASE_BLUR_RADIUS / (1.0 + sample_count);
}

// ─────────────────────────────────────────────────────────────────────────────
// Compute bilateral weight for edge-aware filtering
// Preserves geometric discontinuities by down-weighting samples across edges
// ─────────────────────────────────────────────────────────────────────────────
fn compute_bilateral_weight(
    center_depth: f32,
    center_normal: vec3<f32>,
    sample_depth: f32,
    sample_normal: vec3<f32>
) -> f32 {
    // ─────────────────────────────────────────────────────────────────────
    // Depth weight: Gaussian falloff based on relative depth difference
    // ─────────────────────────────────────────────────────────────────────
    let depth_diff = abs(center_depth - sample_depth) / max(center_depth, 0.001);
    let depth_weight = exp(-depth_diff * depth_diff / (2.0 * DEPTH_SIGMA * DEPTH_SIGMA));
    
    // ─────────────────────────────────────────────────────────────────────
    // Normal weight: Sharp falloff for facing angle difference
    // Using power function for sharper edge preservation
    // ─────────────────────────────────────────────────────────────────────
    let normal_dot = max(dot(center_normal, sample_normal), 0.0);
    let normal_weight = pow(normal_dot, NORMAL_POWER);
    
    return depth_weight * normal_weight;
}

// ─────────────────────────────────────────────────────────────────────────────
// Compute the screen-space direction of surface tilt
// This is the direction in which the surface "falls away" from the viewer
// Returns: 2D unit vector in screen space, or (0,0) if viewing perpendicular
// ─────────────────────────────────────────────────────────────────────────────
fn compute_screen_space_tilt(
    normal: vec3<f32>,
    view_matrix: mat4x4<f32>
) -> vec2<f32> {
    // Transform normal to view space
    let view_normal = (view_matrix * vec4<f32>(normal, 0.0)).xyz;
    
    // The screen-space tilt is the XY component of the view-space normal
    // When normal.z (view-space) is 1, we're looking straight at the surface
    // When normal.z approaches 0, we're at a grazing angle
    let screen_tilt = vec2<f32>(view_normal.x, -view_normal.y); // Flip Y for screen coords
    
    let tilt_length = length(screen_tilt);
    if (tilt_length < 0.001) {
        return vec2<f32>(0.0, 0.0);
    }
    
    return screen_tilt / tilt_length;
}

// ─────────────────────────────────────────────────────────────────────────────
// Apply anisotropic scaling to a 2D sample offset
// At grazing angles, compresses the offset in the tilt direction
// 
// Parameters:
//   offset: Original sample offset (e.g., from Poisson disk)
//   tilt_dir: Screen-space direction of surface tilt (unit vector)
//   n_dot_v: Cosine of viewing angle (1 = perpendicular, 0 = grazing)
//   sample_count: Number of accumulated samples (for progressive relaxation)
//
// Returns: Scaled sample offset
// ─────────────────────────────────────────────────────────────────────────────
fn apply_anisotropic_scaling(
    offset: vec2<f32>,
    tilt_dir: vec2<f32>,
    n_dot_v: f32,
    sample_count: f32
) -> vec2<f32> {
    // ─────────────────────────────────────────────────────────────────────
    // Compute anisotropic scale factor based on viewing angle
    // At perpendicular viewing (n_dot_v = 1): scale = 1.0 (isotropic)
    // At grazing viewing (n_dot_v = 0): scale = MIN_ANISO_SCALE (compressed)
    // ─────────────────────────────────────────────────────────────────────
    let angle_factor = 1.0 - n_dot_v;  // 0 at perpendicular, 1 at grazing
    let aniso_scale = mix(1.0, MIN_ANISO_SCALE, angle_factor);
    
    // ─────────────────────────────────────────────────────────────────────
    // Progressive relaxation: as samples accumulate, move towards isotropic
    // This prevents permanent stretching artifacts over time
    // ─────────────────────────────────────────────────────────────────────
    let normalized_count = saturate(sample_count / MAX_ACCUMULATED_FRAMES);
    let final_scale = mix(aniso_scale, 1.0, normalized_count);
    
    // ─────────────────────────────────────────────────────────────────────
    // Apply scaling: compress offset in the tilt direction
    // Decompose offset into parallel and perpendicular components to tilt
    // ─────────────────────────────────────────────────────────────────────
    let tilt_len_sq = dot(tilt_dir, tilt_dir);
    if (tilt_len_sq < 0.0001) {
        // No significant tilt, return original offset
        return offset;
    }
    
    // Component parallel to tilt direction (this gets scaled)
    let parallel_amount = dot(offset, tilt_dir);
    let parallel_component = tilt_dir * parallel_amount;
    
    // Component perpendicular to tilt direction (this stays the same)
    let perp_component = offset - parallel_component;
    
    // Scale the parallel component and recombine
    return perp_component + parallel_component * final_scale;
}

// =============================================================================
// MAIN COMPUTE SHADER
// =============================================================================

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(raw_accumulation);
    
    // ─────────────────────────────────────────────────────────────────────────
    // Bounds check
    // ─────────────────────────────────────────────────────────────────────────
    if (gid.x >= res.x || gid.y >= res.y) {
        return;
    }
    
    let pixel_coord = vec2<i32>(i32(gid.x), i32(gid.y));
    
    // ─────────────────────────────────────────────────────────────────────────
    // Load center pixel's CURRENT accumulated radiance and sample count
    // This is the fresh temporal accumulation from pixel_accumulate
    // ─────────────────────────────────────────────────────────────────────────
    let center_data = textureLoad(raw_accumulation, pixel_coord, 0);
    let center_radiance = center_data.rgb;
    let sample_count = center_data.a;
    
    // ─────────────────────────────────────────────────────────────────────────
    // Load center G-buffer data
    // ─────────────────────────────────────────────────────────────────────────
    let center_position = textureLoad(gbuffer_position, pixel_coord, 0).xyz;
    let center_normal_data = textureLoad(gbuffer_normal, pixel_coord, 0);
    let center_normal = safe_normalize(center_normal_data.xyz);
    
    // ─────────────────────────────────────────────────────────────────────────
    // Early out: Skip sky pixels (no geometry to blur)
    // ─────────────────────────────────────────────────────────────────────────
    if (length(center_normal_data.xyz) <= 0.0) {
        textureStore(pixel_radiance_curr, pixel_coord, vec4<f32>(center_radiance, sample_count));
        textureStore(gi_output, pixel_coord, vec4<f32>(center_radiance, 1.0));
        return;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Get view data for depth computation and anisotropic kernel
    // ─────────────────────────────────────────────────────────────────────────
    let view = view_buffer[u32(frame_info.view_index)];
    let camera_position = view.view_position.xyz;
    let view_matrix = view.view_matrix;
    let center_depth = length(center_position - camera_position);
    
    // ─────────────────────────────────────────────────────────────────────────
    // Compute view direction and N·V for anisotropic kernel
    // ─────────────────────────────────────────────────────────────────────────
    let view_dir = normalize(camera_position - center_position);
    let n_dot_v = saturate(abs(dot(center_normal, view_dir)));
    
    // ─────────────────────────────────────────────────────────────────────────
    // Compute screen-space tilt direction for anisotropic scaling
    // This is used to stretch/compress the kernel based on surface orientation
    // ─────────────────────────────────────────────────────────────────────────
    let tilt_dir = compute_screen_space_tilt(center_normal, view_matrix);
    
    // ─────────────────────────────────────────────────────────────────────────
    // Compute adaptive blur radius in pixels
    // ─────────────────────────────────────────────────────────────────────────
    let effective_radius = compute_adaptive_radius(sample_count);
    
    // ─────────────────────────────────────────────────────────────────────────
    // Accumulate samples using anisotropic screen-space kernel
    // ─────────────────────────────────────────────────────────────────────────
    var accumulated_radiance = center_radiance;
    var total_sample_count = sample_count;
    var total_weight = 1.0;
    
    for (var i = 0u; i < NUM_SAMPLES; i = i + 1u) {
        // ─────────────────────────────────────────────────────────────────
        // Get base Poisson disk offset
        // ─────────────────────────────────────────────────────────────────
        let base_offset = POISSON_DISK[i];
        
        // ─────────────────────────────────────────────────────────────────
        // Apply anisotropic scaling (Ingredient #4)
        // At grazing angles, compress kernel in the tilt direction
        // ─────────────────────────────────────────────────────────────────
        let scaled_offset = apply_anisotropic_scaling(
            base_offset,
            tilt_dir,
            n_dot_v,
            sample_count
        );
        
        // ─────────────────────────────────────────────────────────────────
        // Compute sample pixel coordinates
        // ─────────────────────────────────────────────────────────────────
        let sample_offset = scaled_offset * effective_radius;
        let sample_coord = pixel_coord + vec2<i32>(
            i32(round(sample_offset.x)),
            i32(round(sample_offset.y))
        );
        
        // ─────────────────────────────────────────────────────────────────
        // Bounds check for sample
        // ─────────────────────────────────────────────────────────────────
        if (sample_coord.x < 0 || sample_coord.x >= i32(res.x) ||
            sample_coord.y < 0 || sample_coord.y >= i32(res.y)) {
            continue;
        }
        
        // ─────────────────────────────────────────────────────────────────
        // Load sample G-buffer data
        // ─────────────────────────────────────────────────────────────────
        let sample_position = textureLoad(gbuffer_position, sample_coord, 0).xyz;
        let sample_normal_data = textureLoad(gbuffer_normal, sample_coord, 0);
        let sample_normal = safe_normalize(sample_normal_data.xyz);
        
        // Skip invalid samples (sky pixels)
        if (length(sample_normal_data.xyz) <= 0.0) {
            continue;
        }
        
        // ─────────────────────────────────────────────────────────────────
        // Load sample data from raw accumulation
        // ─────────────────────────────────────────────────────────────────
        let sample_data = textureLoad(raw_accumulation, sample_coord, 0);
        let sample_radiance = sample_data.rgb;
        let neighbor_sample_count = sample_data.a;
        let sample_depth = length(sample_position - camera_position);
        
        // ─────────────────────────────────────────────────────────────────
        // Compute bilateral weight for edge preservation
        // ─────────────────────────────────────────────────────────────────
        let bilateral_weight = compute_bilateral_weight(
            center_depth, center_normal,
            sample_depth, sample_normal
        );
        
        // ─────────────────────────────────────────────────────────────────
        // Spatial weight: Gaussian falloff with distance
        // Use original (non-scaled) offset for consistent weighting
        // ─────────────────────────────────────────────────────────────────
        let dist_sq = dot(base_offset, base_offset);
        let spatial_weight = exp(-dist_sq * 0.5);
        
        // ─────────────────────────────────────────────────────────────────
        // Combined weight
        // ─────────────────────────────────────────────────────────────────
        let weight = bilateral_weight * spatial_weight;
        
        // ─────────────────────────────────────────────────────────────────
        // Accumulate weighted sample
        // ─────────────────────────────────────────────────────────────────
        accumulated_radiance += sample_radiance * weight;
        total_sample_count += neighbor_sample_count * weight;
        total_weight += weight;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Normalize and output
    // ─────────────────────────────────────────────────────────────────────────
    let final_radiance = accumulated_radiance / total_weight;
    let final_sample_count = total_sample_count / total_weight;
    
    // Output to blur buffer (becomes blur_prev next frame for recurrence)
    // Preserve sample count in alpha for next frame's adaptive radius
    textureStore(pixel_radiance_curr, pixel_coord, vec4<f32>(final_radiance, final_sample_count));
    
    // Output to final GI texture for deferred lighting passes
    textureStore(gi_output, pixel_coord, vec4<f32>(final_radiance, 1.0));
}
