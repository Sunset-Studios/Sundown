// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║               PER-PIXEL PATH TRACING - TEMPORAL ACCUMULATION              ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Implements "Ingredient #2: Accurate Temporal Accumulation" from          ║
// ║  NVIDIA's "Fast Denoising with Self-Stabilizing Recurrent Blurs" paper.   ║
// ║                                                                           ║
// ║  Key Features:                                                            ║
// ║  ─────────────                                                            ║
// ║  1. Linear Accumulation Weights:                                          ║
// ║     speed = 1/(1+N), where N = accumulated frame count                    ║
// ║     This gives true averaging: after N frames, each contributes 1/N       ║
// ║                                                                           ║
// ║  2. Ghosting-Free Bilinear Reprojection:                                  ║
// ║     - Each corner of bilinear footprint is tested independently           ║
// ║     - Invalid corners are excluded via custom bilinear weights            ║
// ║     - Graceful fallback to nearest valid sample                           ║
// ║                                                                           ║
// ║  3. Geometry-Only Validation:                                             ║
// ║     - Depth similarity test per corner                                    ║
// ║     - Normal similarity test per corner                                   ║
// ║                                                                           ║
// ║  Note: pixel_radiance_prev contains the BLURRED output from last frame    ║
// ║  (output of recurrent blur), ensuring history is "clean background".      ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

#include "common.wgsl"
#include "gi/gi_common.wgsl"
#include "raytracing/restir_common.wgsl"
#include "postprocess_common.wgsl"

// =============================================================================
// DEFINES
// =============================================================================
// Uncomment to skip temporal accumulation (output raw reservoir radiance)
// define SKIP_ACCUMULATION

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read> spatial_reservoir: array<GIReservoirData>;
@group(1) @binding(2) var pixel_radiance_prev_direct: texture_2d<f32>;
@group(1) @binding(3) var pixel_radiance_prev_indirect_diffuse: texture_2d<f32>;
@group(1) @binding(4) var pixel_radiance_prev_indirect_specular: texture_2d<f32>;
@group(1) @binding(5) var depth_texture: texture_2d<f32>;
@group(1) @binding(6) var prev_depth_texture: texture_2d<f32>;
@group(1) @binding(7) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(8) var gbuffer_normal_prev: texture_2d<f32>;
@group(1) @binding(9) var gbuffer_motion: texture_2d<f32>;
@group(1) @binding(10) var raw_accumulation_direct: texture_storage_2d<rgba16float, write>;
@group(1) @binding(11) var raw_accumulation_indirect_diffuse: texture_storage_2d<rgba16float, write>;
@group(1) @binding(12) var raw_accumulation_indirect_specular: texture_storage_2d<rgba16float, write>;
#if SPECULAR_MASK_ENABLED
@group(1) @binding(13) var specular_mask: texture_2d<u32>;
#endif

// =============================================================================
// CONSTANTS
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// Geometry Validation Thresholds (per-corner of bilinear footprint)
// These are used for disocclusion detection, NOT luminance-based rejection
// ─────────────────────────────────────────────────────────────────────────────
const DEPTH_THRESHOLD = 0.05;         // Relative depth difference threshold
const NORMAL_THRESHOLD = 0.95;         // Normal dot product threshold
// ─────────────────────────────────────────────────────────────────────────────
// Maximum Accumulated Frame Count
// ─────────────────────────────────────────────────────────────────────────────
const MAX_ACCUMULATED_FRAMES = 16.0;

// ─────────────────────────────────────────────────────────────────────────────
// Compute bilinear filter origin and weights from sub-pixel UV coordinates
// ─────────────────────────────────────────────────────────────────────────────
struct BilinearFilter {
    origin: vec2<i32>,   // Integer coordinates of top-left corner
    weights: vec2<f32>,  // Fractional weights for interpolation
}

fn get_bilinear_filter(pixel_center: vec2<f32>) -> BilinearFilter {
    // pixel_center is already in pixel coordinates (with 0.5 offset)
    let coord = pixel_center - 0.5;
    var result: BilinearFilter;
    result.origin = vec2<i32>(floor(coord));
    result.weights = coord - vec2<f32>(result.origin);
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Compute custom bilinear weights based on per-corner occlusion flags
// This is the key to ghosting-free reprojection:
// - Occluded corners get zero weight
// - Remaining corners are renormalized to sum to 1
// - If all corners are occluded, fall back to equal weights (nearest valid)
// ─────────────────────────────────────────────────────────────────────────────
fn get_bilinear_custom_weights(bilinear: BilinearFilter, occlusion: vec4<f32>) -> vec4<f32> {
    let bw = bilinear.weights;
    
    // Standard bilinear weights for 4 corners:
    // [0] = top-left,     [1] = top-right
    // [2] = bottom-left,  [3] = bottom-right
    let bilinear_weights = vec4<f32>(
        (1.0 - bw.x) * (1.0 - bw.y),  // top-left
        bw.x * (1.0 - bw.y),           // top-right
        (1.0 - bw.x) * bw.y,           // bottom-left
        bw.x * bw.y                    // bottom-right
    );
    
    // Mask out occluded corners
    var custom_weights = bilinear_weights * occlusion;
    
    // Renormalize so weights sum to 1
    let weight_sum = dot(custom_weights, vec4<f32>(1.0));
    
    // If all corners are occluded, fall back to occlusion flags as weights
    // (this gives equal weight to all valid corners)
    custom_weights = select(
        custom_weights / weight_sum,
        occlusion / max(dot(occlusion, vec4<f32>(1.0)), 0.0001),
        weight_sum < 0.0001
    );
    
    return custom_weights;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test if a single corner passes geometry validation
// Returns 1.0 if valid, 0.0 if occluded/disoccluded
// ─────────────────────────────────────────────────────────────────────────────
fn test_corner_validity(
    corner_coord: vec2<i32>,
    current_normal: vec3<f32>,
    current_depth: f32,
    camera_position: vec3<f32>,
    res: vec2<u32>,
    upscale_factor: u32,
    full_res: vec2<u32>
) -> f32 {
    // Bounds check
    if (corner_coord.x < 0 || corner_coord.y < 0 ||
        corner_coord.x >= i32(res.x) || corner_coord.y >= i32(res.y)) {
        return 0.0;
    }
    
    // Load previous frame's geometry
    let corner_full_coord = gi_pixel_to_full_res_pixel_coord(vec2<u32>(corner_coord), upscale_factor, full_res);
    let prev_depth = textureLoad(prev_depth_texture, corner_full_coord, 0u).r;
    let prev_normal_data = textureLoad(gbuffer_normal_prev, corner_full_coord, 0u);
    let prev_normal = safe_normalize(prev_normal_data.xyz);
    
    // Skip sky pixels
    if (length(prev_normal_data.xyz) < 0.01 || prev_depth >= 1.0) {
        return 0.0;
    }
    
    // Depth similarity test (relative difference)
    let prev_position = reconstruct_prev_world_position(
        coord_to_uv(vec2<i32>(corner_full_coord), full_res),
        prev_depth,
        u32(frame_info.view_index)
    );
    let prev_linear_depth = length(prev_position - camera_position);
    let depth_diff = abs(current_depth - prev_linear_depth) / max(current_depth, 0.001);
    let depth_valid = depth_diff < DEPTH_THRESHOLD;
    
    // Normal similarity test
    let normal_similarity = dot(current_normal, prev_normal);
    let normal_valid = normal_similarity > NORMAL_THRESHOLD;
    
    return select(0.0, 1.0, depth_valid && normal_valid);
}

// =============================================================================
// MAIN COMPUTE SHADER
// =============================================================================

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let full_res = vec2<u32>(u32(gi_params.full_resolution_x), u32(gi_params.full_resolution_y));
    let res = vec2<u32>(u32(gi_params.gi_resolution_x), u32(gi_params.gi_resolution_y));
    
    if (gid.x >= res.x || gid.y >= res.y) {
        return;
    }
    
    let pixel_coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let upscale_factor = u32(gi_params.upscale_factor);
    let full_pixel_coord = gi_pixel_to_full_res_pixel_coord(gid.xy, upscale_factor, full_res);
    
    let view = view_buffer[u32(frame_info.view_index)];
    let camera_position = view.view_position.xyz;

    // ─────────────────────────────────────────────────────────────────────────
    // Read G-buffer for current pixel
    // ─────────────────────────────────────────────────────────────────────────
    let depth = textureLoad(depth_texture, full_pixel_coord, 0u).r;
    let normal_data = textureLoad(gbuffer_normal, full_pixel_coord, 0u);
    let normal = safe_normalize(normal_data.xyz);
    let normal_length = length(normal_data.xyz);
    let position = reconstruct_world_position(
        coord_to_uv(vec2<i32>(full_pixel_coord), full_res),
        depth,
        u32(frame_info.view_index)
    );
    let current_depth = length(position - camera_position);
    
    // Skip sky pixels (no geometry)
    if (normal_length < 0.01) {
        textureStore(raw_accumulation_direct, pixel_coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(raw_accumulation_indirect_diffuse, pixel_coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(raw_accumulation_indirect_specular, pixel_coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

#if SPECULAR_MASK_ENABLED
    if (textureLoad(specular_mask, vec2<i32>(i32(gid.x), i32(gid.y)), 0).x == 0u) {
        textureStore(raw_accumulation_direct, pixel_coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(raw_accumulation_indirect_diffuse, pixel_coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(raw_accumulation_indirect_specular, pixel_coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }
#endif
    
    // Spatially reused reservoir sample -> evaluate RIS estimator (paper Eq. 6)
    let reservoir_index = gid.y * res.x + gid.x;
    let reservoir_entry = spatial_reservoir[reservoir_index];
    var has_current_sample = reservoir_entry.reservoir.m > 0u;

    // Evaluate (f(y) = BSDF * cos * L_o(sample_point)) * W_s at the current visible point.
    var current_radiance_direct = select(
        vec3<f32>(0.0),
        reservoir_entry.sample.outgoing_radiance_direct.xyz * reservoir_entry.reservoir.w,
        has_current_sample
    );
    var current_radiance_indirect_diffuse = select(
        vec3<f32>(0.0),
        reservoir_entry.sample.outgoing_radiance_indirect_diffuse.xyz * reservoir_entry.reservoir.w,
        has_current_sample
    );
    var current_radiance_indirect_specular = select(
        vec3<f32>(0.0),
        reservoir_entry.sample.outgoing_radiance_indirect_specular.xyz * reservoir_entry.reservoir.w,
        has_current_sample
    );

    // Pre-clamp current radiance to prevent fireflies from entering accumulation
    current_radiance_direct = safe_clamp_vec3_max(current_radiance_direct, MAX_RADIANCE_LUMINANCE);
    current_radiance_indirect_diffuse = safe_clamp_vec3_max(current_radiance_indirect_diffuse, MAX_RADIANCE_LUMINANCE);
    current_radiance_indirect_specular = safe_clamp_vec3_max(current_radiance_indirect_specular, MAX_RADIANCE_LUMINANCE);
    
    // ═════════════════════════════════════════════════════════════════════════
    // GHOSTING-FREE BILINEAR REPROJECTION
    // ═════════════════════════════════════════════════════════════════════════
    // Based on ReBLUR's "Ingredient #2: Accurate Temporal Accumulation"
    // Each corner of the bilinear footprint is tested independently for
    // geometry similarity. Invalid corners are excluded via custom weights.
    
    // ─────────────────────────────────────────────────────────────────────────
    // Compute reprojected position using motion vectors
    // ─────────────────────────────────────────────────────────────────────────
    let motion_sample = textureLoad(gbuffer_motion, vec2<i32>(full_pixel_coord), 0);
    let full_pixel_velocity = vec2<f32>(-0.5 * motion_sample.x, 0.5 * motion_sample.y) * vec2<f32>(f32(full_res.x) - 1.0, f32(full_res.y) - 1.0);
    let pixel_velocity = full_pixel_velocity / max(f32(upscale_factor), 1.0);
    
    let pixel_center = vec2<f32>(gid.xy) + 0.5;
    let pixel_prev_center = pixel_center + pixel_velocity;
    
    // ─────────────────────────────────────────────────────────────────────────
    // Get bilinear filter parameters
    // ─────────────────────────────────────────────────────────────────────────
    let bilinear = get_bilinear_filter(pixel_prev_center);
    
    // The four corners of the bilinear footprint
    let corner_00 = bilinear.origin;                          // top-left
    let corner_10 = bilinear.origin + vec2<i32>(1, 0);        // top-right
    let corner_01 = bilinear.origin + vec2<i32>(0, 1);        // bottom-left
    let corner_11 = bilinear.origin + vec2<i32>(1, 1);        // bottom-right
    
    // ─────────────────────────────────────────────────────────────────────────
    // Test each corner for geometry validity (depth + normal similarity)
    // Returns 1.0 if valid, 0.0 if occluded/disoccluded
    // ─────────────────────────────────────────────────────────────────────────
    let validity = vec4<f32>(
        test_corner_validity(corner_00, normal, current_depth, camera_position, res, upscale_factor, full_res),
        test_corner_validity(corner_10, normal, current_depth, camera_position, res, upscale_factor, full_res),
        test_corner_validity(corner_01, normal, current_depth, camera_position, res, upscale_factor, full_res),
        test_corner_validity(corner_11, normal, current_depth, camera_position, res, upscale_factor, full_res)
    );
    
    // ─────────────────────────────────────────────────────────────────────────
    // Compute custom bilinear weights (occluded corners get zero weight)
    // ─────────────────────────────────────────────────────────────────────────
    let custom_weights = get_bilinear_custom_weights(bilinear, validity);
    let any_valid = dot(validity, vec4<f32>(1.0)) > 0.0;
    
    // ─────────────────────────────────────────────────────────────────────────
    // Sample radiance and frame count from each corner, weighted by validity
    // ─────────────────────────────────────────────────────────────────────────
    var prev_radiance_direct = vec3<f32>(0.0);
    var prev_radiance_indirect_diffuse = vec3<f32>(0.0);
    var prev_radiance_indirect_specular = vec3<f32>(0.0);
    var count_sum = 0.0;
    
    if (any_valid) {
        // Sample each corner and apply custom weights
        let data_00_direct = textureLoad(pixel_radiance_prev_direct, corner_00, 0);
        let data_10_direct = textureLoad(pixel_radiance_prev_direct, corner_10, 0);
        let data_01_direct = textureLoad(pixel_radiance_prev_direct, corner_01, 0);
        let data_11_direct = textureLoad(pixel_radiance_prev_direct, corner_11, 0);

        let data_00_indirect_diffuse = textureLoad(pixel_radiance_prev_indirect_diffuse, corner_00, 0);
        let data_10_indirect_diffuse = textureLoad(pixel_radiance_prev_indirect_diffuse, corner_10, 0);
        let data_01_indirect_diffuse = textureLoad(pixel_radiance_prev_indirect_diffuse, corner_01, 0);
        let data_11_indirect_diffuse = textureLoad(pixel_radiance_prev_indirect_diffuse, corner_11, 0);

        let data_00_indirect_specular = textureLoad(pixel_radiance_prev_indirect_specular, corner_00, 0);
        let data_10_indirect_specular = textureLoad(pixel_radiance_prev_indirect_specular, corner_10, 0);
        let data_01_indirect_specular = textureLoad(pixel_radiance_prev_indirect_specular, corner_01, 0);
        let data_11_indirect_specular = textureLoad(pixel_radiance_prev_indirect_specular, corner_11, 0);
        
        // Weighted average of radiance
        prev_radiance_direct = data_00_direct.rgb * custom_weights.x +
                               data_10_direct.rgb * custom_weights.y +
                               data_01_direct.rgb * custom_weights.z +
                               data_11_direct.rgb * custom_weights.w;
        prev_radiance_indirect_diffuse = data_00_indirect_diffuse.rgb * custom_weights.x +
                                         data_10_indirect_diffuse.rgb * custom_weights.y +
                                         data_01_indirect_diffuse.rgb * custom_weights.z +
                                         data_11_indirect_diffuse.rgb * custom_weights.w;
        prev_radiance_indirect_specular = data_00_indirect_specular.rgb * custom_weights.x +
                                          data_10_indirect_specular.rgb * custom_weights.y +
                                          data_01_indirect_specular.rgb * custom_weights.z +
                                          data_11_indirect_specular.rgb * custom_weights.w;
        
        // Weighted average of frame count (use minimum for conservative estimate)
        // This ensures we don't over-trust history when mixing different counts
        count_sum =
            min(data_00_direct.w + f32(reservoir_entry.reservoir.m), MAX_ACCUMULATED_FRAMES) * custom_weights.x +
            min(data_10_direct.w + f32(reservoir_entry.reservoir.m), MAX_ACCUMULATED_FRAMES) * custom_weights.y +
            min(data_01_direct.w + f32(reservoir_entry.reservoir.m), MAX_ACCUMULATED_FRAMES) * custom_weights.z +
            min(data_11_direct.w + f32(reservoir_entry.reservoir.m), MAX_ACCUMULATED_FRAMES) * custom_weights.w;
    }
    
    // ═════════════════════════════════════════════════════════════════════════
    // TEMPORAL ACCUMULATION WITH LINEAR WEIGHTS
    // ═════════════════════════════════════════════════════════════════════════
    // Paper: speed = 1/(1+N), where N = number of accumulated frames
    // This gives true averaging: after N frames, each contributes 1/N
    //
    // Proof (from paper):
    // history[0] = curr0 * (1/1) = curr0
    // history[1] = history[0] * (1/2) + curr1 * (1/2) = (curr0 + curr1) / 2
    // history[2] = history[1] * (2/3) + curr2 * (1/3) = (curr0 + curr1 + curr2) / 3
    
    var final_radiance_direct: vec3<f32>;
    var final_radiance_indirect_diffuse: vec3<f32>;
    var final_radiance_indirect_specular: vec3<f32>;
    var final_count: f32;

    // Linear accumulation: alpha = 1 / (1 + N)
    let alpha = 1.0 / (1.0 + count_sum);
    // Blend current sample with history
    #if SKIP_ACCUMULATION
    final_radiance_direct = current_radiance_direct;
    final_radiance_indirect_diffuse = current_radiance_indirect_diffuse;
    final_radiance_indirect_specular = current_radiance_indirect_specular;
    #else
    final_radiance_direct = select(current_radiance_direct, mix(prev_radiance_direct, current_radiance_direct, alpha), any_valid);
    final_radiance_indirect_diffuse = select(current_radiance_indirect_diffuse, mix(prev_radiance_indirect_diffuse, current_radiance_indirect_diffuse, alpha), any_valid);
    final_radiance_indirect_specular = select(current_radiance_indirect_specular, mix(prev_radiance_indirect_specular, current_radiance_indirect_specular, alpha), any_valid);
    #endif

    final_count = select(1.0, count_sum, any_valid);
    
    // ─────────────────────────────────────────────────────────────────────────
    // Final firefly clamp on output radiance
    // ─────────────────────────────────────────────────────────────────────────
    final_radiance_direct = safe_clamp_vec3_max(final_radiance_direct, MAX_RADIANCE_LUMINANCE);
    final_radiance_indirect_diffuse = safe_clamp_vec3_max(final_radiance_indirect_diffuse, MAX_RADIANCE_LUMINANCE);
    final_radiance_indirect_specular = safe_clamp_vec3_max(final_radiance_indirect_specular, MAX_RADIANCE_LUMINANCE);
    
    // ─────────────────────────────────────────────────────────────────────────
    // Output
    // ─────────────────────────────────────────────────────────────────────────
    // Store accumulated radiance with sample count in alpha channel
    // Sample count is used by recurrent blur for adaptive radius:
    // blur_radius = BASE_RADIUS / (1 + sample_count)
    textureStore(raw_accumulation_direct, pixel_coord, vec4<f32>(final_radiance_direct, final_count));
    textureStore(raw_accumulation_indirect_diffuse, pixel_coord, vec4<f32>(final_radiance_indirect_diffuse, final_count));
    textureStore(raw_accumulation_indirect_specular, pixel_coord, vec4<f32>(final_radiance_indirect_specular, final_count));
}
