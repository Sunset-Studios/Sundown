#include "common_types.wgsl"

// ------------------------------------------------------------------------------------
// Buffers
// ------------------------------------------------------------------------------------ 

@group(0) @binding(0) var<storage, read> vertex_buffer: array<Vertex>;
@group(0) @binding(1) var<storage, read> view_buffer: array<View>;
@group(0) @binding(2) var global_sampler: sampler;
@group(0) @binding(3) var non_filtering_sampler: sampler;
@group(0) @binding(4) var clamped_sampler: sampler;
@group(0) @binding(5) var comparison_sampler: sampler_comparison;
@group(0) @binding(6) var<uniform> frame_info: FrameInfo;

// ------------------------------------------------------------------------------------
// Helper Functions
// ------------------------------------------------------------------------------------ 

fn mat4_from_scaling(scale: vec3<f32>) -> mat4x4<f32> {
    return mat4x4<f32>(
        vec4<f32>(scale.x, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, scale.y, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, scale.z, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );
}

fn get_entity_row(entity: u32) -> u32 {
    // row_field = (chunk_index << LOCAL_SLOT_BITS) | local_index 
    return entity & ENTITY_ROW_MASK;
}

fn cubemap_direction_to_uv(direction: vec3<f32>) -> vec3<f32> {
    let abs_dir = abs(direction);
    var layer: f32;
    var texcoord: vec2<f32>;

    if (abs_dir.x > abs_dir.y && abs_dir.x > abs_dir.z) {
        layer = select(1.0, 0.0, direction.x > 0.0);
        texcoord = vec2<f32>(-direction.z, direction.y) / abs_dir.x;
    } else if (abs_dir.y > abs_dir.z) {
        layer = select(3.0, 2.0, direction.y > 0.0);
        texcoord = vec2<f32>(direction.x, -direction.z) / abs_dir.y;
    } else {
        layer = select(5.0, 4.0, direction.z > 0.0);
        texcoord = vec2<f32>(direction.x, direction.y) / abs_dir.z;
    }
    
    // Flip the x coordinate for the positive faces
    if (layer % 2.0 == 0.0) {
        texcoord.x = -texcoord.x;
    }
    // Bottom face needs additional adjustment due to other face flips
    if (layer == 3.0) {
        texcoord.y = -texcoord.y;
        texcoord.x = -texcoord.x;
    }
    
    // Convert from [-1, 1] to [0, 1]
    texcoord = texcoord * 0.5 + 0.5;

    return vec3<f32>(texcoord, layer);
}

fn random_seed(seed: u32) -> u32 {
    let x = seed * 1103515245u + 12345u;
    let y = x ^ (x >> 16u);
    return y * 2654435769u;
}

// Converts a 32-bit seed to a uniform float in the range [0,1).
fn rand_float(seed: u32) -> f32 {
  return f32(random_seed(seed)) * one_over_float_max;
}

fn dither_mask(uv: vec2<f32>, resolution: vec2<f32>) -> f32 {
    // Scale UV coordinates to the size of the screen
    let scaled_uv = uv * resolution;

    // Calculate the index in the Bayer matrix
    let x = u32(scaled_uv.x) % 4u;
    let y = u32(scaled_uv.y) % 4u;
    let index = y * 4u + x;

    // Return the dither value from the Bayer matrix
    return bayer_matrix[index];
}

fn approx(a: f32, b: f32) -> bool {
    return abs(a - b) <= epsilon; 
}

fn max3(v: vec3<f32>) -> f32 {
    return max(max(v.x, v.y), v.z);
}

fn isinf(x: f32) -> bool {
    return x == x && x != 0.0 && x * 2.0 == x;
}

fn isinf3(v: vec3<f32>) -> vec3<bool> {
    return vec3<bool>(isinf(v.x), isinf(v.y), isinf(v.z));
}

fn is_nan(x: f32) -> bool {
    return x != x;
}

fn is_nan3(v: vec3<f32>) -> vec3<bool> {
    return vec3<bool>(is_nan(v.x), is_nan(v.y), is_nan(v.z));
}

// A helper function to compute the median of three values.
fn median3(a: f32, b: f32, c: f32) -> f32 {
    // Sort the three values and pick the middle one
    // A simple way: median = a + b + c - min(a,b,c) - max(a,b,c)
    let min_val = min(a, min(b, c));

    let max_val = max(a, max(b, c));
    return (a + b + c) - min_val - max_val;
}

// Simple hash function
fn hash(x: u32) -> u32 {
    var y = x;
    y = y ^ (y >> u32(16));
    y = y * 0x85ebca6bu;
    y = y ^ (y >> u32(13));
    y = y * 0xc2b2ae35u;
    y = y ^ (y >> u32(16));
    return y;
}

// Convert uint to float in [0, 1) range
fn uint_to_normalized_float(x: u32) -> f32 {
    return f32(f32(x) * one_over_float_max);
}

// Interpolate between two values
fn interpolate(v0: f32, v1: f32, t: f32) -> f32 {
    return v0 * (1.0 - t) + v1 * t;
}

// ------------------------------------------------------------------------------------
// Bindless pool sampling helpers
// ------------------------------------------------------------------------------------
// Computes a mip level (LOD) for a given UV and texture size using screen-space
// derivatives. Intended for fragment stages where dpdx/dpdy are defined.
// - uv: normalized texture coordinates in [0,1]
// - tex_size: texture dimensions in pixels (width, height)
// Returns: log2 of the max gradient magnitude in texel space.
// Notes:
//   • For compute stages, derivatives are undefined; approximate via shared-memory
//     neighborhood gradients and feed them into a custom variant if needed.
fn compute_lod_from_uv(uv: vec2<f32>, tex_size: vec2<f32>) -> f32 {
    // Convert to texel space so gradients are measured in pixels
    let uv_texel = uv * tex_size;

    // Screen-space gradients of the texel-space coordinates
    let d_uv_dx = dpdx(uv_texel);
    let d_uv_dy = dpdy(uv_texel);

    // Max length across axes gives the footprint scale (rho)
    let rho = max(length(d_uv_dx), length(d_uv_dy));

    // Guard against log2(0). Negative LODs (minification < 1) are fine; clamp input only.
    let safe_rho = max(rho, 1e-8);
    return log2(safe_rho);
}

fn sample_handle_rgba(tex_handle: u32, uv: vec2<f32>, pool: texture_2d_array<f32>, lod: f32) -> vec4<f32> {
    let max_lod = f32(textureNumLevels(pool) - 1u);
    let clamped_lod = clamp(lod, 0.0, max(0.0, max_lod));
    return textureSampleLevel(pool, global_sampler, uv, tex_handle, clamped_lod);
}

fn sample_texture_or_vec4_param_handle(
    tex_handle: u32,
    uv_coords: vec2<f32>,
    param_val: vec4<f32>,
    flag: u32,
    pool: texture_2d_array<f32>,
    lod: f32
) -> vec4<f32> {
    if ((flag & 1u) != 0u) {
        return sample_handle_rgba(tex_handle, uv_coords, pool, lod);
    }
    return param_val;
}

fn sample_texture_or_float_param_handle(
    tex_handle: u32,
    uv_coords: vec2<f32>,
    param_val: f32,
    flag: u32,
    pool: texture_2d_array<f32>,
    lod: f32
) -> f32 {
    if ((flag & 1u) != 0u) {
        let sampled_val = sample_handle_rgba(tex_handle, uv_coords, pool, lod);
        let channel_index = (flag >> 1u) & 3u;
        return select(select(select(sampled_val.r, sampled_val.g, channel_index == 1u), sampled_val.b, channel_index == 2u), sampled_val.a, channel_index == 3u);
    }
    return param_val;
}

// Helper function to safely normalize a vector
fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
  let len = length(v);
  return select(normalize(v), vec3<f32>(0.0), len < 1e-6);
}

// A billboard function that works with local position and entity transform
// Uses model-view matrix manipulation for robust billboarding
fn billboard_vertex_local(uv: vec2<f32>, entity_transform: mat4x4<f32>) -> vec4<f32> {
    // Get view and projection matrices
    let view_index = u32(frame_info.view_index);
    let view = view_buffer[view_index].view_matrix;
    // Extract translation from the entity transform (4th column)
    let world_position = vec3<f32>(
        entity_transform[3][0],
        entity_transform[3][1],
        entity_transform[3][2]
    );
    // Extract scale from the entity transform
    // Scale is the magnitude of each of the first three column vectors
    let scale = vec3<f32>(
        length(vec3<f32>(entity_transform[0][0], entity_transform[0][1], entity_transform[0][2])),
        length(vec3<f32>(entity_transform[1][0], entity_transform[1][1], entity_transform[1][2])),
        length(vec3<f32>(entity_transform[2][0], entity_transform[2][1], entity_transform[2][2]))
    );
    // Calculate the billboard size - use the entity's scale
    // Using average of X and Y scale for consistent sizing
    let billboard_size = 0.6 * (scale.x + scale.y) * 0.5;
    // Calculate the vertex position in local space (centered quad)
    let billboard_local_pos = vec4<f32>(
        (uv.x - 0.5) * billboard_size,
        (uv.y - 0.5) * billboard_size,
        0.0,
        1.0
    );
    // Transform back to world space using the inverse view matrix
    let inverse_view = mat4x4<f32>(
        view[0][0], view[1][0], view[2][0], 0.0,
        view[0][1], view[1][1], view[2][1], 0.0,
        view[0][2], view[1][2], view[2][2], 0.0,
        0.0, 0.0, 0.0, 1.0
    );
    // Calculate the final world position
    let final_world_position = world_position + 
        inverse_view[0].xyz * billboard_local_pos.x +
        inverse_view[1].xyz * billboard_local_pos.y;
    
    return vec4<f32>(final_world_position, 1.0);
}

fn log_depth(view_space_z: f32) -> f32 {
    let view_index = u32(frame_info.view_index);
    let far_plane = -view_buffer[view_index].frustum[5].w;
    let near_plane = -view_buffer[view_index].frustum[4].w;
    let z = -view_space_z;
    return
    (log(LOG_DEPTH_C * z + 1.0) - log(LOG_DEPTH_C * near_plane + 1.0)) 
        / (log(LOG_DEPTH_C * far_plane + 1.0) - log(LOG_DEPTH_C * near_plane + 1.0));
}

fn linearize_depth(d: f32, near_plane: f32, far_plane: f32, view_index: u32) -> f32 {
    // Works for both perspective and orthographic projections.
    // Depth coming from the texture is in the [-1,1] clip-space range.
    let depth01 = (d * 0.5) + 0.5;               // -> [0,1]

    // Orthographic matrices have projection[3][3] == 1, perspective == 0
    let proj_33    = view_buffer[view_index].projection_matrix[3][3];
    let is_ortho   = proj_33 > 0.5;

    // Perspective: z_eye = (n f) / (f – depth01·(f – n))
    let persp_z = (near_plane * far_plane) /
                  (far_plane - depth01 * (far_plane - near_plane));

    // Orthographic: z_eye = n + depth01·(f – n)
    let ortho_z = near_plane + depth01 * (far_plane - near_plane);

    // `select(a, b, cond)` chooses b when cond is true
    return select(persp_z, ortho_z, is_ortho);
}

fn rotate_hue(color: vec4<f32>, hue_rotation: f32) -> vec4<f32> {
    // Convert RGB to HSV
    let rgb = color.rgb;
    let max_val = max(max(rgb.r, rgb.g), rgb.b);
    let min_val = min(min(rgb.r, rgb.g), rgb.b);
    let delta = max_val - min_val;
    
    // Calculate hue
    var hue: f32 = 0.0;
    if (delta > 0.0) {
        // Use a formula that avoids branching for hue calculation
        let r_dist = select((rgb.g - rgb.b) / delta, 0.0, max_val == rgb.r);
        let g_dist = select((rgb.b - rgb.r) / delta, 0.0, max_val == rgb.g);
        let b_dist = select((rgb.r - rgb.g) / delta, 0.0, max_val == rgb.b);
        
        hue = fract((r_dist + 6.0 * select(1.0, 0.0, max_val == rgb.r) + 
                     g_dist + 2.0 * select(1.0, 0.0, max_val == rgb.g) + 
                     b_dist + 4.0 * select(1.0, 0.0, max_val == rgb.b)) / 6.0);
    }
    
    // Calculate saturation and value
    let saturation = select(0.0, delta / max_val, max_val == 0.0);
    let value = max_val;
    
    // Apply hue rotation
    hue = fract(hue + hue_rotation / (2.0 * 3.14159265359));
    
    // Convert back to RGB using a more efficient approach
    let hue_6 = hue * 6.0;
    let hue_sector = floor(hue_6);
    let hue_fract = hue_6 - hue_sector;
    
    // Calculate the RGB components using the HSV color wheel
    let p = value * (1.0 - saturation);
    let q = value * (1.0 - saturation * hue_fract);
    let t = value * (1.0 - saturation * (1.0 - hue_fract));
    
    // Create a lookup table for the RGB values based on the hue sector
    let sector_0 = vec3<f32>(value, t, p);
    let sector_1 = vec3<f32>(q, value, p);
    let sector_2 = vec3<f32>(p, value, t);
    let sector_3 = vec3<f32>(p, q, value);
    let sector_4 = vec3<f32>(t, p, value);
    let sector_5 = vec3<f32>(value, p, q);
    
    // Select the appropriate sector using dot products with a mask
    let sector_mask = vec3<f32>(
        select(1.0, 0.0, hue_sector == 0.0 || hue_sector == 5.0),
        select(1.0, 0.0, hue_sector == 1.0 || hue_sector == 2.0),
        select(1.0, 0.0, hue_sector == 3.0 || hue_sector == 4.0)
    );
    
    let r = dot(vec3<f32>(sector_0.x, sector_1.x, sector_2.x) * sector_mask, vec3<f32>(1.0)) + 
            dot(vec3<f32>(sector_3.x, sector_4.x, sector_5.x) * sector_mask, vec3<f32>(1.0));
    let g = dot(vec3<f32>(sector_0.y, sector_1.y, sector_2.y) * sector_mask, vec3<f32>(1.0)) + 
            dot(vec3<f32>(sector_3.y, sector_4.y, sector_5.y) * sector_mask, vec3<f32>(1.0));
    let b = dot(vec3<f32>(sector_0.z, sector_1.z, sector_2.z) * sector_mask, vec3<f32>(1.0)) + 
            dot(vec3<f32>(sector_3.z, sector_4.z, sector_5.z) * sector_mask, vec3<f32>(1.0));
    
    return vec4<f32>(r, g, b, color.a);
}

// Computes the inverse of a 4x4 matrix using Cramer's rule.
// Returns the inverse matrix. If the matrix is not invertible, the result is undefined.
fn inverse4x4(m: mat4x4<f32>) -> mat4x4<f32> {
    let m00 = m[0][0]; let m01 = m[0][1]; let m02 = m[0][2]; let m03 = m[0][3];
    let m10 = m[1][0]; let m11 = m[1][1]; let m12 = m[1][2]; let m13 = m[1][3];
    let m20 = m[2][0]; let m21 = m[2][1]; let m22 = m[2][2]; let m23 = m[2][3];
    let m30 = m[3][0]; let m31 = m[3][1]; let m32 = m[3][2]; let m33 = m[3][3];

    let coef00 = m22 * m33 - m32 * m23;
    let coef02 = m12 * m33 - m32 * m13;
    let coef03 = m12 * m23 - m22 * m13;

    let coef04 = m21 * m33 - m31 * m23;
    let coef06 = m11 * m33 - m31 * m13;
    let coef07 = m11 * m23 - m21 * m13;

    let coef08 = m21 * m32 - m31 * m22;
    let coef10 = m11 * m32 - m31 * m12;
    let coef11 = m11 * m22 - m21 * m12;

    let coef12 = m20 * m33 - m30 * m23;
    let coef14 = m10 * m33 - m30 * m13;
    let coef15 = m10 * m23 - m20 * m13;

    let coef16 = m20 * m32 - m30 * m22;
    let coef18 = m10 * m32 - m30 * m12;
    let coef19 = m10 * m22 - m20 * m12;

    let coef20 = m20 * m31 - m30 * m21;
    let coef22 = m10 * m31 - m30 * m11;
    let coef23 = m10 * m21 - m20 * m11;

    let fac0 = vec4<f32>(coef00, coef00, coef02, coef03);
    let fac1 = vec4<f32>(coef04, coef04, coef06, coef07);
    let fac2 = vec4<f32>(coef08, coef08, coef10, coef11);
    let fac3 = vec4<f32>(coef12, coef12, coef14, coef15);
    let fac4 = vec4<f32>(coef16, coef16, coef18, coef19);
    let fac5 = vec4<f32>(coef20, coef20, coef22, coef23);

    let v0 = vec4<f32>(m10, m00, m00, m00);
    let v1 = vec4<f32>(m11, m01, m01, m01);
    let v2 = vec4<f32>(m12, m02, m02, m02);
    let v3 = vec4<f32>(m13, m03, m03, m03);

    let inv0 =  v1 * fac0 - v2 * fac1 + v3 * fac2;
    let inv1 = -v0 * fac0 + v2 * fac3 - v3 * fac4;
    let inv2 =  v0 * fac1 - v1 * fac3 + v3 * fac5;
    let inv3 = -v0 * fac2 + v1 * fac4 - v2 * fac5;

    let sign_a = vec4<f32>( 1.0, -1.0,  1.0, -1.0);
    let sign_b = vec4<f32>(1.0, -1.0, 1.0, -1.0);

    let col1 = inv0 * sign_a;
    let col2 = inv1 * sign_b;
    let col3 = inv2 * sign_a;
    let col4 = inv3 * sign_b;

    let row0 = vec4<f32>(col1[0], col2[0], col3[0], col4[0]);
    let det = dot(vec4<f32>(m00, m01, m02, m03), row0);

    let inverse = mat4x4<f32>(
        col1 / det,
        col2 / det,
        col3 / det,
        col4 / det
    );

    return inverse;
}

fn mask_popcount(mask: vec4<u32>) -> u32 {
    let ones = countOneBits(mask);
    return ones.x + ones.y + ones.z + ones.w;
}

fn safe_clamp_vec3(value: vec3<f32>) -> vec3<f32> {
    // Sanitize: clamp NaN/Inf/negative to 0 to prevent accumulator poisoning.
    let x = select(value.x, 0.0, is_nan(value.x) || isinf(value.x) || value.x < 0.0);
    let y = select(value.y, 0.0, is_nan(value.y) || isinf(value.y) || value.y < 0.0);
    let z = select(value.z, 0.0, is_nan(value.z) || isinf(value.z) || value.z < 0.0);
    return vec3<f32>(x, y, z);
}

// ─────────────────────────────────────────────────────────────────────────────
// Firefly-Safe Radiance Clamping
// Clamps a vec3 radiance value to a maximum luminance while preserving hue.
// This prevents fireflies by limiting extreme values while maintaining color.
// ─────────────────────────────────────────────────────────────────────────────
fn safe_clamp_vec3_max(value: vec3<f32>, max_luminance: f32) -> vec3<f32> {
    // First sanitize: remove NaN, Inf, and negative values
    let sanitized = safe_clamp_vec3(value);
    
    // Compute luminance
    let luminance = dot(sanitized, vec3<f32>(0.2126, 0.7152, 0.0722));
    
    // If luminance exceeds max, scale down proportionally to preserve hue
    let scale = select(1.0, max_luminance / luminance, luminance > max_luminance);
    
    return sanitized * scale;
}

// =============================================================================
// Halton Sequence Generation (Low-Discrepancy Sampling)
// =============================================================================
fn halton_base2(index: u32) -> f32 {
    var bits = index;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10; // / 2^32
}

fn halton_base3(index: u32) -> f32 {
    var result = 0.0;
    var f = 1.0 / 3.0;
    var i = index;
    
    for (var iter = 0u; iter < 16u; iter = iter + 1u) {
        if (i == 0u) { break; }
        result += f32(i % 3u) * f;
        i /= 3u;
        f /= 3.0;
    }
    
    return result;
}

fn halton_2d(index: u32) -> vec2<f32> {
    return vec2<f32>(halton_base2(index), halton_base3(index));
}

// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║          LOW-DISCREPANCY SEQUENCES (Scrambled Halton & Sobol)            ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  These functions provide low-discrepancy sampling with per-pixel/probe   ║
// ║  scrambling to decorrelate samples across the image while preserving     ║
// ║  the superior convergence properties of quasi-Monte Carlo sequences.     ║
// ║                                                                           ║
// ║  Usage pattern (replaces rand_float for RIS/importance sampling):        ║
// ║    let r1 = rand_halton(rng, sample_idx, 0u);  // dimension 0            ║
// ║    let r2 = rand_halton(rng, sample_idx, 1u);  // dimension 1            ║
// ║    let r3 = rand_halton(rng, sample_idx, 2u);  // dimension 2            ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

// Prime bases for Halton sequence (first 8 primes for 8 dimensions)
const HALTON_PRIMES = array<u32, 8>(2u, 3u, 5u, 7u, 11u, 13u, 17u, 19u);

// -----------------------------------------------------------------------------
// Generalized Halton sequence for arbitrary prime base
// More flexible than halton_base2/base3 - works for any dimension
// -----------------------------------------------------------------------------
fn halton_base(index: u32, base: u32) -> f32 {
    var result = 0.0;
    var f = 1.0 / f32(base);
    var i = index;
    
    // Iterate through digits in the given base (max 20 iterations for u32)
    for (var iter = 0u; iter < 20u; iter = iter + 1u) {
        if (i == 0u) { break; }
        result += f32(i % base) * f;
        i /= base;
        f /= f32(base);
    }
    
    return result;
}

// -----------------------------------------------------------------------------
// Scrambled Halton Sequence (Cranley-Patterson Rotation)
// -----------------------------------------------------------------------------
// Uses the RNG to generate a per-pixel/probe offset that decorrelates the
// sequence while preserving low-discrepancy properties within each pixel.
//
// Parameters:
//   rng         - Scrambling seed (derived from pixel/probe position + frame)
//   sample_idx  - Index in the sequence (0, 1, 2, ... for each sample)
//   dimension   - Which dimension to sample (0, 1, 2, ... up to 7)
//
// Returns: A value in [0, 1) with low-discrepancy properties
// -----------------------------------------------------------------------------
fn rand_halton(rng: u32, sample_idx: u32, dimension: u32) -> f32 {
    let base = HALTON_PRIMES[dimension % 8u];
    
    // +1 to sample_idx avoids first sample always being 0
    let halton_val = halton_base(sample_idx + 1u, base);
    
    // Cranley-Patterson rotation: add per-dimension random offset
    // Using hash to decorrelate dimensions from each other
    let scramble = rand_float(hash(rng ^ (dimension * 0x9E3779B9u)));
    
    return fract(halton_val + scramble);
}

// =============================================================================
// Sobol Sequence Implementation
// =============================================================================
// Sobol sequences are another family of low-discrepancy sequences with
// excellent multidimensional uniformity. They use direction vectors to
// generate samples via XOR operations.
// =============================================================================

// Direction vectors for Sobol sequence (first 4 dimensions)
// These are standard Sobol direction numbers shifted to fill 32 bits
const SOBOL_V0 = array<u32, 16>(
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
    0x08000000u, 0x04000000u, 0x02000000u, 0x01000000u,
    0x00800000u, 0x00400000u, 0x00200000u, 0x00100000u,
    0x00080000u, 0x00040000u, 0x00020000u, 0x00010000u
);

const SOBOL_V1 = array<u32, 16>(
    0x80000000u, 0xc0000000u, 0xa0000000u, 0xf0000000u,
    0x88000000u, 0xcc000000u, 0xaa000000u, 0xff000000u,
    0x80800000u, 0xc0c00000u, 0xa0a00000u, 0xf0f00000u,
    0x88880000u, 0xcccc0000u, 0xaaaa0000u, 0xffff0000u
);

const SOBOL_V2 = array<u32, 16>(
    0x80000000u, 0xc0000000u, 0x60000000u, 0x90000000u,
    0xe8000000u, 0x5c000000u, 0x8e000000u, 0xc5000000u,
    0x68800000u, 0x9cc00000u, 0xee600000u, 0x55900000u,
    0x80e80000u, 0xc05c0000u, 0x608e0000u, 0x90c50000u
);

const SOBOL_V3 = array<u32, 16>(
    0x80000000u, 0xc0000000u, 0x20000000u, 0x50000000u,
    0xf8000000u, 0x74000000u, 0xa2000000u, 0x93000000u,
    0xd8800000u, 0x25400000u, 0x59e00000u, 0xe6d00000u,
    0x78080000u, 0xb40c0000u, 0x82020000u, 0xc3050000u
);

// -----------------------------------------------------------------------------
// Core Sobol sample generation for a single dimension
// -----------------------------------------------------------------------------
fn sobol_sample_dim(index: u32, dimension: u32) -> u32 {
    var result = 0u;
    var i = index;
    var bit = 0u;
    
    // XOR direction vectors based on which bits are set in the index
    while (i != 0u && bit < 16u) {
        if ((i & 1u) != 0u) {
            // Select direction vector based on dimension
            let v = select(
                select(
                    select(SOBOL_V3[bit], SOBOL_V2[bit], dimension == 2u),
                    SOBOL_V1[bit],
                    dimension == 1u
                ),
                SOBOL_V0[bit],
                dimension == 0u
            );
            result ^= v;
        }
        i >>= 1u;
        bit += 1u;
    }
    
    return result;
}

// -----------------------------------------------------------------------------
// Scrambled Sobol Sequence (XOR Scrambling / Owen-like)
// -----------------------------------------------------------------------------
// Uses the RNG to generate a per-pixel/probe XOR mask that decorrelates
// the sequence while preserving stratification properties.
//
// Parameters:
//   rng         - Scrambling seed (derived from pixel/probe position + frame)
//   sample_idx  - Index in the sequence (0, 1, 2, ... for each sample)
//   dimension   - Which dimension to sample (0, 1, 2, 3 supported)
//
// Returns: A value in [0, 1) with low-discrepancy properties
// -----------------------------------------------------------------------------
fn rand_sobol(rng: u32, sample_idx: u32, dimension: u32) -> f32 {
    // Generate the raw Sobol sample (+1 to avoid index 0)
    let sobol_raw = sobol_sample_dim(sample_idx + 1u, dimension % 4u);
    
    // XOR scrambling with per-dimension random mask
    // This is a simplified form of Owen scrambling
    let scramble_mask = hash(rng ^ (dimension * 0x85ebca6bu));
    let scrambled = sobol_raw ^ scramble_mask;
    
    // Convert to [0, 1) float
    return f32(scrambled) * 2.3283064365386963e-10;  // = 1.0 / 2^32
}

// -----------------------------------------------------------------------------
// Convenience: 2D/3D sampling helpers
// -----------------------------------------------------------------------------
fn rand_halton_2d(rng: u32, sample_idx: u32) -> vec2<f32> {
    return vec2<f32>(
        rand_halton(rng, sample_idx, 0u),
        rand_halton(rng, sample_idx, 1u)
    );
}

fn rand_halton_3d(rng: u32, sample_idx: u32) -> vec3<f32> {
    return vec3<f32>(
        rand_halton(rng, sample_idx, 0u),
        rand_halton(rng, sample_idx, 1u),
        rand_halton(rng, sample_idx, 2u)
    );
}

fn rand_sobol_2d(rng: u32, sample_idx: u32) -> vec2<f32> {
    return vec2<f32>(
        rand_sobol(rng, sample_idx, 0u),
        rand_sobol(rng, sample_idx, 1u)
    );
}

fn rand_sobol_3d(rng: u32, sample_idx: u32) -> vec3<f32> {
    return vec3<f32>(
        rand_sobol(rng, sample_idx, 0u),
        rand_sobol(rng, sample_idx, 1u),
        rand_sobol(rng, sample_idx, 2u)
    );
}

// ============================================================================
// O(1) Helper function to compute pixel coordinates from linear index
// ============================================================================
// This computes the 2D pixel coordinate for checkerboard/interlaced rendering
// patterns where pixels are sampled at intervals of trace_rate.
// The pattern shifts by 2 pixels per row to maintain temporal stability.
// ============================================================================
fn compute_phased_pixel_coords(linear_index: u32, res: vec2<u32>, trace_rate: u32, frame_phase: u32) -> vec2<u32> {
    // Fast path: full resolution (no checkerboarding)
    if (trace_rate <= 1u) {
        return vec2<u32>(linear_index % res.x, linear_index / res.x);
    }
    
    // Compute average pixels per row
    let pixels_per_row = res.x / trace_rate;
    
    // Direct mathematical computation of row and offset
    let estimated_y = linear_index / max(pixels_per_row, 1u);
    let offset_in_row = linear_index % max(pixels_per_row, 1u);
    
    // Compute the first x position for the estimated row
    // Pattern: shifts by 2 mod trace_rate each row for temporal coherence
    let first_x = (frame_phase + trace_rate - (estimated_y * 2u) % trace_rate) % trace_rate;
    
    // Compute actual pixels available in this row given the first_x position
    let actual_pixels_in_row = (res.x + trace_rate - 1u - first_x) / trace_rate;
    
    // Check if our offset fits in this row or if we need to adjust
    var pixel_coords = vec2u(first_x + offset_in_row * trace_rate, estimated_y);
    if (offset_in_row >= actual_pixels_in_row) {
        // Rare case: offset spills into next row due to alignment mismatch
        // This happens when first_x causes the row to have fewer pixels
        pixel_coords.y = estimated_y + 1u;
        let next_first_x = (frame_phase + trace_rate - (pixel_coords.y * 2u) % trace_rate) % trace_rate;
        pixel_coords.x = next_first_x + (offset_in_row - actual_pixels_in_row) * trace_rate;
    }

    return pixel_coords;
}

// Copy the sign bit from B onto A.
fn copysign(a: f32, b: f32) -> f32 {
    return bitcast<f32>((bitcast<u32>(a) & 0x7FFFFFFF) | (bitcast<u32>(b) & 0x80000000));
}

// Constructs a right-handed orthonormal basis from a given unit Z vector.
fn orthonormalize(z_basis: vec3<f32>) -> mat3x3<f32> {
    let sign = copysign(1.0, z_basis.z);
    let a = -1.0 / (sign + z_basis.z);
    let b = z_basis.x * z_basis.y * a;
    let x_basis = vec3(1.0 + sign * z_basis.x * z_basis.x * a, sign * b, -sign * z_basis.x);
    let y_basis = vec3(b, sign + z_basis.y * z_basis.y * a, -z_basis.y);
    return mat3x3(x_basis, y_basis, z_basis);
}

fn luminance(v: vec3<f32>) -> f32 {
    return v.x * 0.2126 + v.y * 0.7152 + v.z * 0.0722;
}

fn uv_to_coord(uv: vec2<f32>, resolution: vec2<u32>) -> vec2<i32> {
    let pixel = vec2<i32>(floor(uv * vec2<f32>(f32(resolution.x), f32(resolution.y))));
    return vec2<i32>(
        clamp(pixel.x, 0, i32(resolution.x) - 1),
        clamp(pixel.y, 0, i32(resolution.y) - 1)
    );
}

fn coord_to_uv(coord: vec2<i32>, resolution: vec2<u32>) -> vec2<f32> {
    return (vec2<f32>(coord) + 0.5) / vec2<f32>(f32(resolution.x), f32(resolution.y));
}

fn reconstruct_world_position(uv: vec2<f32>, depth: f32, view_index: u32) -> vec3<f32> {
    let clip = vec4<f32>(
        uv.x * 2.0 - 1.0,
        (1.0 - uv.y) * 2.0 - 1.0,
        depth,
        1.0
    );
    let world = view_buffer[view_index].inverse_view_projection_matrix * clip;
    return world.xyz / world.w;
}

fn reconstruct_prev_world_position(uv: vec2<f32>, depth: f32, view_index: u32) -> vec3<f32> {
    let clip = vec4<f32>(
        uv.x * 2.0 - 1.0,
        (1.0 - uv.y) * 2.0 - 1.0,
        depth,
        1.0
    );
    let world = view_buffer[view_index].prev_inverse_view_projection_matrix * clip;
    return world.xyz / world.w;
}
