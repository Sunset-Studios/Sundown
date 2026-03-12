// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                    GI COMMON DEFINITIONS                                  ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Shared structures and utilities for the per-pixel GI system:             ║
// ║  • GI parameter structures                                                ║
// ║  • Path state for ray tracing                                             ║
// ║  • Tile-based stochastic sampling helpers                                 ║
// ║  • Direction encoding/decoding                                            ║
// ║  • Temporal blending algorithms                                           ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

#include "lighting_common.wgsl"

// =============================================================================
// CONSTANTS
// =============================================================================
const MAX_HIT_DISTANCE: f32 = 65504.0;
const MAX_RADIANCE_LUMINANCE = 10.0;
const MAX_NEE_LUMINANCE = 10.0;

// =============================================================================
// GI COUNTERS
// =============================================================================

struct GICounters {
    light_count: u32,
    active_cache_cell_count: atomic<u32>,
    ray_queue_shadow_head: atomic<u32>,      // Shadow ray work queue consumer head
    ray_queue_primary_head: atomic<u32>,     // Primary ray work queue consumer head
    ray_queue_count: atomic<u32>,            // Active rays added to work queue (atomic for conditional add)
    probe_update_count: atomic<u32>,         // DDGI probe update count (compacted active probes)
};

struct GICountersReadOnly {
    light_count: u32,
    active_cache_cell_count: u32,
    ray_queue_shadow_head: u32,
    ray_queue_primary_head: u32,
    ray_queue_count: u32,
    probe_update_count: u32,
};

// =============================================================================
// GI PARAMETERS
// =============================================================================

struct GIParams {
    screen_ray_count: f32,          // Rays per tile per frame
    world_cache_size: f32,          // Number of world cache entries per LOD
    world_cache_cell_size: f32,     // Base cell size in world units
    total_pixels: f32,              // Total pixels (width * height)
    frame_index: f32,               // Current frame index
    indirect_boost: f32,            // Indirect lighting multiplier
    upscale_factor: f32,            // GI internal resolution scale factor (1, 2, 4, ...)
    world_cache_lod_count: f32,     // Number of LOD levels for world cache
    full_resolution_x: f32,         // Full-resolution X (GBuffer / lighting target)
    full_resolution_y: f32,         // Full-resolution Y (GBuffer / lighting target)
    gi_resolution_x: f32,           // GI internal resolution X (full_resolution / upscale_factor)
    gi_resolution_y: f32,           // GI internal resolution Y (full_resolution / upscale_factor)
    max_ray_length: f32,            // Maximum ray travel distance
};

// =============================================================================
// PER-PIXEL PATH STATE
// 
// Stores the complete state of a per-pixel ray for multi-pass path tracing.
// Each tile traces screen_ray_count rays per frame.
// =============================================================================

struct PixelPathState {
    origin_tmin: vec4<f32>,              // xyz = ray origin, w = t_min
    direction_tmax: vec4<f32>,           // xyz = ray direction, w = t_max / prim_store
    normal_section_index: vec4<f32>,     // xyz = hit normal, w = section index
    state_u32: vec4<u32>,                // x = lobe_type (0 = diffuse, 1 = specular), y = alive, z = shadow_visible, w = tri_id
    hit_attr0: vec4<f32>,                // xyz = tangent, w = uv.x
    hit_attr1: vec4<f32>,                // xyz = bitangent, w = uv.y
    shadow_origin: vec4<f32>,            // xyz = shadow ray origin, w = light index
    shadow_direction: vec4<f32>,         // xyz = shadow ray direction, w = max distance
    shadow_radiance: vec4<f32>,          // xyz = potential light contribution, w = weight
    path_weight: vec4<f32>,              // xyz = BRDF weight, w = source PDF
    rng_sample_count_frame_stamp: vec4<f32>, // x = RNG state, y = sample count, z = frame, w = unused
    throughput_direct: vec4<f32>,               // xyz = accumulated direct radiance, w = unused
    throughput_indirect_diffuse: vec4<f32>,     // xyz = accumulated indirect diffuse radiance, w = unused
    throughput_indirect_specular: vec4<f32>,    // xyz = accumulated indirect specular radiance, w = unused
    pixel_coords: vec4<f32>,             // xy = pixel coordinates, zw = unused
};

// =============================================================================
// AO PER-PIXEL PATH STATE
//
// Stores the state of a per-pixel ray for AO.
// Each tile traces screen_ray_count rays per frame.
// =============================================================================
struct AOPixelPathState {
    origin_tmin: vec4<f32>,              // xyz = ray origin, w = t_min
    direction_tmax: vec4<f32>,           // xyz = ray direction, w = t_max / prim_store
    state_u32: vec4<u32>,                // x = lobe_type (0 = diffuse, 1 = specular), y = alive, z = shadow_visible, w = tri_id
}

// =============================================================================
// WORLD CACHE PATH STATE
// =============================================================================

struct WorldCachePathState {
    origin_tmin: vec4<f32>,
    direction_tmax: vec4<f32>,
    normal_section_index: vec4<f32>,
    state_u32: vec4<u32>,
    hit_attr0: vec4<f32>,
    hit_attr1: vec4<f32>,
    shadow_origin: vec4<f32>,
    shadow_direction: vec4<f32>,
    shadow_radiance: vec4<f32>,
    path_weight: vec4<f32>,
    rng_rank_frame_stamp: vec4<f32>,
};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// GI resolution helpers
// 
// We run PTGI at a reduced internal resolution (gi_resolution_*), while
// sampling geometry from the full-resolution GBuffer (full_resolution_*).
// This helper maps a GI pixel coordinate to a representative full-res pixel
// coordinate inside its upscale_factor×upscale_factor footprint.
// ─────────────────────────────────────────────────────────────────────────────
fn gi_pixel_to_full_res_pixel_coord(
    gi_pixel_coord: vec2<u32>,
    upscale_factor: u32,
    full_resolution: vec2<u32>
) -> vec2<u32> {
    let full_x = min(gi_pixel_coord.x * upscale_factor, full_resolution.x - 1u);
    let full_y = min(gi_pixel_coord.y * upscale_factor, full_resolution.y - 1u);
    return vec2<u32>(full_x, full_y);
}

// ─────────────────────────────────────────────────────────────────────────────
// Pack half float into u32 (for distance comparison)
// ─────────────────────────────────────────────────────────────────────────────
fn pack_half_float(value: f32) -> u32 {
    let clamped = clamp(value, 0.0, 65504.0);
    return u32(clamped * 2.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Biased Temporal Hysteresis
// Adapts blend factor based on luminance difference for shadow preservation
// ─────────────────────────────────────────────────────────────────────────────
fn temporal_blend(curr_radiance: vec3<f32>, prev_radiance: vec3<f32>) -> vec3<f32> {
    let l1 = dot(curr_radiance, vec3<f32>(1.0 / 3.0));
    let l2 = dot(prev_radiance, vec3<f32>(1.0 / 3.0));
    
    let numerator = max(l1 - l2 - min(l1, l2), 0.0);
    let denominator = max(max(l1, l2), 1e-4);
    var alpha = numerator / denominator;
    
    alpha = clamp(alpha, 0.0, 0.95);
    alpha = alpha * alpha;
    
    return mix(curr_radiance, prev_radiance, alpha);
}

// ─────────────────────────────────────────────────────────────────────────────
// Cosine-weighted hemisphere sampling
// ─────────────────────────────────────────────────────────────────────────────
fn sample_cosine_hemisphere(u1: f32, u2: f32, normal: vec3<f32>) -> vec3<f32> {
    let phi = 2.0 * PI * u1;
    let cos_theta = sqrt(1.0 - u2);
    let sin_theta = sqrt(u2);
    
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(normal.y) > 0.999);
    let tangent = normalize(cross(up, normal));
    let bitangent = normalize(cross(normal, tangent));
    
    let dir_local = vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    return normalize(tangent * dir_local.x + bitangent * dir_local.y + normal * dir_local.z);
}

// =============================================================================
// OCTAHEDRAL DIRECTION ENCODING
// =============================================================================

fn encode_octahedral(direction: vec3<f32>) -> vec2<f32> {
    let normal = safe_normalize(direction);
    return encode_octahedral_normalized(normal);
}

// Same as encode_octahedral but assumes direction is already unit length (avoids redundant normalize).
fn encode_octahedral_normalized(normal: vec3<f32>) -> vec2<f32> {
    var projected = normal.xy / max(abs(normal.x) + abs(normal.y) + abs(normal.z), 1e-6);
    let wrap_sign = vec2<f32>(
        select(-1.0, 1.0, projected.x >= 0.0),
        select(-1.0, 1.0, projected.y >= 0.0)
    );
    let wrapped = (vec2<f32>(1.0) - abs(projected.yx)) * wrap_sign;
    projected = select(projected, wrapped, normal.z <= 0.0);
    return projected * 0.5 + 0.5;
}

fn decode_octahedral(encoded: vec2<f32>) -> vec3<f32> {
    let f = encoded * 2.0 - 1.0;
    var normal = vec3<f32>(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    let wrap_sign = vec2<f32>(
        select(-1.0, 1.0, normal.x >= 0.0),
        select(-1.0, 1.0, normal.y >= 0.0)
    );
    let wrapped = (vec2<f32>(1.0) - abs(normal.yx)) * wrap_sign;
    normal = select(normal, vec3<f32>(wrapped.x, wrapped.y, normal.z), normal.z < 0.0);
    return safe_normalize(normal);
}

// =============================================================================
// TILE-BASED STOCHASTIC SAMPLING
// 
// Instead of dispatching all pixels and doing early-outs, we dispatch only
// the number of tiles (upscale_factor per tile) and randomly select
// a pixel within each tile. This is more efficient and provides better
// temporal sampling distribution.
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// Tile-to-pixel selection using explicit random values
// Allows using blue noise or white noise for tile selection
// ─────────────────────────────────────────────────────────────────────────────
fn tile_to_pixel_with_offset(
    tile_index: u32,
    tile_grid_width: u32,
    upscale_factor: u32,
    resolution: vec2<u32>,
    rand_offset_x: f32,
    rand_offset_y: f32
) -> vec2<u32> {
    // Compute tile coordinates from linear tile index
    let tile_x = tile_index % tile_grid_width;
    let tile_y = tile_index / tile_grid_width;
    
    // Compute tile corner in pixel space
    let tile_corner_x = tile_x * upscale_factor;
    let tile_corner_y = tile_y * upscale_factor;
    
    // Use provided random values for offset within tile
    let offset_x = u32(rand_offset_x * f32(upscale_factor)) % upscale_factor;
    let offset_y = u32(rand_offset_y * f32(upscale_factor)) % upscale_factor;
    
    // Compute final pixel coordinates (clamp to resolution bounds)
    let pixel_x = min(tile_corner_x + offset_x, resolution.x - 1u);
    let pixel_y = min(tile_corner_y + offset_y, resolution.y - 1u);
    
    return vec2<u32>(pixel_x, pixel_y);
}

