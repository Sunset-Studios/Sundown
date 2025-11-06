// =============================================================================
// GI-1.0 Common Definitions
// Shared structures and utilities for the GI system
// =============================================================================
#include "lighting_common.wgsl"

// =============================================================================
// GI Counters
// - light_count: Number of lights in the scene (copied from lighting system)
// - active_probe_count: Number of probes updated THIS frame (reset each frame)
// 
// Note: Total probe count is derived from grid dimensions and stored in GIParams.total_screen_probes
// =============================================================================
struct GICounters {
    light_count: u32,                      // Number of lights
    active_probe_count: atomic<u32>,       // Probes updated this frame (resets)
    _padding0: u32,
    _padding1: u32,
};

struct GIParams {
    screen_probe_size: f32,         // Side length of square probe footprint in pixels
    screen_ray_count: f32,          // Rays per screen probe
    world_cache_size: f32,          // Number of world cache entries
    world_cache_cell_size: f32,     // Size of world cache cells in world units
    total_screen_probes: f32,       // Total probes in grid (derived from resolution)
    frame_index: f32,               // Current frame for temporal updates
    reset_caches: f32,              // Force reset flag
    indirect_boost: f32,            // Indirect lighting multiplier (f32 bits)
    upscale_x: f32,                 // Temporal upscale factor X
    upscale_y: f32,                 // Temporal upscale factor Y
    world_cache_lod_count: f32,     // Number of LOD levels for world cache
    _padding0: f32,
};

struct ScreenProbe {
    position_radius: vec4<f32>,     // xyz = world position, w = influence radius
    normal_frame: vec4<f32>,        // xyz = normal, w = frame stamp
    radiance_m: vec4<f32>,          // xyz = accumulated radiance, w = sample count (M)
    albedo_roughness: vec4<f32>,    // xyz = albedo, w = roughness
    material_props: vec4<f32>,      // x = metallic, y = reflectance, z = emissive, w = unused
    state: vec4<f32>,               // x = active(0/1), y = pixel_x, z = pixel_y, w = updating(0/1)
};

struct ProbePathState {
    origin_tmin: vec4<f32>,
    direction_tmax: vec4<f32>,
    normal_section_index: vec4<f32>,
    state_u32: vec4<u32>,      // x=bounce, y=alive, z=unused, w=tri_id
    hit_attr0: vec4<f32>,
    hit_attr1: vec4<f32>,
    shadow_origin: vec4<f32>,
    shadow_direction: vec4<f32>,
    shadow_radiance: vec4<f32>,
    path_weight: vec4<f32>,
    rng_sample_count_frame_stamp: vec4<f32>,
    throughput: vec4<f32>,
};

struct ProbePathShade {
    reservoir_radiance_m: vec4<f32>,
    reservoir_direction_w: vec4<f32>,
};

// =============================================================================
// Helper: Pack half float into u32
// =============================================================================
fn pack_half_float(value: f32) -> u32 {
    let clamped = clamp(value, 0.0, 65504.0);
    return u32(clamped * 2.0); // Simple packing (not true fp16, but sufficient for distance comparison)
}

// =============================================================================
// Biased Temporal Hysteresis (GI-1.0 Algorithm 3)
// - Adapts blend factor based on luminance difference
// - Preserves shadows and occlusion better than exponential moving average
// - Acts as firefly removal by filtering out transient bright signals
// =============================================================================
fn temporal_blend(curr_radiance: vec3<f32>, prev_radiance: vec3<f32>) -> vec3<f32> {
    // Compute luminance using equal weighting (1/3, 1/3, 1/3)
    let l1 = dot(curr_radiance, vec3<f32>(1.0 / 3.0));
    let l2 = dot(prev_radiance, vec3<f32>(1.0 / 3.0));
    
    // Compute adaptive alpha based on normalized difference
    // Bias towards darker values to preserve shadows
    let numerator = max(l1 - l2 - min(l1, l2), 0.0);
    let denominator = max(max(l1, l2), 1e-4);
    var alpha = numerator / denominator;
    
    // Clamp and remap with squared falloff
    alpha = clamp(alpha, 0.0, 0.95);
    alpha = alpha * alpha;
    
    // Blend: higher alpha = more previous radiance (temporal stability)
    return mix(curr_radiance, prev_radiance, alpha);
}

// Generate cosine-weighted hemisphere sample
fn sample_cosine_hemisphere(u1: f32, u2: f32, normal: vec3<f32>) -> vec3<f32> {
    let r = sqrt(u1);
    let theta = 2.0 * PI * u2;
    
    let x = r * cos(theta);
    let y = r * sin(theta);
    let z = sqrt(max(0.0, 1.0 - u1));
    
    // Build TBN frame
    // When normal is aligned with Y-axis, use X-axis as up; otherwise use Y-axis
    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(normal.y) < 0.999);
    let tangent = normalize(cross(up, normal));
    let bitangent = normalize(cross(normal, tangent));
    
    return normalize(tangent * x + bitangent * y + normal * z);
}

// =============================================================================
// PROBE GRID HELPER
// =============================================================================
fn grid_dimensions(resolution: vec2<u32>, probe_size: u32) -> vec2<u32> {
    return (resolution + probe_size - 1u) / probe_size;
}

// =============================================================================
// TEMPORAL UPSCALE SELECTION
// Determines if this probe tile should be updated this frame
// =============================================================================
fn should_update_probe_this_frame(
    probe_tile_coords: vec2<u32>,
    frame_index: u32,
    upscale: vec2<u32>
) -> bool {
    let total_frames = upscale.x * upscale.y;
    let frame_in_cycle = frame_index % total_frames;
    
    // Create 2D tiling pattern: map tile coords to frame within upscale block
    let tile_in_block_x = probe_tile_coords.x % upscale.x;
    let tile_in_block_y = probe_tile_coords.y % upscale.y;
    let probe_frame = tile_in_block_y * upscale.x + tile_in_block_x;
    
    return probe_frame == frame_in_cycle;
}

// Compute weight for a probe contribution
fn compute_probe_weight(
    pixel_pos: vec3<f32>,
    pixel_normal: vec3<f32>,
    probe_pos: vec3<f32>,
    probe_normal: vec3<f32>,
    probe_radius: f32
) -> f32 {
    // Distance-based weight
    let dist = length(pixel_pos - probe_pos);
    let dist_weight = max(0.0, 1.0 - dist / max(0.001, probe_radius));
    
    // Normal similarity weight
    let normal_dot = max(0.0, dot(pixel_normal, probe_normal));
    let normal_weight = pow(normal_dot, 4.0); // Higher power for sharper falloff
    
    // View direction weight (prefer probes in front of surface)
    let to_probe = normalize(probe_pos - pixel_pos);
    let view_weight = max(0.0, dot(pixel_normal, to_probe));
    
    return dist_weight * normal_weight * view_weight;
}