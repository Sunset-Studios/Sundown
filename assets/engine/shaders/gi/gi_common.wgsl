// =============================================================================
// GI-1.0 Common Definitions
// Shared structures and utilities for the GI system
// =============================================================================
#include "lighting_common.wgsl"

// =============================================================================
// Shared Probe Encoding Parameters
// =============================================================================
const MAX_SCREEN_PROBE_SIZE = 32u;                     // Supports up to 32x32 tiles
const MAX_SCREEN_PROBE_PIXEL_COUNT = MAX_SCREEN_PROBE_SIZE * MAX_SCREEN_PROBE_SIZE;

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
    active_cache_cell_count: atomic<u32>,
    _padding1: u32,
};

// =============================================================================
// Tile classification counters
// =============================================================================
struct TileCounters {
    empty_count: atomic<u32>,
    override_count: atomic<u32>,
    padding0: u32,
    padding1: u32,
}

// =============================================================================
// GI Parameters
// =============================================================================
struct GIParams {
    screen_probe_size: f32,         // Side length of square probe footprint in pixels
    screen_ray_count: f32,          // Rays per screen probe
    world_cache_size: f32,          // Number of world cache entries
    world_cache_cell_size: f32,     // Size of world cache cells in world units
    total_screen_probes: f32,       // Total probes in grid (derived from resolution)
    frame_index: f32,               // Current frame for temporal updates
    indirect_boost: f32,            // Indirect lighting multiplier (f32 bits)
    upscale_x: f32,                 // Temporal upscale factor X
    upscale_y: f32,                 // Temporal upscale factor Y
    world_cache_lod_count: f32,     // Number of LOD levels for world cache
    trace_rate: f32,                // Trace rate for path tracing
    padding: f32,                   // Padding
};

// =============================================================================
// Screen Probe
// =============================================================================
struct ScreenProbe {
    state: vec4<f32>,               // x = active(0/1), y = pixel_x, z = pixel_y, w = updated_this_frame(0/1)
};

// =============================================================================
// Probe Path State
// =============================================================================
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
    reservoir_radiance_m: vec4<f32>,
    reservoir_direction_w: vec4<f32>,
};

struct WorldCachePathState {
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
// OCTAHEDRAL DIRECTION ENCODING
// =============================================================================
fn encode_octahedral(direction: vec3<f32>) -> vec2<f32> {
    let normal = safe_normalize(direction);
    var projected = normal.xy / max(abs(normal.x) + abs(normal.y) + abs(normal.z), 1e-6);
    let wrap_sign = vec2<f32>(
        select(-1.0, 1.0, projected.x >= 0.0),
        select(-1.0, 1.0, projected.y >= 0.0)
    );
    let wrapped = (vec2<f32>(1.0) - abs(projected.yx)) * wrap_sign;
    projected = select(projected, wrapped, normal.z < 0.0);

    return projected * 0.5 + 0.5;
}

fn decode_octahedral(encoded: vec2<f32>) -> vec3<f32> {
    let f = encoded * 2.0 - 1.0;
    var normal = vec3<f32>(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    let t = clamp(-normal.z, 0.0, 1.0);
    let correction = vec2<f32>(
        select(-t, t, f.x >= 0.0),
        select(-t, t, f.y >= 0.0)
    );
    normal = vec3<f32>(f.x + correction.x, f.y + correction.y, normal.z);
    return safe_normalize(normal);
}

fn direction_to_probe_local_coord(direction: vec3<f32>, probe_size: u32) -> vec2<u32> {
    let encoded = encode_octahedral(direction);
    let clamped_size = max(min(probe_size, u32(MAX_SCREEN_PROBE_SIZE)), 1u);
    let scaled = clamp(
        encoded * vec2<f32>(f32(clamped_size)),
        vec2<f32>(0.0),
        vec2<f32>(f32(clamped_size) - 1e-4)
    );
    let coord = vec2<u32>(scaled);
    return coord;
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