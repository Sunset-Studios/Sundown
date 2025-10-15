// =============================================================================
// GI-1.0 Screen Probe Spawning
// - Spawns screen-space probes directly on primary visible surfaces
// - Uses stochastic sampling to place probes based on spawn rate
// - Manages probe lifecycle and temporal stability
// =============================================================================
#include "common.wgsl"

// Configuration for screen probe placement
struct GIParams {
    screen_probe_spawn_rate: u32,  // 1 in N pixels spawns a probe (e.g., 16 = 1/16 pixels)
    screen_probe_size: u32,         // Side length of square probe footprint in pixels
    screen_ray_count: u32,          // Rays per screen probe
    world_cache_size: u32,          // Number of world cache entries
    max_screen_probes: u32,         // Maximum number of screen probes
    frame_index: u32,               // Current frame for temporal updates
    reset_caches: u32,              // Force reset flag
    indirect_boost: u32,            // Indirect lighting multiplier (f32 bits)
    upscale_x: u32,                 // Temporal upscale factor X
    upscale_y: u32,                 // Temporal upscale factor Y
    cell_size_heuristic: u32,       // Spatial error tolerance (f32 bits)
    padding: u32,                   // Padding for alignment
};

// Screen probe data - stores incoming radiance at primary surfaces
struct ScreenProbe {
    position_radius: vec4<f32>,     // xyz = world position, w = influence radius
    normal_frame: vec4<f32>,        // xyz = normal, w = frame stamp
    radiance_m: vec4<f32>,          // xyz = accumulated radiance, w = sample count (M)
    albedo_roughness: vec4<f32>,    // xyz = albedo, w = roughness
    state: vec4<u32>,               // x = active(0/1), y = pixel_x, z = pixel_y, w = probe_id
};

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read_write> screen_probes: array<ScreenProbe>;
@group(1) @binding(2) var<storage, read_write> screen_probe_counter: array<atomic<u32>>;
@group(1) @binding(3) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(4) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(5) var gbuffer_albedo: texture_2d<f32>;
@group(1) @binding(6) var gbuffer_smra: texture_2d<f32>;
@group(1) @binding(7) var gbuffer_motion: texture_2d<f32>;

// =============================================================================
// Halton Sequence Generation (Low-Discrepancy Sampling)
// =============================================================================

// Halton sequence base 2 (van der Corput sequence)
fn halton_base2(index: u32) -> f32 {
    var bits = index;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10; // / 2^32
}

// Halton sequence base 3
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

// Generate 2D Halton sample
fn halton_2d(index: u32) -> vec2<f32> {
    return vec2<f32>(halton_base2(index), halton_base3(index));
}

// =============================================================================
// Temporal Upscale Logic (GI-1.0 Paper Section 2.1.1)
// =============================================================================

// Compute spawn tile coordinates from pixel coordinates
fn get_spawn_tile_coords(pixel_coords: vec2<u32>, upscale: vec2<u32>) -> vec2<u32> {
    let probe_size = 8u;
    let spawn_tile_size = probe_size * upscale;
    return pixel_coords / spawn_tile_size;
}

// Check if this pixel should spawn a probe this frame using Halton sequence
fn should_spawn_probe_temporal(
    pixel_coords: vec2<u32>,
    frame_index: u32,
    upscale: vec2<u32>
) -> bool {
    let probe_size = 8u;
    let spawn_tile_size = probe_size * upscale;
    
    // Get spawn tile coordinates
    let tile_coords = pixel_coords / spawn_tile_size;
    let tile_local = pixel_coords % spawn_tile_size;
    
    // Total frames needed to fill all probes
    let total_frames = upscale.x * upscale.y;
    let frame_in_cycle = frame_index % total_frames;
    
    // Generate Halton jitter for this frame in the cycle
    let halton_sample = halton_2d(frame_in_cycle);
    let jitter_x = u32(halton_sample.x * f32(spawn_tile_size.x));
    let jitter_y = u32(halton_sample.y * f32(spawn_tile_size.y));
    
    // Check if this pixel matches the jittered position
    return tile_local.x == jitter_x && tile_local.y == jitter_y;
}

// =============================================================================
// Collaborative Probe Reprojection (Algorithm 1 from GI-1.0 Paper)
// =============================================================================

// Shared memory for workgroup collaboration
var<workgroup> best_prev_probe_index: atomic<u32>;
var<workgroup> best_prev_probe_error: atomic<u32>; // Stored as bits of f32

// Compute error metric for probe reuse
fn compute_reuse_error(
    current_pos: vec3<f32>,
    current_normal: vec3<f32>,
    prev_pos: vec3<f32>,
    prev_normal: vec3<f32>,
    cell_size: f32
) -> f32 {
    // Position error
    let pos_dist = length(current_pos - prev_pos);
    let pos_error = pos_dist / max(cell_size, 0.01);
    
    // Normal similarity error
    let normal_similarity = dot(current_normal, prev_normal);
    let normal_error = 1.0 - max(0.0, normal_similarity);
    
    // Combined error (lower is better)
    return pos_error + normal_error * 2.0;
}

// Find best probe from previous frame for temporal reuse
// Returns probe index or 0xFFFFFFFFu if no suitable probe found
fn find_best_previous_probe(
    current_pixel: vec2<u32>,
    current_pos: vec3<f32>,
    current_normal: vec3<f32>,
    local_id: vec3<u32>,
    cell_size: f32,
    probe_count: u32
) -> u32 {
    // Initialize shared memory on first thread
    if (local_id.x == 0u && local_id.y == 0u) {
        atomicStore(&best_prev_probe_index, 0xFFFFFFFFu);
        atomicStore(&best_prev_probe_error, bitcast<u32>(1000000.0)); // Very large error
    }
    
    workgroupBarrier();
    
    // Read motion vector to find previous pixel location
    let motion = textureLoad(gbuffer_motion, vec2<i32>(current_pixel), 0).xy;
    let prev_pixel = vec2<i32>(current_pixel) - vec2<i32>(motion);
    
    // Each thread in the 8x8 workgroup checks a subset of probes
    // This distributes the work across all 64 threads
    let thread_index = local_id.y * 8u + local_id.x;
    let probes_per_thread = (probe_count + 63u) / 64u;
    
    for (var i = 0u; i < probes_per_thread; i = i + 1u) {
        let probe_idx = thread_index + i * 64u;
        if (probe_idx >= probe_count) {
            break;
        }
        
        let prev_probe = screen_probes[probe_idx];
        
        // Check if probe was active
        if (prev_probe.state.x == 0u) {
            continue;
        }
        
        let prev_probe_pixel = vec2<u32>(prev_probe.state.y, prev_probe.state.z);
        
        // Check if this probe's previous pixel is near our reprojected location
        let pixel_dist = length(vec2<f32>(prev_probe_pixel) - vec2<f32>(prev_pixel));
        if (pixel_dist > 8.0) { // Only consider probes within 8 pixels
            continue;
        }
        
        // Compute reuse error
        let error = compute_reuse_error(
            current_pos,
            current_normal,
            prev_probe.position_radius.xyz,
            prev_probe.normal_frame.xyz,
            cell_size
        );
        
        // Try to update best probe atomically
        var current_best_error = bitcast<f32>(atomicLoad(&best_prev_probe_error));
        loop {
            if (error >= current_best_error) {
                break;
            }
            
            let exchanged = atomicCompareExchangeWeak(
                &best_prev_probe_error,
                bitcast<u32>(current_best_error),
                bitcast<u32>(error)
            );
            
            if (exchanged.exchanged) {
                atomicStore(&best_prev_probe_index, probe_idx);
                break;
            }
            
            current_best_error = bitcast<f32>(exchanged.old_value);
        }
    }
    
    workgroupBarrier();
    
    // Check if we found a suitable probe (error threshold)
    let final_best_error = bitcast<f32>(atomicLoad(&best_prev_probe_error));
    let final_best_index = atomicLoad(&best_prev_probe_index);
    
    // Only reuse if error is below threshold
    let max_acceptable_error = 1.0;
    if (final_best_error < max_acceptable_error) {
        return final_best_index;
    }
    
    return 0xFFFFFFFFu;
}

@compute @workgroup_size(8, 8, 1)
fn cs(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let res = textureDimensions(gbuffer_position);
    
    if (gid.x >= res.x || gid.y >= res.y) {
        return;
    }
    
    let pixel_coords = gid.xy;
    let pixel_coord_i = vec2<i32>(i32(pixel_coords.x), i32(pixel_coords.y));
    
    // === Temporal Upscale with Halton Sequence ===
    let upscale = vec2<u32>(gi_params.upscale_x, gi_params.upscale_y);
    
    // Check if we should spawn a probe at this pixel using temporal upscale
    if (!should_spawn_probe_temporal(pixel_coords, gi_params.frame_index, upscale)) {
        return;
    }
    
    // Read G-buffer data
    let position = textureLoad(gbuffer_position, pixel_coord_i, 0).xyz;
    let normal_data = textureLoad(gbuffer_normal, pixel_coord_i, 0);
    let normal = safe_normalize(normal_data.xyz);
    let normal_length = length(normal_data.xyz);
    
    // Only spawn probe if there's valid geometry
    if (normal_length < 0.01) {
        return;
    }
    
    let albedo = textureLoad(gbuffer_albedo, pixel_coord_i, 0).rgb;
    let smra = textureLoad(gbuffer_smra, pixel_coord_i, 0);
    let roughness = smra.g;
    
    // === Probe Reprojection ===
    // Try to find and reuse a probe from the previous frame
    let cell_size = bitcast<f32>(gi_params.cell_size_heuristic);
    let prev_probe_count = atomicLoad(&screen_probe_counter[0]);
    
    let should_reset = gi_params.reset_caches != 0u;
    var reuse_probe_index = 0xFFFFFFFFu;
    
    if (!should_reset && prev_probe_count > 0u) {
        // Collaborative search across workgroup for best probe to reuse
        reuse_probe_index = find_best_previous_probe(
            pixel_coords,
            position,
            normal,
            lid,
            cell_size,
            prev_probe_count
        );
    }
    
    // === Allocate or Reuse Probe ===
    var probe_index: u32;
    var is_reused = false;
    
    if (reuse_probe_index != 0xFFFFFFFFu) {
        // Reuse existing probe
        probe_index = reuse_probe_index;
        is_reused = true;
    } else {
        // Allocate new probe slot
        probe_index = atomicAdd(&screen_probe_counter[0], 1u);
        
        // Bounds check
        if (probe_index >= gi_params.max_screen_probes) {
            atomicSub(&screen_probe_counter[0], 1u);
            return;
        }
    }
    
    // === Update Probe Data ===
    let probe_size_world = f32(gi_params.screen_probe_size) * 0.01; // Heuristic scale
    
    // Update geometry info
    screen_probes[probe_index].position_radius = vec4<f32>(position, probe_size_world);
    screen_probes[probe_index].normal_frame = vec4<f32>(normal, f32(gi_params.frame_index));
    screen_probes[probe_index].albedo_roughness = vec4<f32>(albedo, roughness);
    
    // Update state
    screen_probes[probe_index].state = vec4<u32>(
        1u,              // active
        pixel_coords.x,  // pixel_x
        pixel_coords.y,  // pixel_y
        probe_index      // probe_id
    );
    
    // If not reused or reset requested, clear radiance
    if (!is_reused || should_reset) {
        screen_probes[probe_index].radiance_m = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    // Otherwise, keep the existing radiance from the reused probe (temporal stability)
}

