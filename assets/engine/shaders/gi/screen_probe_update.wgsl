// =============================================================================
// GI-1.0 Screen Probe Update with Progressive Sample Accumulation
//
// Accumulates ray samples over time for proper Monte Carlo convergence.
// Critical for low spp/frame (e.g. 1 ray per probe per frame).
//
// Storage Format:
// - RGB: accumulated radiance SUM (sum of all samples)
// - W: sample count (number of rays accumulated)
//
// Algorithm:
// 1. Accumulate radiance from traced rays (sum of this frame's rays)
// 2. Read historical sum + count from texture
// 3. If below cap (16 samples): add to sum, increment count
// 4. If at cap: blend averages to slowly replace old samples with new
//
// Why This Works:
// - Accumulation phase: progressively reduces noise as samples build up
// - Blend phase: prevents stale samples, allows adaptation to changes
// - Proper sum/count representation: mathematically correct averaging
//
// With 1 ray/probe/frame: reaches ~8 samples in 8 frames, then stabilizes
// =============================================================================
#include "common.wgsl"
#include "gi/gi_common.wgsl"
#include "gi/world_cache_common.wgsl"

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read_write> gi_counters: GICounters;
@group(1) @binding(2) var<storage, read_write> screen_probe_metadata: array<ScreenProbe>;
@group(1) @binding(3) var probe_radiance_curr: texture_2d<f32>; // Read from current (reprojection already populated it)
@group(1) @binding(4) var<storage, read> probe_path_state: array<ProbePathState>;
@group(1) @binding(5) var<storage, read_write> world_cache: array<WorldCacheCell>;
@group(1) @binding(6) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(7) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(8) var probe_radiance_output: texture_storage_2d<rgba16float, write>; // Write to ping-pong output

// Progressive accumulation cap: how many samples to accumulate before blending
const MAX_ACCUMULATED_SAMPLES = 8.0;  // Balance between convergence speed and adaptability

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Process all grid probes (derived from resolution)
    let probe_count = u32(gi_params.total_screen_probes);
    
    if (gid.x >= probe_count) {
        return;
    }
    
    // Read probe metadata
    let probe = screen_probe_metadata[gid.x];
    if (probe.state.x == 0.0) {
        return; // Skip inactive probes
    }
    
    let probe_age = max(probe.state.w, 0.0);
    let rays_per_probe = u32(gi_params.screen_ray_count);
    
    // Get camera position for adaptive world cache updates
    let view_index = u32(frame_info.view_index);
    let view = view_buffer[view_index];
    let camera_position = view.view_position.xyz;
    
    // Calculate probe tile coordinates for atlas access
    let res = textureDimensions(gbuffer_position);
    let probe_size = u32(gi_params.screen_probe_size);
    let grid_dims = grid_dimensions(res, probe_size);
    
    let probe_tile = vec2<u32>(
        gid.x % grid_dims.x,
        gid.x / grid_dims.x
    );
    
    // =========================================================================
    // Accumulate radiance from all rays for this probe
    // =========================================================================
    var accumulated_radiance = vec3<f32>(0.0);
    
    for (var i = 0u; i < rays_per_probe; i = i + 1u) {
        let ray_id = gid.x * rays_per_probe + i;
        let path = probe_path_state[ray_id];

        let sample_count = max(path.rng_sample_count_frame_stamp.y, 1.0);
        let accumulated_avg = path.throughput.xyz / sample_count;
        let radiance = safe_clamp_vec3(accumulated_avg);
        // Accumulate ALL rays (even if zero) to increment sample count properly
        accumulated_radiance += radiance;
    }
    
    // =========================================================================
    // Simple Progressive Sample Accumulation
    // Accumulate samples over time until we hit a cap, then start blending
    // =========================================================================    
    // Write accumulated probe data to output atlas
    for (var py = 0u; py < probe_size; py = py + 1u) {
        for (var px = 0u; px < probe_size; px = px + 1u) {
            let atlas_coord = probe_tile * probe_size + vec2<u32>(px, py);
            
            // Read current radiance (already populated by reprojection pass)
            let curr_radiance_data = textureLoad(probe_radiance_curr, vec2<i32>(atlas_coord), 0);
            
            // RGB = accumulated radiance SUM, W = number of samples accumulated
            let curr_sum = curr_radiance_data.rgb;
            let curr_count = curr_radiance_data.w;

            // At cap: blend (exponential moving average for adaptability)
            let base_alpha = f32(rays_per_probe) / MAX_ACCUMULATED_SAMPLES;

            let curr_avg  = curr_sum / max(curr_count, 1.0);
            let frame_mean = accumulated_radiance / max(f32(rays_per_probe), 1.0);

            // luma change
            let frame_luma = dot(frame_mean, vec3<f32>(0.2126, 0.7152, 0.0722));
            let curr_luma  = dot(curr_avg,  vec3<f32>(0.2126, 0.7152, 0.0722));
            let diff       = abs(frame_luma - curr_luma);

            // Heuristic: normalize by some scene-scale constant
            let scale      = 0.1; // tweak: what you consider "big" change in luma
            let t          = clamp(diff / scale, 0.0, 1.0);

            let alpha_min  = 0.01;
            let alpha_max  = 0.2;
            let adaptive_alpha = alpha_min + (alpha_max - alpha_min) * t;
            let alpha = min(adaptive_alpha, base_alpha);

            let blended_avg = curr_avg * (1.0 - alpha) + frame_mean * alpha;

            let new_count = min(curr_count + f32(rays_per_probe), MAX_ACCUMULATED_SAMPLES);
            let new_sum = blended_avg * new_count;
 
            // Write accumulated sum + count
            textureStore(probe_radiance_output, vec2<i32>(atlas_coord), vec4<f32>(new_sum, new_count));
        }
    }
}
