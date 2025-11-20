// =============================================================================
// GI-1.0 Screen Probe Update
// - Accumulates radiance from traced rays back to screen probes
// - Writes FRESH probe data each frame (no temporal blending at probe level)
// - Temporal stability is handled at per-pixel reconstruction stage
// - Only updates probes marked active this frame
// - Writes to probe radiance atlas texture (octahedral layout)
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

const SCREEN_PROBE_SAMPLE_CAP = 32.0;

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
    // Write fresh probe data to output atlas
    // No temporal blending at probe level - that's handled at reconstruction
    // =========================================================================
    for (var py = 0u; py < probe_size; py = py + 1u) {
        for (var px = 0u; px < probe_size; px = px + 1u) {
            let atlas_coord = probe_tile * probe_size + vec2<u32>(px, py);
            
            // Read current radiance (already populated by reprojection pass)
            let curr_radiance_data = textureLoad(probe_radiance_curr, vec2<i32>(atlas_coord), 0);
            let curr_radiance = max(curr_radiance_data.rgb, vec3<f32>(0.0));
            let curr_sample_count = max(curr_radiance_data.w, 0.0);
            
            // We store the SUM of samples, reconstruction divides by count to get average
            // With temporal upscaling, probes update infrequently (e.g., once per 16 frames)
            // A lower cap ensures faster convergence and prevents old dark samples from dominating
            var blended_radiance: vec3<f32>;
            var new_sample_count: f32;
            
            if (curr_sample_count + f32(rays_per_probe) <= SCREEN_PROBE_SAMPLE_CAP) {
                // Below cap: simple accumulation (sum of all samples so far)
                blended_radiance = curr_radiance + accumulated_radiance;
                new_sample_count = curr_sample_count + f32(rays_per_probe);
            } else {
                // At/above cap: rolling window - remove N oldest samples
                // Keep (cap - rays) worth of old samples, add new samples
                let keep_ratio = (SCREEN_PROBE_SAMPLE_CAP - f32(rays_per_probe)) / SCREEN_PROBE_SAMPLE_CAP;
                blended_radiance = curr_radiance * keep_ratio + accumulated_radiance * (1.0 + keep_ratio);
                new_sample_count = SCREEN_PROBE_SAMPLE_CAP - f32(rays_per_probe);
            }
            
            // Write to ping-pong output texture
            textureStore(probe_radiance_output, vec2<i32>(atlas_coord), vec4<f32>(blended_radiance, new_sample_count));
        }
    }
}
