// =============================================================================
// GI-1.0 Screen Probe Spawning with Temporal Reprojection
// 
// DESIGN (following GI-1.0 paper Algorithm 1):
// - All probes are allocated in a fixed screen-space grid
// - Grid spacing = screen_probe_size (e.g. 8x8 pixel tiles)
// - Each workgroup processes one probe tile (4x4 threads = 16 lanes)
// - Temporal upscaling: only fraction of tiles update each frame
//
// TWO-PART DISPATCH (per tile, per frame):
//
// PART 1 - All threads participate (reprojection competition):
//    a. Each thread samples ONE pixel in current tile
//    b. Use motion vector to find where that pixel came from (prev frame)
//    c. Determine which probe tile contained that previous pixel
//    d. Validate: plane_distance < cell_size && normal_similarity > 0.90
//    e. Calculate reprojection score (3D distance between probe and pixel)
//    f. Store pixel coords and prev probe index in shared memory [lane_index]
//    g. Atomic competition: pack (score << 16) | lane_index, atomicMin()
//
// PART 2 - Thread 0 reads back result and places probe:
//    a. Read winning lane_index from LDS atomic
//    b. If valid reprojection found:
//       - Use WINNING PIXEL as destination for probe placement
//       - Copy accumulated radiance from previous probe
//    c. If no valid reprojection:
//       - Use Halton-jittered pixel to spawn new probe
//       - Initialize radiance to zero
// =============================================================================
#include "common.wgsl"
#include "gi/gi_common.wgsl"

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read_write> gi_counters: GICounters;
@group(1) @binding(2) var<storage, read_write> screen_probes: array<ScreenProbe>;
@group(1) @binding(3) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(4) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(5) var gbuffer_albedo: texture_2d<f32>;
@group(1) @binding(6) var gbuffer_smra: texture_2d<f32>;
@group(1) @binding(7) var gbuffer_motion: texture_2d<f32>;

// Workgroup-shared memory for reprojection (AMD GI-1.0 algorithm)
// Format: (reprojection_score << 16) | (lane_index & 0xFFFF)
// Lower 16 bits store which thread/lane won the competition
// Upper 16 bits store reprojection score (distance) for atomic competition
var<workgroup> reprojection_best: atomic<u32>;

// Each thread stores its pixel coordinates and previous probe index
// Indexed by lane_index (lid.y * 4 + lid.x for 4x4 workgroup)
var<workgroup> thread_pixel_coords: array<vec2<u32>, 16>;
var<workgroup> thread_prev_probe_index: array<u32, 16>;

@compute @workgroup_size(4, 4, 1)
fn cs(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let res = textureDimensions(gbuffer_position);
    let probe_size = u32(gi_params.screen_probe_size);
    let grid_dims = grid_dimensions(res, probe_size);
    
    // Calculate which probe this workgroup is processing
    let probe_tile = wid.xy;
    let probe_index = probe_tile.y * grid_dims.x + probe_tile.x;
    
    // Bounds check
    if (probe_index >= u32(gi_params.total_screen_probes)) {
        return;
    }
    
    // Check if this probe should be updated this frame (temporal upscaling)
    let upscale = vec2<u32>(u32(gi_params.upscale_x), u32(gi_params.upscale_y));
    let should_update = should_update_probe_this_frame(probe_tile, u32(gi_params.frame_index), upscale);
    
    // Early exit if not updating this frame (unless reset flag is set)
    if (!should_update && u32(gi_params.reset_caches) == 0u) {
        screen_probes[probe_index].state.w = 0.0;
        return;
    }
    
    // If reset flag is set but not updating, just clear and return
    if (!should_update && u32(gi_params.reset_caches) != 0u) {
        screen_probes[probe_index].radiance_m = vec4<f32>(0.0);
        screen_probes[probe_index].state.x = 0.0;
        screen_probes[probe_index].state.w = 0.0;
        return;
    }

    // Initialize shared memory: store invalid lane index (0xFFFF) with max distance
    if (lid.x == 0u && lid.y == 0u) {
        atomicStore(&reprojection_best, (pack_half_float(65504.0) << 16u) | 0xFFFFu);
    }
    workgroupBarrier();
    
    // =========================================================================
    // STEP 1: REPROJECTION - Find previous probe via motion vectors (AMD Algorithm)
    // =========================================================================
    // Each thread samples one pixel in the current tile
    let tile_corner = probe_tile * probe_size;
    let pixel = tile_corner + lid.xy;
    let lane_index = lid.y * 4u + lid.x;  // Thread's lane index in workgroup
    
    // Check if pixel is within bounds
    if (pixel.x < res.x && pixel.y < res.y) {
        let pixel_i32 = vec2<i32>(pixel);
        let normal_data = textureLoad(gbuffer_normal, pixel_i32, 0);
        let normal_length = length(normal_data.xyz);
        
        // Only valid geometry pixels participate in reprojection
        if (normal_length > 0.0) {
            let position_current = textureLoad(gbuffer_position, pixel_i32, 0).xyz;
            let normal_current = safe_normalize(normal_data.xyz);
            
            // Use motion vector to find where this pixel came from in previous frame
            let motion_ndc = textureLoad(gbuffer_motion, pixel_i32, 0).xy;
            
            // Convert current pixel to NDC, apply motion, then back to pixel space
            // Pixel space: (0,0) = top-left, Y increases downward
            // NDC space: (-1,-1) = bottom-left, Y increases upward
            let current_ndc_x = (f32(pixel.x) + 0.5) / f32(res.x) * 2.0 - 1.0;
            let current_ndc_y = 1.0 - (f32(pixel.y) + 0.5) / f32(res.y) * 2.0;  // Y-flip
            let prev_ndc = vec2<f32>(current_ndc_x, current_ndc_y) - motion_ndc;
            let pixel_prev_f = vec2<f32>(
                (prev_ndc.x + 1.0) * 0.5 * f32(res.x),
                (1.0 - prev_ndc.y) * 0.5 * f32(res.y)  // Y-flip back
            );
            let pixel_prev = vec2<i32>(pixel_prev_f);
            
            // Check if previous pixel is within bounds
            if (pixel_prev.x >= 0 && pixel_prev.x < i32(res.x) && 
                pixel_prev.y >= 0 && pixel_prev.y < i32(res.y)) {
                
                // Find which probe tile contained pixel_prev
                let probe_tile_prev = vec2<u32>(pixel_prev) / probe_size;
                let probe_index_prev = probe_tile_prev.y * grid_dims.x + probe_tile_prev.x;
                
                // Validate probe index
                if (probe_index_prev < u32(gi_params.total_screen_probes)) {
                    let probe_prev = screen_probes[probe_index_prev];
                    
                    // Check if probe was active in previous frame
                    if (probe_prev.state.x > 0.0) {
                        let world_probe = probe_prev.position_radius.xyz;
                        let normal_probe = probe_prev.normal_frame.xyz;
                        
                        // Second validation: World-space similarity
                        // For static geometry, positions should match
                        // For dynamic geometry, this will fail and spawn new probe (correct behavior)
                        let plane_dist = abs(dot(world_probe - position_current, normal_current));
                        let normal_similarity = dot(normal_probe, normal_current);
                        
                        // If validation passes, compete for selection (AMD Algorithm)
                        if (plane_dist < gi_params.cell_size_heuristic && normal_similarity > 0.95) {
                            // Store this thread's pixel coordinates and previous probe index
                            thread_pixel_coords[lane_index] = pixel;
                            thread_prev_probe_index[lane_index] = probe_index_prev;
                            
                            // Score by 3D distance (reprojection error) between probe and current pixel
                            let dist_3d = distance(world_probe, position_current);
                            
                            // Pack reprojection score (top 16 bits) + lane index (bottom 16 bits)
                            let packed_score = (pack_half_float(dist_3d) << 16u) | (lane_index & 0xFFFFu);
                            atomicMin(&reprojection_best, packed_score);
                        }
                    }
                }
            }
        }
    }
    
    workgroupBarrier();
    
    // =========================================================================
    // STEP 2: Read back reprojection result and place probe (AMD Algorithm)
    // =========================================================================
    let final_best = atomicLoad(&reprojection_best);
    let winning_lane_index = final_best & 0xFFFFu;
    let found_valid_reprojection = winning_lane_index != 0xFFFFu;
    
    // Only thread 0 performs the update
    if (lid.x == 0u && lid.y == 0u) {
        var spawn_pixel: vec2<u32>;
        var best_probe_prev_index: u32 = 0u;
        
        if (found_valid_reprojection) {
            // AMD Algorithm: Use the winning pixel as destination for reprojected probe
            spawn_pixel = thread_pixel_coords[winning_lane_index];
            best_probe_prev_index = thread_prev_probe_index[winning_lane_index];
        } else {
            // No valid reprojection found - spawn new probe with Halton jittering
            let total_frames = upscale.x * upscale.y;
            let frame_in_cycle = u32(gi_params.frame_index) % total_frames;
            let halton_sample = halton_2d(frame_in_cycle);
            let jitter = halton_sample * f32(probe_size);
            let tile_corner_f = vec2<f32>(probe_tile * probe_size);
            spawn_pixel = vec2<u32>(tile_corner_f + jitter);
        }

        // Sample G-buffer at the chosen pixel (winner's pixel or Halton-jittered)
        let pixel_i32 = vec2<i32>(spawn_pixel);
        let position = textureLoad(gbuffer_position, pixel_i32, 0).xyz;
        let normal_data = textureLoad(gbuffer_normal, pixel_i32, 0);
        let normal = safe_normalize(normal_data.xyz);
        let normal_length = length(normal_data.xyz);
        let albedo = textureLoad(gbuffer_albedo, pixel_i32, 0).rgb;
        let smra = textureLoad(gbuffer_smra, pixel_i32, 0);

        if (normal_length > 0.0) {
            // Place probe at chosen pixel location
            // If reprojection succeeded: at winning pixel, copy radiance from previous probe
            // If reprojection failed: at Halton-jittered pixel, initialize new probe
            screen_probes[probe_index].position_radius = vec4<f32>(
                position,
                gi_params.cell_size_heuristic
            );
            screen_probes[probe_index].normal_frame = vec4<f32>(
                normal,
                gi_params.frame_index
            );
            screen_probes[probe_index].albedo_roughness = vec4<f32>(
                albedo,
                smra.g
            );
            screen_probes[probe_index].radiance_m = select(
                vec4<f32>(0.0),
                max(screen_probes[best_probe_prev_index].radiance_m, vec4<f32>(0.0)),
                found_valid_reprojection && u32(gi_params.reset_caches) == 0u
            );
            screen_probes[probe_index].state = vec4<f32>(
                1.0,                 // active
                f32(spawn_pixel.x),  // pixel x
                f32(spawn_pixel.y),  // pixel y
                1.0                  // updating
            );
            
            atomicAdd(&gi_counters.active_probe_count, 1u);
        } else {
            // No valid geometry - mark inactive and not updating
            screen_probes[probe_index].state.x = 0.0;
            screen_probes[probe_index].state.w = 0.0;
        }
    }
}

