// =============================================================================
// GI-1.0 Screen Probe Spawning with Temporal Reprojection
// 
// DESIGN (following GI-1.0 paper Algorithm 1):
// - All probes are allocated in a fixed screen-space grid
// - Grid spacing = screen_probe_size (e.g. 8x8 pixel tiles)
// - Each workgroup processes one probe tile
// - Temporal upscaling: only fraction of tiles update each frame
//
// ALGORITHM (per tile, per frame):
// 1. Check if tile should update this frame (temporal upscaling pattern)
// 2. REPROJECTION: Use motion vectors to find previous probe:
//    a. Each thread samples one pixel in current tile
//    b. Use motion vector to find where that pixel came from (prev frame)
//    c. Determine which probe tile contained that previous pixel
//    d. Validate: plane_distance < cell_size && normal_dot > 0.95
//    e. Threads compete atomically for best match (closest 3D distance)
// 3. If reprojection SUCCEEDS: reuse that probe's data (keep radiance!)
// 4. If reprojection FAILS: SPAWN new probe:
//    a. Use Halton jitter to pick a sample pixel in tile
//    b. Sample G-buffer at that pixel for world position/normal
//    c. Initialize probe with new data (reset radiance)
// =============================================================================
#include "common.wgsl"
#include "lighting_common.wgsl"
#include "gi/gi_common.wgsl"

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read_write> gi_counters: GICounters;
@group(1) @binding(2) var<storage, read_write> screen_probes: array<ScreenProbe>;
@group(1) @binding(3) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(4) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(5) var gbuffer_albedo: texture_2d<f32>;
@group(1) @binding(6) var gbuffer_smra: texture_2d<f32>;
@group(1) @binding(7) var gbuffer_motion: texture_2d<f32>;

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

// Workgroup-shared memory for reprojection
// Format: (packed_distance << 16) | (probe_index_prev & 0xFFFF)
// Lower 16 bits store which previous probe to reuse
// Upper 16 bits store distance for atomic competition
var<workgroup> reprojection_best: atomic<u32>;

@compute @workgroup_size(8, 8, 1)
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
        return;
    }
    
    // If reset flag is set but not updating, just clear and return
    if (!should_update && u32(gi_params.reset_caches) != 0u) {
        if (lid.x == 0u && lid.y == 0u) {
            screen_probes[probe_index].radiance_m = vec4<f32>(0.0);
            screen_probes[probe_index].state.x = 0.0;
        }
        return;
    }

    // Initialize shared memory: store invalid probe index (0xFFFF) with max distance
    if (lid.x == 0u && lid.y == 0u) {
        atomicStore(&reprojection_best, (pack_half_float(65504.0) << 16u) | 0xFFFFu);
    }
    workgroupBarrier();
    
    // =========================================================================
    // STEP 1: REPROJECTION - Find previous probe via motion vectors
    // =========================================================================
    
    // Each thread samples one pixel in the current tile
    let tile_corner = probe_tile * probe_size;
    let pixel = tile_corner + lid.xy;
    
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
            
            // Convert NDC motion to pixel motion
            // Motion vector = (current_ndc - prev_ndc) in range [-2, 2]
            // To get pixel motion: multiply by half resolution
            let pixel_motion = motion_ndc * vec2<f32>(res) * vec2<f32>(0.5);
            let pixel_prev_f = vec2<f32>(pixel) - pixel_motion;
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
                        
                        // Validation criteria from GI-1.0 paper:
                        // - Plane distance: does probe position lie close to current surface?
                        // - Normal alignment: do normals match?
                        let plane_dist = abs(dot(world_probe - position_current, normal_current));
                        let normal_similarity = dot(normal_probe, normal_current);
                        
                        // If validation passes, compete for selection
                        if (plane_dist < gi_params.cell_size_heuristic && normal_similarity > 0.95) {
                            // Score by 3D distance between probe and current pixel
                            let dist_3d = distance(world_probe, position_current);
                            let packed_score = (pack_half_float(dist_3d) << 16u) | (probe_index_prev & 0xFFFFu);
                            atomicMin(&reprojection_best, packed_score);
                        }
                    }
                }
            }
        }
    }
    
    workgroupBarrier();
    
    // =========================================================================
    // STEP 2: Decision - REPROJECT or SPAWN?
    // =========================================================================
    let final_best = atomicLoad(&reprojection_best);
    let best_probe_prev_index = final_best & 0xFFFFu;
    let found_valid_reprojection = best_probe_prev_index != 0xFFFFu;
    
    // Only thread 0 performs the update
    if (lid.x == 0u && lid.y == 0u) {
        // Use Halton sequence to pick a sample pixel in the tile
        let total_frames = upscale.x * upscale.y;
        let frame_in_cycle = u32(gi_params.frame_index) % total_frames;
        let halton_sample = halton_2d(frame_in_cycle);
        let jitter = halton_sample * f32(probe_size);
        let tile_corner_f = vec2<f32>(probe_tile * probe_size);
        let spawn_pixel = vec2<u32>(tile_corner_f + jitter);

        // Sample G-buffer at the jittered pixel
        let pixel_i32 = vec2<i32>(spawn_pixel);
        let position = textureLoad(gbuffer_position, pixel_i32, 0).xyz;
        let normal_data = textureLoad(gbuffer_normal, pixel_i32, 0);
        let normal = safe_normalize(normal_data.xyz);
        let normal_length = length(normal_data.xyz);
        let albedo = textureLoad(gbuffer_albedo, pixel_i32, 0).rgb;
        let smra = textureLoad(gbuffer_smra, pixel_i32, 0);

        if (normal_length > 0.0) {
            // Copy all probe data (including accumulated radiance!)
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
            let new_spawn = screen_probes[probe_index].state.x == 0.0;
            let existing_probe_radiance = select(
                vec4<f32>(0.0),
                screen_probes[probe_index].radiance_m,
                !new_spawn
            );
            screen_probes[probe_index].radiance_m = select(
                existing_probe_radiance,
                screen_probes[best_probe_prev_index].radiance_m,
                found_valid_reprojection && u32(gi_params.reset_caches) == 0u
            );
            screen_probes[probe_index].state = vec4<f32>(
                1.0,                 // active
                f32(spawn_pixel.x),  // pixel x
                f32(spawn_pixel.y),  // pixel y
                f32(new_spawn)       // new spawn flag
            );
            
            atomicAdd(&gi_counters.active_probe_count, 1u);
        } else {
            // No valid geometry - mark inactive and clear radiance
            screen_probes[probe_index].state.x = 0.0;
            screen_probes[probe_index].radiance_m = vec4<f32>(0.0);
        }
    }
}

