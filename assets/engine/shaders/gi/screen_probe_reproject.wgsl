// =============================================================================
// GI-1.0 Screen Probe Temporal Reprojection (Algorithm 1)
// 
// This pass attempts to temporally reproject screen probes from the previous
// frame using motion vectors and world-space consistency checks.
//
// Per-Tile Algorithm:
// 1. Each thread samples one pixel in the current tile
// 2. Use motion vectors to find corresponding pixel in previous frame
// 3. Check if there's a valid probe at that location (from metadata)
// 4. Validate: plane_distance < cell_size && normal_similarity > 0.95
// 5. Compete atomically for best match (closest 3D distance)
// 6. Winner: Reproject probe data (radiance atlas + pixel coords) to current tile
// 7. Classify tile: EMPTY (no match) or OVERRIDE (found match)
//
// Output Queues:
// - empty_tiles: Tiles that need new probes
// - override_tiles: Tiles with reprojected probes (can be reassigned in patch pass)
// =============================================================================
#include "common.wgsl"
#include "gi/gi_common.wgsl"

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var probe_radiance_prev: texture_2d<f32>;
@group(1) @binding(2) var<storage, read_write> screen_probe_metadata: array<ScreenProbe>;
@group(1) @binding(3) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(4) var gbuffer_position_prev: texture_2d<f32>;
@group(1) @binding(5) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(6) var gbuffer_normal_prev: texture_2d<f32>;
@group(1) @binding(7) var gbuffer_albedo: texture_2d<f32>;
@group(1) @binding(8) var gbuffer_smra: texture_2d<f32>;
@group(1) @binding(9) var gbuffer_motion: texture_2d<f32>;
@group(1) @binding(10) var<storage, read_write> empty_tiles: array<u32>;
@group(1) @binding(11) var<storage, read_write> override_tiles: array<u32>;
@group(1) @binding(12) var<storage, read_write> tile_counters: TileCounters;
@group(1) @binding(13) var probe_radiance_curr: texture_storage_2d<rgba16float, write>;

// Workgroup-shared memory for atomic reprojection competition
// Format: (reprojection_score << 16) | (lane_index & 0xFFFF)
var<workgroup> reprojection_best: atomic<u32>;
var<workgroup> thread_pixel_coords: array<vec2<u32>, 16u>;
var<workgroup> thread_prev_probe_index: array<u32, 16u>;

const SCREEN_PROBE_SIZE = 4u; // 4x4 = 16 threads per tile

@compute @workgroup_size(SCREEN_PROBE_SIZE, SCREEN_PROBE_SIZE, 1)
fn cs(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let res = textureDimensions(gbuffer_position);
    let probe_size = u32(gi_params.screen_probe_size);
    let grid_dims = grid_dimensions(res, probe_size);

    let view_index = u32(frame_info.view_index);
    let view = view_buffer[view_index];
    let camera_position = view.view_position.xyz;

    // Adaptive cell size calculation for reprojection tolerance
    let view_dimensions = vec2<f32>(f32(res.x), f32(res.y));
    let reciprocal_height = 1.0 / max(view_dimensions.y, 1.0);
    let anisotropic_term = view_dimensions.y / max(view_dimensions.x * view_dimensions.x, 1.0);
    let pixel_footprint = max(reciprocal_height, anisotropic_term);
    let distance_scale = tan(view.fov * gi_params.screen_probe_size * pixel_footprint);
    
    // Current probe tile being processed
    let probe_tile = wid.xy;
    let probe_index = probe_tile.y * grid_dims.x + probe_tile.x;
    
    // Bounds check
    if (probe_index >= u32(gi_params.total_screen_probes)) {
        return;
    }
    
    // Initialize reprojection best to a large value
    if (lid.x == 0u && lid.y == 0u) {
        atomicStore(&reprojection_best, (pack_half_float(65504.0) << 16u) | 0xFFFFu);
    }
    workgroupBarrier();
    
    // =========================================================================
    // STEP 1: Reprojection Competition (Algorithm 1)
    // =========================================================================
    // Each thread samples one pixel in the current tile
    let tile_corner = probe_tile * probe_size;
    let pixel = tile_corner + lid.xy;
    let lane_index = lid.y * SCREEN_PROBE_SIZE + lid.x;
    
    // Check if pixel is within bounds
    if (pixel.x < res.x && pixel.y < res.y) {
        let pixel_i32 = vec2<i32>(pixel);
        let normal_data = textureLoad(gbuffer_normal, pixel_i32, 0);
        let normal_current = safe_normalize(normal_data.xyz);
        
        // Only valid geometry pixels participate in reprojection
        if (length(normal_data.xyz) > 0.0) {
            // Use NDC-space velocity stored in G-buffer to recover previous pixel position
            let motion_sample = textureLoad(gbuffer_motion, pixel_i32, 0);
            let pixel_velocity = motion_sample.xy * vec2<f32>(f32(res.x), f32(res.y)) * vec2<f32>(0.5, -0.5);
            
            // Use sub-pixel precision for previous pixel calculation to avoid integer truncation artifacts
            let pixel_center = vec2<f32>(pixel) + 0.5;
            let pixel_prev_center = pixel_center + -pixel_velocity;
            let pixel_prev = vec2<i32>(floor(pixel_prev_center));

            // Check if previous pixel is within bounds
            if (pixel_prev.x >= 0 && pixel_prev.x < i32(res.x) && 
                pixel_prev.y >= 0 && pixel_prev.y < i32(res.y)) {
                
                // Find which probe tile contained pixel_prev
                let probe_tile_prev = vec2<u32>(pixel_prev) / probe_size;
                let probe_index_prev = probe_tile_prev.y * grid_dims.x + probe_tile_prev.x;
                
                // Check if probe tile is within grid bounds and was active
                if (probe_index_prev < u32(gi_params.total_screen_probes)) {
                    let position_current = textureLoad(gbuffer_position, pixel_i32, 0).xyz;
                    let distance_to_camera = distance(camera_position, position_current);
                    let adaptive_cell_size = max(distance_scale * distance_to_camera, 0.0001);
                    
                    // Only reproject from probes that were active AND updated last frame
                    if (screen_probe_metadata[probe_index_prev].state.x > 0.0) {
                        // World-space geometry validation using the probe's representative pixel
                        let prev_probe_pixel = vec2<u32>(
                            u32(screen_probe_metadata[probe_index_prev].state.y),
                            u32(screen_probe_metadata[probe_index_prev].state.z)
                        );
                        let prev_probe_position = textureLoad(gbuffer_position_prev, prev_probe_pixel, 0).xyz;
                        let prev_probe_normal_data = textureLoad(gbuffer_normal_prev, prev_probe_pixel, 0).xyz;
                        let prev_probe_normal = safe_normalize(prev_probe_normal_data.xyz);
                        
                        // Check plane distance and normal similarity
                        let plane_dist = abs(dot(prev_probe_position - position_current, normal_current));
                        let normal_similarity = dot(prev_probe_normal, normal_current);
                        let min_normal_similarity = 0.95;

                        if (plane_dist < adaptive_cell_size && normal_similarity > min_normal_similarity) {
                            // Valid reprojection candidate!
                            let world_dist_3d = distance(prev_probe_position, position_current);
                            thread_pixel_coords[lane_index] = pixel;
                            thread_prev_probe_index[lane_index] = probe_index_prev;
                            // Pack score and lane index for atomic competition
                            let packed_score = (pack_half_float(world_dist_3d) << 16u) | (lane_index & 0xFFFFu);
                            atomicMin(&reprojection_best, packed_score);
                        }
                    }
                }
            }
        }
    }
    
    workgroupBarrier();
    
    // =========================================================================
    // STEP 2: Read Back Reprojection Result and Classify Tile
    // =========================================================================
    if (lid.x == 0u && lid.y == 0u) {
        let final_best = atomicLoad(&reprojection_best);
        let winning_lane_index = final_best & 0xFFFFu;
        let found_valid_reprojection = winning_lane_index != 0xFFFFu;
        
        if (found_valid_reprojection) {
            // =====================================================================
            // REPROJECTION SUCCESS: Copy probe data from previous frame
            // =====================================================================
            let winner_pixel = thread_pixel_coords[winning_lane_index];
            let prev_probe_index = thread_prev_probe_index[winning_lane_index];
            let prev_probe = screen_probe_metadata[prev_probe_index];
            
            // Calculate previous probe tile coordinates
            let prev_probe_tile = vec2<u32>(
                prev_probe_index % grid_dims.x,
                prev_probe_index / grid_dims.x
            );

            // Copy radiance data from previous probe tile to current probe tile
            for (var py = 0u; py < probe_size; py = py + 1u) {
                for (var px = 0u; px < probe_size; px = px + 1u) {
                    let src_atlas_coord = prev_probe_tile * probe_size + vec2<u32>(px, py);
                    let dst_atlas_coord = probe_tile * probe_size + vec2<u32>(px, py);
                    let radiance_sample = textureLoad(probe_radiance_prev, vec2<i32>(src_atlas_coord), 0);
                    textureStore(probe_radiance_curr, vec2<i32>(dst_atlas_coord), radiance_sample);
                }
            }
            
            // Increment probe age (stability counter) for successful reprojection
            let prev_age = prev_probe.state.w;
            let new_age = min(prev_age + 1.0, 1024.0);
            
            // Update probe metadata: store winner pixel coords and age
            screen_probe_metadata[probe_index].state = vec4<f32>(
                1.0,                          // active
                f32(winner_pixel.x),          // pixel_x
                f32(winner_pixel.y),          // pixel_y
                new_age                       // age (frames since spawn)
            );
            
            // Add to override queue (can be reassigned in patch pass)
            let override_index = atomicAdd(&tile_counters.override_count, 1u);
            override_tiles[override_index] = probe_index;
        } else {
            // =====================================================================
            // REPROJECTION FAILED: Mark tile as empty
            // =====================================================================
            // Clear probe radiance atlas
            for (var py = 0u; py < probe_size; py = py + 1u) {
                for (var px = 0u; px < probe_size; px = px + 1u) {
                    let atlas_coord = probe_tile * probe_size + vec2<u32>(px, py);
                    textureStore(probe_radiance_curr, vec2<i32>(atlas_coord), vec4<f32>(0.0));
                }
            }

            screen_probe_metadata[probe_index].state.w = -1.0;
            
            // Mark probe as invalid (will be respawned) and add to empty queue
            let empty_index = atomicAdd(&tile_counters.empty_count, 1u);
            empty_tiles[empty_index] = probe_index;
        }
    }
}
