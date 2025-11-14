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
//    b. Use world-space velocity to reconstruct previous pixel position
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
//       - Query world cache for initial radiance seed
// =============================================================================
#include "common.wgsl"
#include "gi/gi_common.wgsl"
#include "gi/world_cache_common.wgsl"

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read_write> gi_counters: GICounters;
@group(1) @binding(2) var<storage, read_write> screen_probes: array<ScreenProbe>;
@group(1) @binding(3) var<storage, read_write> world_cache: array<WorldCacheCell>;
@group(1) @binding(4) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(5) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(6) var gbuffer_albedo: texture_2d<f32>;
@group(1) @binding(7) var gbuffer_smra: texture_2d<f32>;
@group(1) @binding(8) var gbuffer_motion: texture_2d<f32>;

const SCREEN_PROBE_SIZE = 4u;

// Workgroup-shared memory for reprojection (AMD GI-1.0 algorithm)
// Format: (reprojection_score << 16) | (lane_index & 0xFFFF)
// Lower 16 bits store which thread/lane won the competition
// Upper 16 bits store reprojection score (distance) for atomic competition
var<workgroup> reprojection_best: atomic<u32>;
// Each thread stores its pixel coordinates and previous probe index
var<workgroup> thread_pixel_coords: array<vec2<u32>, 16u>;
var<workgroup> thread_prev_probe_index: array<u32, 16u>;

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
    let prev_view_projection = view.prev_projection_matrix * view.prev_view_matrix;
    let camera_position = view.view_position.xyz;
    let view_dimensions = vec2<f32>(f32(res.x), f32(res.y));

    // distance_scale = tan(fov_y * proj_size * max(1/view_height, view_height/view_width^2))
    // adaptive_cell_size = distance_scale * distance_to_camera
    let reciprocal_height = 1.0 / max(view_dimensions.y, 1.0);
    let width_squared = max(view_dimensions.x * view_dimensions.x, 1.0);
    let anisotropic_term = view_dimensions.y / width_squared;
    let pixel_footprint = max(reciprocal_height, anisotropic_term);
    let base_distance_scale = tan(view.fov * gi_params.screen_probe_size * pixel_footprint);
    let distance_scale = max(base_distance_scale, 1e-5);
    
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
    // STEP 1: REPROJECTION - World-velocity guided probe lookup (AMD Algorithm)
    // =========================================================================
    // Each thread samples one pixel in the current tile
    let tile_corner = probe_tile * probe_size;
    let pixel = tile_corner + lid.xy;
    let lane_index = lid.y * SCREEN_PROBE_SIZE + lid.x;  // Thread's lane index in workgroup
    
    // Check if pixel is within bounds
    if (pixel.x < res.x && pixel.y < res.y) {
        let pixel_i32 = vec2<i32>(pixel);
        let normal_data = textureLoad(gbuffer_normal, pixel_i32, 0);
        let normal_length = length(normal_data.xyz);
        
        // Only valid geometry pixels participate in reprojection
        if (normal_length > 0.0) {
            let position_current = textureLoad(gbuffer_position, pixel_i32, 0).xyz;
            let normal_current = safe_normalize(normal_data.xyz);
            let distance_to_camera = length(camera_position - position_current);
            let adaptive_cell_size = max(distance_scale * distance_to_camera, 0.001);
            
            // Use world-space velocity stored in G-buffer to recover the previous position
            let motion_sample = textureLoad(gbuffer_motion, pixel_i32, 0);
            let world_velocity = motion_sample.xyz;
            let position_prev = position_current - world_velocity;
            let prev_clip = prev_view_projection * vec4<f32>(position_prev, 1.0);
            let prev_ndc = prev_clip.xy / prev_clip.w;

            // Convert previous NDC position into pixel coordinates (Y flip for screen space)
            let pixel_prev_f = vec2<f32>(
                (prev_ndc.x + 1.0) * 0.5 * f32(res.x),
                (1.0 - prev_ndc.y) * 0.5 * f32(res.y)
            );
            let pixel_prev = vec2<i32>(pixel_prev_f);

            // Check if previous pixel is within bounds
            if (pixel_prev.x >= 0 && pixel_prev.x < i32(res.x) && 
                pixel_prev.y >= 0 && pixel_prev.y < i32(res.y)) {
                
                // Find which probe tile contained pixel_prev
                let probe_tile_prev = vec2<u32>(pixel_prev) / probe_size;
                let probe_index_prev = probe_tile_prev.y * grid_dims.x + probe_tile_prev.x;
                
                // Validate probe index and active in last frame
                if (screen_probes[probe_index_prev].state.x > 0.0) {
                    let world_probe = screen_probes[probe_index_prev].position_radius.xyz;
                    let normal_probe = screen_probes[probe_index_prev].normal_frame.xyz;
                    
                    let plane_dist = abs(dot(world_probe - position_current, normal_current));
                    let normal_similarity = dot(normal_probe, normal_current);
                    
                    if (plane_dist < adaptive_cell_size && normal_similarity > 0.95) {
                        thread_pixel_coords[lane_index] = pixel;
                        thread_prev_probe_index[lane_index] = probe_index_prev;
                        
                        let dist_3d = distance(world_probe, position_current);
                        
                        let packed_score = (pack_half_float(dist_3d) << 16u) | (lane_index & 0xFFFFu);
                        atomicMin(&reprojection_best, packed_score);
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
        let distance_to_camera_spawn = length(camera_position - position);
        let adaptive_cell_size_spawn = max(distance_scale * distance_to_camera_spawn, 0.001);
        let albedo = textureLoad(gbuffer_albedo, pixel_i32, 0).rgb;
        let smra = textureLoad(gbuffer_smra, pixel_i32, 0);
        let motion_emissive = textureLoad(gbuffer_motion, pixel_i32, 0);

        if (normal_length > 0.0) {
            // Extract material properties from G-buffer
            let roughness = smra.g;
            let metallic = smra.b;
            let reflectance = smra.r * 0.0009765625; // Decode: 1.0 / 1024
            let emissive = motion_emissive.w;
            
            // =====================================================================
            // Initialize probe radiance based on spawn type
            // =====================================================================
            var initial_radiance = vec4<f32>(0.0);
            
            if (found_valid_reprojection && u32(gi_params.reset_caches) == 0u) {
                // Reprojection succeeded: Copy accumulated radiance from previous probe
                initial_radiance = max(screen_probes[best_probe_prev_index].radiance_m, vec4<f32>(0.0));
            }
            
            // Place probe at chosen pixel location
            // If reprojection succeeded: at winning pixel, copy radiance from previous probe
            // If reprojection failed: at Halton-jittered pixel, seed from world cache
            screen_probes[probe_index].position_radius = vec4<f32>(
                position,
                adaptive_cell_size_spawn
            );
            screen_probes[probe_index].normal_frame = vec4<f32>(
                normal,
                gi_params.frame_index
            );
            screen_probes[probe_index].albedo_roughness = vec4<f32>(
                albedo,
                roughness
            );
            screen_probes[probe_index].material_props = vec4<f32>(
                metallic,
                reflectance,
                emissive,
                0.0  // unused
            );
            screen_probes[probe_index].radiance_m = initial_radiance;
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

