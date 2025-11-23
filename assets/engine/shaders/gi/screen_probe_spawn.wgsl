// =============================================================================
// GI-1.0 Screen Probe Spawning (Post-Reprojection)
// 
// This pass spawns NEW probes for tiles that were classified as "empty"
// by the reprojection pass. These are tiles where temporal reprojection failed
// (disocclusions, new geometry, etc.)
//
// Algorithm:
// 1. Read tile indices from empty_tiles queue
// 2. For each empty tile:
//    - Use Halton jitter to select spawn pixel within tile
//    - Sample G-buffer at spawn pixel
//    - Store pixel coordinates in probe metadata
// 3. Radiance will be populated by subsequent trace and update passes
//
// Note: We only store pixel coordinates; all surface properties are sampled
// from G-buffer on-demand in subsequent passes
// =============================================================================
#include "common.wgsl"
#include "gi/gi_common.wgsl"
#include "gi/world_cache_common.wgsl"

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read_write> gi_counters: GICounters;
@group(1) @binding(2) var<storage, read_write> world_cache: array<WorldCacheCell>;
@group(1) @binding(3) var<storage, read_write> screen_probe_metadata: array<ScreenProbe>;
@group(1) @binding(4) var<storage, read> empty_tiles: array<u32>;
@group(1) @binding(5) var<storage, read_write> tile_counters: TileCounters;
@group(1) @binding(6) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(7) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(8) var gbuffer_albedo: texture_2d<f32>;
@group(1) @binding(9) var gbuffer_smra: texture_2d<f32>;
@group(1) @binding(10) var gbuffer_motion: texture_2d<f32>;

@compute @workgroup_size(128, 1, 1)
fn cs(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let empty_count = atomicLoad(&tile_counters.empty_count);
    
    // Bounds check: only process valid empty tiles
    if (gid.x >= empty_count) {
        return;
    }
    
    // Read tile index from empty queue
    let probe_index = empty_tiles[gid.x];
    
    // Calculate probe tile coordinates from index
    let res = textureDimensions(gbuffer_position);
    let probe_size = u32(gi_params.screen_probe_size);
    let grid_dims = grid_dimensions(res, probe_size);
    
    let probe_tile = vec2<u32>(
        probe_index % grid_dims.x,
        probe_index / grid_dims.x
    );
    
    // Bounds check
    if (probe_tile.x >= grid_dims.x || probe_tile.y >= grid_dims.y) {
        return;
    }

    // Check if this probe should be spawned this frame (temporal upscaling)
    let upscale = vec2<u32>(u32(gi_params.upscale_x), u32(gi_params.upscale_y));
    let should_spawn = should_update_probe_this_frame(probe_tile, u32(gi_params.frame_index), upscale);
    
    if (!should_spawn) {
        // Tile is skipped this frame—mark as inactive for this update cycle
        // age = -1.0 signals "not tracing this frame" (temporal upscale skip)
        screen_probe_metadata[probe_index].state.w = -1.0;
        return;
    }
    
    let view_index = u32(frame_info.view_index);
    let view = view_buffer[view_index];
    let camera_position = view.view_position.xyz;
    
    // =========================================================================
    // Spawn New Probe with Halton Jitter (Temporal Upscaling)
    // =========================================================================
    // Halton sequence provides good spatial distribution over time
    let total_frames = upscale.x * upscale.y;
    let frame_in_cycle = u32(gi_params.frame_index) % total_frames;
    let halton_sample = halton_2d(frame_in_cycle);
    let jitter = halton_sample * f32(probe_size);
    let tile_corner = vec2<f32>(probe_tile * probe_size);
    let spawn_pixel = vec2<u32>(tile_corner + jitter);
    
    // Sample G-buffer at spawn pixel
    let pixel_i32 = vec2<i32>(spawn_pixel);
    
    // Check if pixel is within bounds
    if (spawn_pixel.x >= res.x || spawn_pixel.y >= res.y) {
        // Out of bounds - mark probe as invalid
        screen_probe_metadata[probe_index].state = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return;
    }
    
    let position = textureLoad(gbuffer_position, pixel_i32, 0).xyz;
    let normal_data = textureLoad(gbuffer_normal, pixel_i32, 0);
    let normal = safe_normalize(normal_data.xyz);
    let normal_length = length(normal_data.xyz);
    
    if (normal_length > 0.0) {
        // Valid geometry - spawn probe by storing pixel coordinates and surface properties
        screen_probe_metadata[probe_index].state = vec4<f32>(
            1.0,                    // active
            f32(spawn_pixel.x),     // pixel_x
            f32(spawn_pixel.y),     // pixel_y
            0.0                     // age = 0 (newly spawned, no history)
        );
        
        // Increment active probe count
        atomicAdd(&gi_counters.active_probe_count, 1u);
    } else {
        // No valid geometry - mark probe as invalid
        screen_probe_metadata[probe_index].state = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
}
