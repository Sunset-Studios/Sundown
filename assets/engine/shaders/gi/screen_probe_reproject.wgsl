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
@group(1) @binding(7) var gbuffer_motion: texture_2d<f32>;
@group(1) @binding(8) var<storage, read_write> empty_tiles: array<u32>;
@group(1) @binding(9) var<storage, read_write> override_tiles: array<u32>;
@group(1) @binding(10) var<storage, read_write> tile_counters: TileCounters;
@group(1) @binding(11) var probe_radiance_curr: texture_storage_2d<rgba16float, write>;

const WORKGROUP_SIZE_X = 8u;
const WORKGROUP_SIZE_Y = 8u;
const WORKGROUP_LANE_COUNT = 64u; // 8x8 = 64 threads per workgroup
const MIN_NORMAL_SIMILARITY = 0.95;

// Workgroup-shared memory for atomic reprojection competition
// Format per tile: (reprojection_score << 16) | (pixel_index & 0xFFFF)
var<workgroup> reprojection_best: array<atomic<u32>, WORKGROUP_LANE_COUNT>;
var<workgroup> tile_prev_tile_coords: array<vec2<u32>, WORKGROUP_LANE_COUNT>;
var<workgroup> tile_success_flag: array<u32, WORKGROUP_LANE_COUNT>;

var<workgroup> slot_prev_probe_index: array<u32, MAX_SCREEN_PROBE_PIXEL_COUNT>;
var<workgroup> slot_prev_pixel_coords: array<vec2<u32>, MAX_SCREEN_PROBE_PIXEL_COUNT>;
var<workgroup> slot_has_candidate: array<u32, MAX_SCREEN_PROBE_PIXEL_COUNT>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn cs(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let res = textureDimensions(gbuffer_position);
    let res_i32 = vec2<i32>(res);
    let probe_size = max(u32(gi_params.screen_probe_size), 1u);
    let grid_dims = grid_dimensions(res, probe_size);
    let total_probes = u32(gi_params.total_screen_probes);

    let view_index = u32(frame_info.view_index);
    let view = view_buffer[view_index];
    let camera_position = view.view_position.xyz;

    // Adaptive cell size calculation for reprojection tolerance
    let view_dimensions = vec2<f32>(f32(res.x), f32(res.y));
    let reciprocal_height = 1.0 / max(view_dimensions.y, 1.0);
    let anisotropic_term = view_dimensions.y / max(view_dimensions.x * view_dimensions.x, 1.0);
    let pixel_footprint = max(reciprocal_height, anisotropic_term);
    let distance_scale = tan(view.fov * gi_params.screen_probe_size * pixel_footprint);

    let lane_index = lid.y * WORKGROUP_SIZE_X + lid.x;
    let fits_x = probe_size <= WORKGROUP_SIZE_X;
    let fits_y = probe_size <= WORKGROUP_SIZE_Y;
    let tiles_per_axis = vec2<u32>(
        select(1u, max(1u, WORKGROUP_SIZE_X / probe_size), fits_x),
        select(1u, max(1u, WORKGROUP_SIZE_Y / probe_size), fits_y)
    );
    let tiles_per_group = max(tiles_per_axis.x * tiles_per_axis.y, 1u);
    let tile_pixel_count = probe_size * probe_size;
    let total_slots = tiles_per_group * tile_pixel_count;
    let tile_group_base = wid.xy * tiles_per_axis;

    // Reset slot bookkeeping using strided access
    for (var slot_index = lane_index; slot_index < total_slots; slot_index += WORKGROUP_LANE_COUNT) {
        slot_has_candidate[slot_index] = 0u;
    }

    workgroupBarrier();

    // Initialize reprojection competition state per tile
    if (lane_index < tiles_per_group) {
        atomicStore(&reprojection_best[lane_index], (pack_half_float(65504.0) << 16u) | 0xFFFFu);
    }

    workgroupBarrier();

    // =========================================================================
    // STEP 1: Reprojection Competition
    // =========================================================================
    for (var slot_index = lane_index; slot_index < total_slots; slot_index += WORKGROUP_LANE_COUNT) {
        let tile_local_index = slot_index / tile_pixel_count;
        let tile_local_coord = vec2<u32>(
            tile_local_index % tiles_per_axis.x,
            tile_local_index / tiles_per_axis.x
        );
        let probe_tile = tile_group_base + tile_local_coord;
        if (probe_tile.x >= grid_dims.x || probe_tile.y >= grid_dims.y) {
            continue;
        }

        let probe_index = probe_tile.y * grid_dims.x + probe_tile.x;
        if (probe_index >= total_probes) {
            continue;
        }

        let pixel_local_index = slot_index % tile_pixel_count;
        let local_offset = vec2<u32>(
            pixel_local_index % probe_size,
            pixel_local_index / probe_size
        );
        let pixel = probe_tile * probe_size + local_offset;

        let pixel_i32 = vec2<i32>(pixel);
        let normal_data = textureLoad(gbuffer_normal, pixel_i32, 0).xyz;

        if (length(normal_data) <= 0.0) {
            continue;
        }

        let normal_current = safe_normalize(normal_data);
        let motion_sample = textureLoad(gbuffer_motion, pixel_i32, 0);
        let pixel_velocity = motion_sample.xy * vec2<f32>(f32(res.x), f32(res.y)) * vec2<f32>(0.5, -0.5);

        let pixel_center = vec2<f32>(pixel) + 0.5;
        let pixel_prev_center = pixel_center - pixel_velocity;
        let pixel_prev = vec2<i32>(floor(pixel_prev_center));

        let pixel_prev_u32 = vec2<u32>(u32(pixel_prev.x), u32(pixel_prev.y));
        let probe_tile_prev = pixel_prev_u32 / probe_size;
        let probe_index_prev = probe_tile_prev.y * grid_dims.x + probe_tile_prev.x;

        if (probe_index_prev >= total_probes) {
            continue;
        }

        let position_current = textureLoad(gbuffer_position, pixel_i32, 0).xyz;
        let distance_to_camera = distance(camera_position, position_current);
        let adaptive_cell_size = max(distance_scale * distance_to_camera, 0.0001);

        let prev_probe_state = screen_probe_metadata[probe_index_prev].state;
        if (prev_probe_state.x <= 0.0) {
            continue;
        }

        let prev_probe_pixel = vec2<i32>(
            i32(prev_probe_state.y),
            i32(prev_probe_state.z)
        );

        let prev_probe_position = textureLoad(gbuffer_position_prev, prev_probe_pixel, 0).xyz;
        let prev_probe_normal = safe_normalize(textureLoad(gbuffer_normal_prev, prev_probe_pixel, 0).xyz);

        let plane_dist = abs(dot(prev_probe_position - position_current, normal_current));
        let normal_similarity = dot(prev_probe_normal, normal_current);

        if (plane_dist < adaptive_cell_size && normal_similarity > MIN_NORMAL_SIMILARITY) {
            let world_dist_3d = distance(prev_probe_position, position_current);

            slot_prev_probe_index[slot_index] = probe_index_prev;
            slot_prev_pixel_coords[slot_index] = pixel_prev_u32;
            slot_has_candidate[slot_index] = 1u;

            let packed_score = (pack_half_float(world_dist_3d) << 16u) | (pixel_local_index & 0xFFFFu);
            atomicMin(&reprojection_best[tile_local_index], packed_score);
        }
    }

    workgroupBarrier();

    if (lane_index < tiles_per_group) {
        tile_success_flag[lane_index] = 0u;
    }

    workgroupBarrier();

    // =========================================================================
    // STEP 2: Read Back Reprojection Result and Classify Tile
    // =========================================================================
    if (lane_index == 0u) {
        for (var tile_index = 0u; tile_index < tiles_per_group; tile_index += 1u) {
            let tile_local_coord = vec2<u32>(
                tile_index % tiles_per_axis.x,
                tile_index / tiles_per_axis.x
            );
            let probe_tile = tile_group_base + tile_local_coord;

            if (probe_tile.x >= grid_dims.x || probe_tile.y >= grid_dims.y) {
                continue;
            }

            let probe_index = probe_tile.y * grid_dims.x + probe_tile.x;
            if (probe_index >= total_probes) {
                continue;
            }

            let final_best = atomicLoad(&reprojection_best[tile_index]);
            let winning_pixel_index = final_best & 0xFFFFu;
            let found_valid_reprojection = winning_pixel_index != 0xFFFFu;

            if (found_valid_reprojection) {
                let slot_index = tile_index * tile_pixel_count + winning_pixel_index;
                let prev_probe_index = slot_prev_probe_index[slot_index];
                let prev_probe = screen_probe_metadata[prev_probe_index];

                let tile_corner = probe_tile * probe_size;
                let winner_pixel_offset = vec2<u32>(
                    winning_pixel_index % probe_size,
                    winning_pixel_index / probe_size
                );
                let winner_pixel = tile_corner + winner_pixel_offset;

                tile_prev_tile_coords[tile_index] = vec2<u32>(
                    prev_probe_index % grid_dims.x,
                    prev_probe_index / grid_dims.x
                );

                screen_probe_metadata[probe_index].state = vec4<f32>(
                    1.0,
                    f32(winner_pixel.x),
                    f32(winner_pixel.y),
                    min(prev_probe.state.w + 1.0, 1024.0);
                );

                tile_success_flag[tile_index] = 1u;

                let override_index = atomicAdd(&tile_counters.override_count, 1u);
                override_tiles[override_index] = probe_index;
            } else {
                screen_probe_metadata[probe_index].state.w = -1.0;
                let empty_index = atomicAdd(&tile_counters.empty_count, 1u);
                empty_tiles[empty_index] = probe_index;
            }
        }
    }

    workgroupBarrier();

    // =========================================================================
    // STEP 3: Per-Pixel Radiance Copy
    // =========================================================================
    for (var slot_index = lane_index; slot_index < total_slots; slot_index += WORKGROUP_LANE_COUNT) {
        let tile_local_index = slot_index / tile_pixel_count;
        let tile_local_coord = vec2<u32>(
            tile_local_index % tiles_per_axis.x,
            tile_local_index / tiles_per_axis.x
        );
        let probe_tile = tile_group_base + tile_local_coord;

        let pixel_local_index = slot_index % tile_pixel_count;
        let local_offset = vec2<u32>(
            pixel_local_index % probe_size,
            pixel_local_index / probe_size
        );
        let atlas_coord = probe_tile * probe_size + local_offset;

        if (tile_success_flag[tile_local_index] == 1u) {
            var prev_pixel_coord = select(
                tile_prev_tile_coords[tile_local_index] * probe_size + local_offset,
                slot_prev_pixel_coords[slot_index],
                slot_has_candidate[slot_index] == 1u
            );
            let radiance_sample = textureLoad(probe_radiance_prev, vec2<i32>(prev_pixel_coord), 0);
            textureStore(probe_radiance_curr, vec2<i32>(atlas_coord), radiance_sample);
        } else {
            textureStore(probe_radiance_curr, vec2<i32>(atlas_coord), vec4<f32>(0.0));
        }
    }
}
