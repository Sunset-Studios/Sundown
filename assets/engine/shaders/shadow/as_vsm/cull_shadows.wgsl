#include "common.wgsl"
#include "shadow/shadows_common.wgsl"

// ------------------------------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------------------------------ 

struct DrawCullData {
    draw_count: u32,
    view_index: u32,
    clipmap_index: u32,
}

// ------------------------------------------------------------------------------------
// Buffers
// ------------------------------------------------------------------------------------ 

@group(1) @binding(0) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(1) var<storage, read> visible_object_instances_no_occlusion: array<i32>;
@group(1) @binding(2) var<storage, read_write> visible_object_instances: array<i32>;
@group(1) @binding(3) var<storage, read> object_instances: array<ObjectInstance>;
@group(1) @binding(4) var<uniform> draw_cull_data: DrawCullData;
@group(1) @binding(5) var<storage, read_write> draw_indirect_buffer: array<DrawCommand>;
@group(1) @binding(6) var<uniform> vsm_settings: ASVSMSettings;
@group(1) @binding(7) var<storage, read> entity_flags: array<u32>;
@group(1) @binding(8) var<storage, read> bitmask: array<u32>;
@group(1) @binding(9) var page_table: texture_storage_2d_array<r32uint, read_write>;
@group(1) @binding(10) var page_offset: texture_storage_2d_array<rgba32float, write>;

// ------------------------------------------------------------------------------------
// Helper Functions
// ------------------------------------------------------------------------------------ 

// Returns a swept AABB (min, max) along the light direction
fn compute_swept_aabb(
    min_point : vec3<f32>,
    max_point : vec3<f32>,
    light_dir : vec3<f32>,
    sweep_dist: f32
) -> array<vec3<f32>, 2> {
    let min_swept = min(min_point, min_point + light_dir * sweep_dist);
    let max_swept = max(max_point, max_point + light_dir * sweep_dist);
    return array<vec3<f32>, 2>(min_swept, max_swept);
}

// ------------------------------------------------------------------------------------
// Compute Shader
// ------------------------------------------------------------------------------------ 

@compute @workgroup_size(256)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
#if SHADOWS_ENABLED
    let g_id = global_id.x;
    if (g_id >= u32(draw_cull_data.draw_count)) {
        return;
    }
    
    // Early-out for invalid clip-level
    let clipmap_index = draw_cull_data.clipmap_index;
    if (clipmap_index >= u32(vsm_settings.max_lods)) {
        return;
    }

    // Resolve the visible instance
    let object_instance_index = visible_object_instances_no_occlusion[g_id];
    if (object_instance_index == -1) {
        return;
    }

    let light_shadow_idx = global_id.y;
    let object_instance  = object_instances[object_instance_index];
    let entity_index     = get_entity_row(object_instance.row);

    // ────────────────────────────────────────────────────────────────
    // Derive position & radius from the entity's transform
    // ────────────────────────────────────────────────────────────────
    let entity_transform = entity_transforms[entity_index].transform;

    // World-space translation
    let position = entity_transform[3].xyz;

    // Largest axis-scale → conservative sphere radius
    let scale_x = length(entity_transform[0].xyz);
    let scale_y = length(entity_transform[1].xyz);
    let scale_z = length(entity_transform[2].xyz);
    let radius  = max(max(scale_x, scale_y), scale_z);

    // Re-use existing swept-AABB logic by constructing a box around the sphere
    var min_point = position - vec3<f32>(radius);
    var max_point = position + vec3<f32>(radius);

    // ────────────────────────────────────────────────────────────────
    // Project bounds into virtual-tile space
    // ────────────────────────────────────────────────────────────────
    let view      = view_buffer[draw_cull_data.view_index];
    let vp_matrix = view.view_projection_matrix;

    let swept          = compute_swept_aabb(min_point, max_point, normalize(view.view_direction.xyz), 100.0);
    let swept_min      = swept[0];
    let swept_max      = swept[1];

    let first_tile_info  = vsm_world_to_virtual_tile_for_clip(vec4<f32>(min_point, 1.0),  vp_matrix, vsm_settings, clipmap_index);
    let second_tile_info = vsm_world_to_virtual_tile_for_clip(vec4<f32>(max_point, 1.0), vp_matrix, vsm_settings, clipmap_index);
    let third_tile_info  = vsm_world_to_virtual_tile_for_clip(vec4<f32>(swept_min, 1.0),  vp_matrix, vsm_settings, clipmap_index);
    let fourth_tile_info = vsm_world_to_virtual_tile_for_clip(vec4<f32>(swept_max, 1.0), vp_matrix, vsm_settings, clipmap_index);

    var min_tile = min(first_tile_info.tile_coords, second_tile_info.tile_coords);
    min_tile     = min(min_tile, third_tile_info.tile_coords);
    min_tile     = min(min_tile, fourth_tile_info.tile_coords);

    var max_tile = max(first_tile_info.tile_coords, second_tile_info.tile_coords);
    max_tile     = max(max_tile, third_tile_info.tile_coords);
    max_tile     = max(max_tile, fourth_tile_info.tile_coords);

    let slice_idx = light_shadow_idx * u32(vsm_settings.max_lods) + clipmap_index;

    // ────────────────────────────────────────────────────────────────
    // Dirty-page tracking (wrap-aware)
    // ────────────────────────────────────────────────────────────────
    var dirty = false;
    let flag  = entity_flags[entity_index];
    let vtr   = u32(vsm_settings.virtual_tiles_per_row);   // tiles-per-row

    if ((flag & EF_MOVED) != 0u) {
        for (var y_off = min_tile.y; y_off <= max_tile.y; y_off = y_off + 1u) {
            for (var x_off = min_tile.x; x_off <= max_tile.x; x_off = x_off + 1u) {
                let tile_coords = vec2<u32>(x_off, y_off);

                let word_and_mask = vsm_get_virtual_tile_word_and_mask(
                    tile_coords,
                    clipmap_index,
                    light_shadow_idx,
                    vsm_settings
                );
                let word = word_and_mask.x;
                let mask = word_and_mask.y;
                let not_visible = (bitmask[word] & mask) == 0u; 
                if (not_visible) {
                    continue;
                }

                let pte      = textureLoad(page_table, tile_coords, slice_idx).r;
                textureStore(page_table, tile_coords, slice_idx, vec4<u32>(pte | pte_dirty_mask));

                textureStore(page_offset, tile_coords, slice_idx, vec4<f32>(view.view_matrix[3]));

                dirty = true;
            }
        }
    }

    // Fallback scan (temporary workaround for any remaining edge-cases)
    for (var y = 0u; y < vtr; y = y + 1u) {
        for (var x = 0u; x < vtr; x = x + 1u) {
            let tile_coords = vec2<u32>(x, y);
            dirty = dirty || vsm_pte_is_dirty(textureLoad(page_table, tile_coords, slice_idx).r);
        }
    }

    if (!dirty) {
        return;
    }

    // ────────────────────────────────────────────────────────────────
    // Append visible instance to the indirect draw buffer
    // ────────────────────────────────────────────────────────────────
    let batch_index    = object_instance.batch;
    let first_instance = draw_indirect_buffer[batch_index].first_instance;
    let count_index    = atomicAdd(&draw_indirect_buffer[batch_index].instance_count, 1u);
    let instance_index = first_instance + count_index;
    visible_object_instances[instance_index] = object_instance_index;
#endif
}