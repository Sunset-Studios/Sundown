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
@group(1) @binding(5) var<uniform> vsm_settings: ASVSMSettings;
@group(1) @binding(6) var<storage, read> entity_flags: array<u32>;
@group(1) @binding(7) var<storage, read> bitmask: array<u32>;
@group(1) @binding(8) var page_table: texture_storage_2d_array<r32uint, read_write>;
@group(1) @binding(9) var page_offset: texture_storage_2d_array<rgba32float, write>;

// Helper function to compute positive modulo
fn positive_mod(a: i32, b: i32) -> i32 {
    let mod_val = a % b;
    return select(mod_val, mod_val + b, mod_val < 0);
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
    if (entity_index >= arrayLength(&entity_flags)) {
        return;
    }

    // Setup some view variables
    let view = view_buffer[draw_cull_data.view_index];
    let vp_matrix = view.view_projection_matrix;
    let slice_idx = light_shadow_idx * u32(vsm_settings.max_lods) + clipmap_index;
    let vtr_i = i32(vsm_settings.virtual_tiles_per_row);
    let light_direction = normalize(view.view_direction.xyz);
    let vp_clip = vsm_projection_translation_clip(vp_matrix, clipmap_index, vsm_settings);

    // Derive position & radius from the entity's transform
    let entity_moved = (entity_flags[entity_index] & EF_MOVED) != 0u;
    let entity_transform = entity_transforms[entity_index];

    // Current world-space translation and per-axis scales
    let position = entity_transform.transform[3].xyz;
    let scale = vec3f(
        length(entity_transform.transform[0].xyz),
        length(entity_transform.transform[1].xyz),
        length(entity_transform.transform[2].xyz)
    ) / 100.0;

    // Compute current AABB
    let half_size = scale * 0.5;
    let curr_min = position - half_size;
    let curr_max = position + half_size;

    // Swept AABB covering previous and current positions
    let swept_min = min(curr_min + light_direction * 100.0, curr_min);
    let swept_max = max(curr_max + light_direction * 100.0, curr_max);

    // Project swept AABB into clipmap NDC space
    let corners = array<vec4<f32>, 8>(
        vec4<f32>(swept_min.x, swept_min.y, swept_min.z, 1.0),
        vec4<f32>(swept_max.x, swept_min.y, swept_min.z, 1.0),
        vec4<f32>(swept_min.x, swept_max.y, swept_min.z, 1.0),
        vec4<f32>(swept_max.x, swept_max.y, swept_min.z, 1.0),
        vec4<f32>(swept_min.x, swept_min.y, swept_max.z, 1.0),
        vec4<f32>(swept_max.x, swept_min.y, swept_max.z, 1.0),
        vec4<f32>(swept_min.x, swept_max.y, swept_max.z, 1.0),
        vec4<f32>(swept_max.x, swept_max.y, swept_max.z, 1.0)
    );

    var min_ndc = vec2<f32>(1e6, 1e6);
    var max_ndc = vec2<f32>(-1e6, -1e6);

    for (var i = 0u; i < 8u; i = i + 1u) {
        let ndc = vsm_calculate_render_clip_value_from_world_pos(
            corners[i],
            clipmap_index,
            vp_matrix,
            vsm_settings
        ).xy;
        min_ndc = min(min_ndc, ndc);
        max_ndc = max(max_ndc, ndc);
    }

    // If object is completely outside this clipmap's [-1,1] area, skip
    if (entity_moved && min_ndc.x <= 1.0 && min_ndc.y <= 1.0 && max_ndc.x >= -1.0 && max_ndc.y >= -1.0) {
        // Convert NDC to [0,1] UV coordinates on the virtual shadow map
        let uv_min = (min_ndc - vp_clip.xy) * 0.5 + 0.5;
        let uv_max = (max_ndc - vp_clip.xy) * 0.5 + 0.5;

        // Compute the length of the object
        let length = uv_max - uv_min;
        let adjusted_uv_min = uv_min - length * 0.5;
        let adjusted_uv_max = uv_max - length * 0.5;

        // Map UV to tile indices (page table coordinates) - raw (unwrapped)
        let raw_tile_min = vec2<i32>(floor(adjusted_uv_min * vsm_settings.virtual_dim) / vsm_settings.tile_size);
        let raw_tile_max = vec2<i32>(floor(adjusted_uv_max * vsm_settings.virtual_dim) / vsm_settings.tile_size);

        // Compute wrapped ranges for x and y independently
        let extent_x = raw_tile_max.x - raw_tile_min.x + 1;
        let extent_y = raw_tile_max.y - raw_tile_min.y + 1;

        var x_ranges: array<vec2<i32>, 2>;
        if (extent_x <= 0) {
            x_ranges = array(vec2<i32>(0, -1), vec2<i32>(0, -1));
        } else if (extent_x >= vtr_i) {
            x_ranges = array(vec2<i32>(0, vtr_i - 1), vec2<i32>(0, -1));
        } else {
            let start_x = positive_mod(raw_tile_min.x, vtr_i);
            let end_x = positive_mod(raw_tile_max.x, vtr_i);
            if (start_x <= end_x) {
                x_ranges = array(vec2<i32>(start_x, end_x), vec2<i32>(0, -1));
            } else {
                x_ranges = array(vec2<i32>(start_x, vtr_i - 1), vec2<i32>(0, end_x));
            }
        }

        var y_ranges: array<vec2<i32>, 2>;
        if (extent_y <= 0) {
            y_ranges = array(vec2<i32>(0, -1), vec2<i32>(0, -1));
        } else if (extent_y >= vtr_i) {
            y_ranges = array(vec2<i32>(0, vtr_i - 1), vec2<i32>(0, -1));
        } else {
            let start_y = positive_mod(raw_tile_min.y, vtr_i);
            let end_y = positive_mod(raw_tile_max.y, vtr_i);
            if (start_y <= end_y) {
                y_ranges = array(vec2<i32>(start_y, end_y), vec2<i32>(0, -1));
            } else {
                y_ranges = array(vec2<i32>(start_y, vtr_i - 1), vec2<i32>(0, end_y));
            }
        }

        // Loop over all valid y ranges
        for (var ry: u32 = 0u; ry < 2u; ry = ry + 1u) {
            let y_min = y_ranges[ry].x;
            let y_max = y_ranges[ry].y;
            if (y_min > y_max) { continue; }

            // Loop over all valid x ranges
            for (var rx: u32 = 0u; rx < 2u; rx = rx + 1u) {
                let x_min = x_ranges[rx].x;
                let x_max = x_ranges[rx].y;
                if (x_min > x_max) { continue; }

                for (var y = y_min; y <= y_max; y = y + 1) {
                    for (var x = x_min; x <= x_max; x = x + 1) {
                        let tile_coords = vec2<u32>(u32(x), u32(y));

                        let word_and_mask = vsm_get_virtual_tile_word_and_mask(
                            tile_coords,
                            clipmap_index,
                            light_shadow_idx,
                            vsm_settings
                        );

                        let word = word_and_mask.x;
                        let mask = word_and_mask.y;
                        let is_visible = (bitmask[word] & mask) != 0u;

                        if (is_visible) {
                            var pte = textureLoad(page_table, tile_coords, slice_idx).r;
                            pte = pte | pte_dirty_mask;
                            textureStore(page_table, tile_coords, slice_idx, vec4<u32>(pte));
                            textureStore(page_offset, tile_coords, slice_idx, vec4<f32>(view.view_matrix[3]));
                        }
                    }
                }
            }
        }
    }
#endif
}