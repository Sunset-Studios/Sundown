#include "common.wgsl"
#include "acceleration_common.wgsl"
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

@group(1) @binding(0) var<storage, read> bounds: array<AABB>;
@group(1) @binding(1) var<storage, read> visible_object_instances_no_occlusion: array<i32>;
@group(1) @binding(2) var<storage, read_write> visible_object_instances: array<i32>;
@group(1) @binding(3) var<storage, read> object_instances: array<ObjectInstance>;
@group(1) @binding(4) var<uniform> draw_cull_data: DrawCullData;
@group(1) @binding(5) var<uniform> vsm_settings: ASVSMSettings;
@group(1) @binding(6) var<storage, read> entity_flags: array<u32>;
@group(1) @binding(7) var<storage, read> bitmask: array<u32>;
@group(1) @binding(8) var<storage, read> entity_index_lookup: array<u32>;
@group(1) @binding(9) var page_table: texture_storage_2d_array<r32uint, read_write>;
@group(1) @binding(10) var page_offset: texture_storage_2d_array<rgba32float, write>;

// ------------------------------------------------------------------------------------
// Helper Functions
// ------------------------------------------------------------------------------------ 

// Function to mark a tile
fn mark_tile(tx: u32, ty: u32, slice: u32, clipmap_index: u32, light_shadow_idx: u32, view: ptr<function, View>) {
    let tile_coords = vec2<u32>(tx, ty);
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
        var pte = textureLoad(page_table, tile_coords, slice).r;
        pte = pte | pte_dirty_mask;
        textureStore(page_table, tile_coords, slice, vec4<u32>(pte));
        textureStore(page_offset, tile_coords, slice, vec4<f32>(view.view_matrix[3]));
    }
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
    let entity_index     = entity_index_lookup[get_entity_row(object_instance.row)];
    if (entity_index >= arrayLength(&entity_flags)) {
        return;
    }

    // Setup some view variables
    var view = view_buffer[draw_cull_data.view_index];
    let vp_matrix = view.view_projection_matrix;
    let slice_idx = light_shadow_idx * u32(vsm_settings.max_lods) + clipmap_index;
    let vtr_i = i32(vsm_settings.virtual_tiles_per_row);

    let light_direction = normalize(view.view_direction.xyz);
    let vp_clip = vsm_snapped_translation_for_lod(vp_matrix, clipmap_index, vsm_settings);

    // Derive position & radius from the entity's transform
    let entity_moved = (entity_flags[entity_index] & EF_MOVED) != 0u;
    let entity_bounds = bounds[entity_index];

    // Current world-space translation and per-axis scales
    let position = (entity_bounds.min.xyz + entity_bounds.max.xyz) * 0.5;
    let scale = (entity_bounds.max.xyz - entity_bounds.min.xyz) * 0.5;

    // Replace projection of corners and min/max computation with world-to-tile mapping
    if (entity_moved) {
        var min_render_clip = vec2<f32>(99999.0);
        var max_render_clip = vec2<f32>(-99999.0);
        var min_sample_clip = vec2<f32>(99999.0);
        var max_sample_clip = vec2<f32>(-99999.0);

        // Project all 8 world-space corners
        for (var i = 0u; i < 8u; i = i + 1u) {
            let sx = select(-1.0, 1.0, (i & 1u) != 0u);
            let sy = select(-1.0, 1.0, (i & 2u) != 0u);
            let sz = select(-1.0, 1.0, (i & 4u) != 0u);

            let world = position + scale.xyz * vec3<f32>(sx, sy, sz);

            let render_clip = vsm_calculate_render_clip_value_from_world_pos(
                vec4<f32>(world, 1.0),
                clipmap_index,
                vp_matrix,
                vsm_settings
            ).xy;
            let sample_clip = vsm_calculate_sample_clip_value_from_world_pos(
                vec4<f32>(world, 1.0),
                clipmap_index,
                vp_matrix,
                vsm_settings
            ).xy;

            min_render_clip = min(min_render_clip, render_clip);
            max_render_clip = max(max_render_clip, render_clip);
            min_sample_clip = min(min_sample_clip, sample_clip);
            max_sample_clip = max(max_sample_clip, sample_clip);
        }

        // Early out if entirely outside the render clip bounds
        if (min_render_clip.x > 1.0 || min_render_clip.y > 1.0 || max_render_clip.x < -1.0 || max_render_clip.y < -1.0) {
            return;
        }

        // Compute unwrapped UV range
        let min_uv = min_sample_clip * 0.5 + 0.5;
        let max_uv = max_sample_clip * 0.5 + 0.5;

        let virtual_dim = vsm_settings.virtual_dim;
        let tile_size = vsm_settings.tile_size;

        // Conservative tile range with padding
        let min_tile_x = i32(floor(min_uv.x * virtual_dim / tile_size));
        let max_tile_x = i32(floor(max_uv.x * virtual_dim / tile_size));
        let min_tile_y = i32(floor(min_uv.y * virtual_dim / tile_size));
        let max_tile_y = i32(floor(max_uv.y * virtual_dim / tile_size));

        // Compute spans
        let span_x = max_tile_x - min_tile_x + 1;
        let span_y = max_tile_y - min_tile_y + 1;
        let tile_count = vtr_i;

        // If invalid range, skip
        if (span_x <= 0 || span_y <= 0) {
            return;
        }

        // If area too large, mark all tiles
        if (span_x * span_y > 256) {
            for (var ty = 0; ty < tile_count; ty = ty + 1) {
                for (var tx = 0; tx < tile_count; tx = tx + 1) {
                    mark_tile(u32(tx), u32(ty), slice_idx, clipmap_index, light_shadow_idx, &view);
                }
            }
        } else {
            // Loop over unwrapped tile range, marking wrapped coordinates
            for (var ty = min_tile_y; ty <= max_tile_y; ty = ty + 1) {
                let wrapped_ty = ((ty % tile_count) + tile_count) % tile_count;
                for (var tx = min_tile_x; tx <= max_tile_x; tx = tx + 1) {
                    let wrapped_tx = ((tx % tile_count) + tile_count) % tile_count;
                    mark_tile(u32(wrapped_tx), u32(wrapped_ty), slice_idx, clipmap_index, light_shadow_idx, &view);
                }
            }
        }
    }
#endif
}