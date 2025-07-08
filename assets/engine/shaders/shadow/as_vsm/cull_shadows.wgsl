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

@group(1) @binding(0) var<storage, read> aabb_bounds: array<AABBNodeBounds>;
@group(1) @binding(1) var<storage, read> visible_object_instances_no_occlusion: array<i32>;
@group(1) @binding(2) var<storage, read_write> visible_object_instances: array<i32>;
@group(1) @binding(3) var<storage, read> object_instances: array<ObjectInstance>;
@group(1) @binding(4) var<storage, read> entity_aabb_node_indices: array<u32>;
@group(1) @binding(5) var<uniform> draw_cull_data: DrawCullData;
@group(1) @binding(6) var<storage, read_write> draw_indirect_buffer: array<DrawCommand>;
@group(1) @binding(7) var<uniform> vsm_settings: ASVSMSettings;
@group(1) @binding(8) var<storage, read> light_shadow_idx_buffer: array<u32>;
@group(1) @binding(9) var page_table: texture_storage_2d_array<r32uint, read>;

// ------------------------------------------------------------------------------------
// Helper Functions
// ------------------------------------------------------------------------------------ 

// Returns a swept AABB (min, max) along the light direction
fn compute_swept_aabb(min_point: vec3<f32>, max_point: vec3<f32>, light_dir: vec3<f32>, sweep_dist: f32) -> array<vec3<f32>, 2> {
    let sweep_vec = light_dir * sweep_dist;
    let swept_min = min_point + sweep_vec;
    let swept_max = max_point + sweep_vec;
    return array<vec3<f32>, 2>(swept_min, swept_max);
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
    
    let clipmap_index = draw_cull_data.clipmap_index;
    if (clipmap_index >= u32(vsm_settings.max_lods)) {
        return;
    }

    let object_instance_index = visible_object_instances_no_occlusion[g_id];
    if (object_instance_index == -1) {
        return;
    }

    let light_index = global_id.y;
    let shadow_idx = light_shadow_idx_buffer[light_index];
    if (shadow_idx == 0xffffffffu) {
        return;
    }

    let object_instance = object_instances[object_instance_index];
    let entity_index = get_entity_row(object_instance.row);
    let node_index = entity_aabb_node_indices[entity_index];
    if (node_index == 0u) {
        return;
    }

    let bounds = aabb_bounds[node_index];
    let min_point = bounds.min_point.xyz;
    let max_point = bounds.max_point.xyz;

    // Get light direction (assume directional for now)
    let view      = view_buffer[draw_cull_data.view_index];
    let vp_matrix = view.view_projection_matrix;

    // Compute all 8 corners of the swept AABB (replaced by center + conservative radius approach)
    let swept = compute_swept_aabb(min_point, max_point, normalize(view.view_direction.xyz), 1000.0);
    let swept_min = swept[0];
    let swept_max = swept[1];

    // Project 8 extreme corners of the swept AABB into tile space
    let swept_corners = array<vec3<f32>, 8>(
        vec3<f32>(swept_min.x, swept_min.y, swept_min.z),
        vec3<f32>(swept_min.x, swept_max.y, swept_max.z),
        vec3<f32>(swept_max.x, swept_min.y, swept_min.z),
        vec3<f32>(swept_max.x, swept_max.y, swept_max.z),
        vec3<f32>(swept_min.x, swept_min.y, swept_max.z),
        vec3<f32>(swept_min.x, swept_max.y, swept_min.z),
        vec3<f32>(swept_max.x, swept_min.y, swept_max.z),
        vec3<f32>(swept_max.x, swept_max.y, swept_min.z)
    );

    var min_tile = vec2<u32>(0xffffffffu, 0xffffffffu);
    var max_tile = vec2<u32>(0u, 0u);
    for (var i = 0u; i < 8u; i = i + 1u) {
        let tile_info = vsm_world_to_virtual_tile_for_clip(
            vec4<f32>(swept_corners[i], 1.0),
            vp_matrix,
            vsm_settings,
            clipmap_index
        );
        min_tile = min(min_tile, tile_info.tile_coords);
        max_tile = max(max_tile, tile_info.tile_coords);
    }

    let vtr = u32(vsm_settings.virtual_tiles_per_row);
    let unclamped_min_x = min(min_tile.x, max_tile.x);
    let unclamped_max_x = max(min_tile.x, max_tile.x);
    let unclamped_min_y = min(min_tile.y, max_tile.y);
    let unclamped_max_y = max(min_tile.y, max_tile.y);

    // let min_x = clamp(unclamped_min_x, 0u, vtr);
    // let max_x = clamp(unclamped_max_x, 0u, vtr - 1u);
    // let min_y = clamp(unclamped_min_y, 0u, vtr - 1u);
    // let max_y = clamp(unclamped_max_y, 0u, vtr - 1u);
    // TODO: Remove this once we have a proper way to handle some edge cases, namely
    // Getting the proper tile coords for the swept AABB such that they catch dirty pages properly. 
    let min_x = 0u;
    let max_x = vtr - 1u;
    let min_y = 0u;
    let max_y = vtr - 1u;

    let slice_idx = shadow_idx * u32(vsm_settings.max_lods) + clipmap_index;

    var dirty = false;
    for (var y = min_y; y <= max_y; y = y + 1u) {
        for (var x = min_x; x <= max_x; x = x + 1u) {
            let tile_coords = vec2<u32>(x, y);
            dirty = dirty || vsm_pte_is_dirty(textureLoad(page_table, tile_coords, slice_idx).r);
        }
    }

    if (!dirty) {
        return;
    }

    let batch_index = object_instance.batch;
    let first_instance = draw_indirect_buffer[batch_index].first_instance;
    let count_index = atomicAdd(&draw_indirect_buffer[batch_index].instance_count, 1u);
    let instance_index = first_instance + count_index;
    visible_object_instances[instance_index] = object_instance_index;
#endif
}