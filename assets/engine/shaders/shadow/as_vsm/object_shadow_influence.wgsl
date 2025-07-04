// AS-VSM Stage A2: Object Shadow Influence
// For each entity and tile (for a given light), checks if the entity's shadow volume (AABB swept along the light direction)
// overlaps the tile (if the tile is marked in the feedback bitmask). If so, marks the entity in got_shadow_feedback_buffer.
#include "common.wgsl"
#include "lighting_common.wgsl"
#include "shadow/shadows_common.wgsl"

struct ObjectShadowInfluenceParams {
    entity_offset: u32,
}

@group(1) @binding(0) var<storage, read> aabb_bounds: array<AABBNodeBounds>;
@group(1) @binding(1) var<storage, read> entity_aabb_node_indices: array<u32>;
@group(1) @binding(2) var<uniform> vsm_settings: ASVSMSettings;
@group(1) @binding(3) var<uniform> osi_params: ObjectShadowInfluenceParams;
@group(1) @binding(4) var<storage, read> bitmask: array<u32>;
@group(1) @binding(5) var<storage, read> light_view_buffer: array<u32>;
@group(1) @binding(6) var<storage, read> light_shadow_idx_buffer: array<u32>;
@group(1) @binding(7) var<storage, read> light_count_buffer: array<u32>;
@group(1) @binding(8) var<storage, read_write> got_shadow_feedback_buffer: array<atomic<u32>>;
@group(1) @binding(9) var page_table: texture_storage_2d_array<r32uint, read>;

// Returns a swept AABB (min, max) along the light direction
fn compute_swept_aabb(min_point: vec3<f32>, max_point: vec3<f32>, light_dir: vec3<f32>, sweep_dist: f32) -> array<vec3<f32>, 2> {
    let sweep_vec = normalize(light_dir) * sweep_dist;
    let swept_min = min(min_point, min_point + sweep_vec);
    let swept_max = max(max_point, max_point + sweep_vec);
    return array<vec3<f32>, 2>(swept_min, swept_max);
}

@compute @workgroup_size(32, 8, 1)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
#if SHADOWS_ENABLED
    let clipmap_index = global_id.y;
    if (clipmap_index >= u32(vsm_settings.max_lods)) {
        return;
    }

    let entity_index = osi_params.entity_offset + global_id.x;
    if (entity_index >= arrayLength(&got_shadow_feedback_buffer)) {
        return;
    }

    let light_index = global_id.z;
    let light_count = light_count_buffer[0u];
    if (light_index >= light_count) {
        return;
    }

    let shadow_idx = light_shadow_idx_buffer[light_index];
    if (shadow_idx == 0xffffffffu) {
        return;
    }

    let node_index = entity_aabb_node_indices[entity_index];
    if (node_index == 0u) {
        return;
    }

    let bounds = aabb_bounds[node_index];
    let min_point = bounds.min_point.xyz;
    let max_point = bounds.max_point.xyz;

    // Get light direction (assume directional for now)
    let view_idx  = light_view_buffer[light_index];
    let view      = view_buffer[view_idx];

    // Compute all 8 corners of the swept AABB (replaced by center + conservative radius approach)
    let swept = compute_swept_aabb(min_point, max_point, -view.view_position.xyz, 100.0);
    let swept_min = swept[0];
    let swept_max = swept[1];

    // Project 4 extreme corners of the swept AABB into tile space
    let swept_corners = array<vec3<f32>, 4>(
        vec3<f32>(swept_min.x, swept_min.y, swept_min.z),
        vec3<f32>(swept_min.x, swept_max.y, swept_max.z),
        vec3<f32>(swept_max.x, swept_min.y, swept_min.z),
        vec3<f32>(swept_max.x, swept_max.y, swept_max.z)
    );

    var min_tile = vec2<u32>(0xffffffffu, 0xffffffffu);
    var max_tile = vec2<u32>(0u, 0u);
    for (var i = 0u; i < 4u; i = i + 1u) {
        let tile_info = vsm_world_to_virtual_tile_for_clip(
            vec4<f32>(swept_corners[i], 1.0),
            view.view_projection_matrix,
            vsm_settings,
            clipmap_index
        );
        min_tile = min(min_tile, tile_info.tile_coords);
        max_tile = max(max_tile, tile_info.tile_coords);
    }

    let vtr = u32(vsm_settings.virtual_tiles_per_row);
    let min_x = clamp(min_tile.x, 0u, vtr - 1u);
    let max_x = clamp(max_tile.x, 0u, vtr - 1u);
    let min_y = clamp(min_tile.y, 0u, vtr - 1u);
    let max_y = clamp(max_tile.y, 0u, vtr - 1u);
    
    for (var y = min_y; y <= max_y; y = y + 1u) {
        for (var x = min_x; x <= max_x; x = x + 1u) {
            let tile_coords = vec2<u32>(x, y);
            let word_and_mask = vsm_get_virtual_tile_word_and_mask(
                tile_coords,
                clipmap_index,
                shadow_idx,
                vsm_settings
            );
            let word_index = word_and_mask.x;
            let mask = word_and_mask.y;
            let word = bitmask[word_index];

            if ((word & mask) != 0u) {
                atomicOr(&got_shadow_feedback_buffer[entity_index], u32(1u) << clipmap_index);
                return;
            }
        }
    }
#endif
} 