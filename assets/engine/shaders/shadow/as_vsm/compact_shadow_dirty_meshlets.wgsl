#include "common.wgsl"
#include "visibility/visibility_common.wgsl"
#include "shadow/shadows_common.wgsl"

struct ShadowMeshletCullParams {
    view_index: u32,
    clipmap_index: u32,
    shadow_index: u32,
    meshlet_count: u32,
};

@group(1) @binding(0) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(1) var<storage, read> object_instances: array<ObjectInstance>;
@group(1) @binding(2) var<storage, read> in_visible_meshlets: array<vec4<u32>>;
@group(1) @binding(3) var<storage, read> in_draw_command: array<MeshletDrawCommandNoAtomics>;
@group(1) @binding(4) var<storage, read> entity_index_lookup: array<u32>;
@group(1) @binding(5) var<storage, read> meshlets: array<MeshletRecord>;
@group(1) @binding(6) var<uniform> params: ShadowMeshletCullParams;
@group(1) @binding(7) var<uniform> vsm_settings: ASVSMSettings;
@group(1) @binding(8) var page_table: texture_storage_2d_array<r32uint, read>;
@group(1) @binding(9) var<storage, read_write> out_visible_meshlets: array<vec4<u32>>;
@group(1) @binding(10) var<storage, read_write> out_draw_command: array<MeshletDrawCommand>;

fn tile_is_dirty_and_resident(tile_coords: vec2<u32>, slice_index: u32) -> bool {
    let pte = textureLoad(page_table, tile_coords, slice_index).r;
    return vsm_pte_is_resident(pte) && vsm_pte_is_dirty(pte);
}

fn meshlet_overlaps_dirty_page(
    meshlet: MeshletRecord,
    transform: mat4x4<f32>,
    clipmap_index: u32,
    slice_index: u32,
    view: ptr<function, View>
) -> bool {
    var min_render_clip = vec2<f32>(99999.0);
    var max_render_clip = vec2<f32>(-99999.0);
    var min_sample_clip = vec2<f32>(99999.0);
    var max_sample_clip = vec2<f32>(-99999.0);

    for (var i = 0u; i < 8u; i = i + 1u) {
        let sx = select(-1.0, 1.0, (i & 1u) != 0u);
        let sy = select(-1.0, 1.0, (i & 2u) != 0u);
        let sz = select(-1.0, 1.0, (i & 4u) != 0u);

        let local = vec3<f32>(
            select(meshlet.bounds_min.x, meshlet.bounds_max.x, sx > 0.0),
            select(meshlet.bounds_min.y, meshlet.bounds_max.y, sy > 0.0),
            select(meshlet.bounds_min.z, meshlet.bounds_max.z, sz > 0.0)
        );
        let world = transform * vec4<f32>(local, 1.0);

        let render_clip = vsm_calculate_render_clip_value_from_world_pos(
            world,
            clipmap_index,
            view.view_projection_matrix,
            vsm_settings
        ).xy;
        let sample_clip = vsm_calculate_sample_clip_value_from_world_pos(
            world,
            clipmap_index,
            view.view_projection_matrix,
            vsm_settings
        ).xy;

        min_render_clip = min(min_render_clip, render_clip);
        max_render_clip = max(max_render_clip, render_clip);
        min_sample_clip = min(min_sample_clip, sample_clip);
        max_sample_clip = max(max_sample_clip, sample_clip);
    }

    if (min_render_clip.x > 1.0 || min_render_clip.y > 1.0 || max_render_clip.x < -1.0 || max_render_clip.y < -1.0) {
        return false;
    }

    let min_uv = min_sample_clip * 0.5 + 0.5;
    let max_uv = max_sample_clip * 0.5 + 0.5;
    let min_tile_x = i32(floor(min_uv.x * vsm_settings.virtual_dim / vsm_settings.tile_size));
    let max_tile_x = i32(floor(max_uv.x * vsm_settings.virtual_dim / vsm_settings.tile_size));
    let min_tile_y = i32(floor(min_uv.y * vsm_settings.virtual_dim / vsm_settings.tile_size));
    let max_tile_y = i32(floor(max_uv.y * vsm_settings.virtual_dim / vsm_settings.tile_size));

    let span_x = max_tile_x - min_tile_x + 1;
    let span_y = max_tile_y - min_tile_y + 1;
    let tile_count = i32(vsm_settings.virtual_tiles_per_row);

    if (span_x <= 0 || span_y <= 0 || tile_count <= 0) {
        return false;
    }

    let wrapped_span_x = min(span_x, tile_count);
    let wrapped_span_y = min(span_y, tile_count);

    for (var ty = 0; ty < wrapped_span_y; ty = ty + 1) {
        let wrapped_ty = (((min_tile_y + ty) % tile_count) + tile_count) % tile_count;
        for (var tx = 0; tx < wrapped_span_x; tx = tx + 1) {
            let wrapped_tx = (((min_tile_x + tx) % tile_count) + tile_count) % tile_count;
            if (tile_is_dirty_and_resident(vec2<u32>(u32(wrapped_tx), u32(wrapped_ty)), slice_index)) {
                return true;
            }
        }
    }

    return false;
}

@compute @workgroup_size(128)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
#if SHADOWS_ENABLED
    let visible_index = gid.x;
    let visible_count = in_draw_command[0].instance_count;
    if (visible_index >= visible_count || visible_index >= arrayLength(&in_visible_meshlets)) {
        return;
    }

    let visible_entry = in_visible_meshlets[visible_index];
    let object_instance_index = meshlet_object_index(visible_entry);
    let global_meshlet_index = meshlet_index(visible_entry);
    if (object_instance_index >= arrayLength(&object_instances) || global_meshlet_index >= arrayLength(&meshlets)) {
        return;
    }

    let entity_row = get_entity_row(object_instances[object_instance_index].row);
    if (entity_row >= arrayLength(&entity_index_lookup)) {
        return;
    }

    let entity_resolved = entity_index_lookup[entity_row];
    if (entity_resolved == INVALID_IDX || entity_resolved >= arrayLength(&entity_transforms)) {
        return;
    }

    let slice_index = params.shadow_index * u32(vsm_settings.max_lods) + params.clipmap_index;
    var view = view_buffer[params.view_index];
    let meshlet = meshlets[global_meshlet_index];
    let transform = entity_transforms[entity_resolved].transform;

    if (!meshlet_overlaps_dirty_page(meshlet, transform, params.clipmap_index, slice_index, &view)) {
        return;
    }

    let append_index = atomicAdd(&out_draw_command[0].instance_count, 1u);
    out_visible_meshlets[append_index] = visible_entry;
#endif
}
