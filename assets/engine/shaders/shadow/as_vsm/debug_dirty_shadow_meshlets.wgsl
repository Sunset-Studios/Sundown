#include "common.wgsl"
#include "visibility/visibility_common.wgsl"
#include "shadow/shadows_common.wgsl"

struct DebugDirtyShadowMeshletParams {
    clipmap_index: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(1) @binding(0) var out_image: texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(1) var depth_texture: texture_2d<f32>;
@group(1) @binding(2) var visibility_entity_texture: texture_2d<u32>;
@group(1) @binding(3) var visibility_surface_texture: texture_2d<u32>;
@group(1) @binding(4) var<uniform> vsm_settings: ASVSMSettings;
@group(1) @binding(5) var<storage, read> light_view_buffer: array<u32>;
@group(1) @binding(6) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(7) var<storage, read> object_instances: array<ObjectInstance>;
@group(1) @binding(8) var<storage, read> entity_index_lookup: array<u32>;
@group(1) @binding(9) var<storage, read> dirty_visible_meshlets: array<vec4<u32>>;
@group(1) @binding(10) var<storage, read> dirty_draw_command: array<MeshletDrawCommandNoAtomics>;
@group(1) @binding(11) var<uniform> params: DebugDirtyShadowMeshletParams;

fn base_meshlet_color(meshlet_index: u32) -> vec3<f32> {
    let tint = id_to_color(meshlet_index);
    let shade = 0.78 + 0.18 * fract(f32(meshlet_index) * 0.75487766);
    let gray = vec3<f32>(shade);
    return mix(gray, tint, 0.12);
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
#if SHADOWS_ENABLED
    let dims = textureDimensions(out_image);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let pixel_coord = vec2<i32>(gid.xy);
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let view_index = u32(frame_info.view_index);
    let tex_depth = textureLoad(depth_texture, pixel_coord, 0).x;
    if (tex_depth >= 1.0) {
        return;
    }

    let entity_id = textureLoad(visibility_entity_texture, pixel_coord, 0).x;
    if (entity_id == INVALID_IDX) {
        return;
    }

    let surface = textureLoad(visibility_surface_texture, pixel_coord, 0).x;
    let visible_meshlet_index = unpack_surface_meshlet(surface);
    let base_color = base_meshlet_color(visible_meshlet_index);

    if (params.clipmap_index == INVALID_IDX) {
        textureStore(out_image, gid.xy, vec4<f32>(base_color, 1.0));
        return;
    }

    let light_view_index = light_view_buffer[0u];
    if (light_view_index == INVALID_IDX) {
        return;
    }

    let camera_vp = view_buffer[view_index].view_projection_matrix;
    let clipmap0_vp = view_buffer[light_view_index].view_projection_matrix;
    let world_pos = reconstruct_world_position(uv, tex_depth, view_index);
    let vtile_info = vsm_world_to_virtual_tile(
        vec4<f32>(world_pos, 1.0),
        camera_vp,
        clipmap0_vp,
        vsm_settings
    );

    if (vtile_info.clipmap_index != params.clipmap_index) {
        return;
    }

    var color = base_color;
    let visible_count = min(dirty_draw_command[0].instance_count, arrayLength(&dirty_visible_meshlets));

    for (var i = 0u; i < visible_count; i = i + 1u) {
        let entry = dirty_visible_meshlets[i];
        let object_instance_index = meshlet_object_index(entry);
        let global_meshlet_index = meshlet_index(entry);
        if (global_meshlet_index != visible_meshlet_index) {
            continue;
        }

        if (object_instance_index >= arrayLength(&object_instances)) {
            continue;
        }

        let entity_row = get_entity_row(object_instances[object_instance_index].row);
        if (entity_row >= arrayLength(&entity_index_lookup)) {
            continue;
        }

        let entity_resolved = entity_index_lookup[entity_row];
        if (entity_resolved == INVALID_IDX || entity_resolved >= arrayLength(&entity_transforms)) {
            continue;
        }

        if (entity_resolved == entity_id) {
            color = id_to_color(global_meshlet_index);
            break;
        }
    }

    textureStore(out_image, gid.xy, vec4<f32>(color, 1.0));
#endif
}
