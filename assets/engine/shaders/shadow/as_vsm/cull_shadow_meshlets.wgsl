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
@group(1) @binding(2) var<storage, read> meshlet_instances: array<MeshletInstance>;
@group(1) @binding(3) var<storage, read> entity_index_lookup: array<u32>;
@group(1) @binding(4) var<storage, read> meshlets: array<MeshletRecord>;
@group(1) @binding(5) var<uniform> params: ShadowMeshletCullParams;
@group(1) @binding(6) var<uniform> vsm_settings: ASVSMSettings;
@group(1) @binding(7) var<storage, read> dirty_slices: array<u32>;
@group(1) @binding(8) var<storage, read_write> out_visible_meshlets: array<vec4<u32>>;
@group(1) @binding(9) var<storage, read_write> out_draw_command: array<MeshletDrawCommand>;

@compute @workgroup_size(128)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
#if SHADOWS_ENABLED
    let meshlet_instance_index = gid.x;
    if (meshlet_instance_index >= params.meshlet_count || meshlet_instance_index >= arrayLength(&meshlet_instances)) {
        return;
    }

    let slice_index = params.shadow_index * u32(vsm_settings.max_lods) + params.clipmap_index;
    if (slice_index >= arrayLength(&dirty_slices) || dirty_slices[slice_index] != u32(frame_info.frame_index)) {
        return;
    }

    let meshlet_instance = meshlet_instances[meshlet_instance_index];
    let object_instance_index = meshlet_instance.object_instance_index;
    let global_meshlet_index = meshlet_instance.meshlet_index;
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

    let transform = entity_transforms[entity_resolved].transform;
    let meshlet = meshlets[global_meshlet_index];
    let center_world = transform * vec4<f32>(meshlet.center_radius.xyz, 1.0);
    let radius_world = meshlet.center_radius.w * transform_max_scale(transform) * 1.2;

    var view = view_buffer[params.view_index];
    // TODO: Might want actual frustum/prism plane checks for actual culling here.

    let append_index = atomicAdd(&out_draw_command[0].instance_count, 1u);
    out_visible_meshlets[append_index] = vec4<u32>(object_instance_index, global_meshlet_index, 0u, 0u);
#endif
}
