#include "common.wgsl"
#include "visibility/visibility_common.wgsl"

struct MeshletCullParams {
    view_index: u32,
    meshlet_count: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(1) @binding(0) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(1) var<storage, read> object_instances: array<ObjectInstance>;
@group(1) @binding(2) var<storage, read> meshlet_instances: array<MeshletInstance>;
@group(1) @binding(3) var<storage, read> entity_index_lookup: array<u32>;
@group(1) @binding(4) var<storage, read> meshlets: array<MeshletRecord>;
@group(1) @binding(5) var<uniform> params: MeshletCullParams;
@group(1) @binding(6) var<storage, read_write> out_visible_meshlets: array<vec4<u32>>;
@group(1) @binding(7) var<storage, read_write> out_draw_command: array<MeshletDrawCommand>;

fn is_in_frustum(center: vec4<f32>, radius: f32, view: ptr<function, View>) -> u32 {
    var visible = 1u;

    // Check all frustum planes
    visible *= u32(dot(view.frustum[0], center) > -radius);
    visible *= u32(dot(view.frustum[1], center) > -radius);
    visible *= u32(dot(view.frustum[2], center) > -radius);
    visible *= u32(dot(view.frustum[3], center) > -radius);
    visible *= u32(dot(view.frustum[4], center) > -radius);
    visible *= u32(dot(view.frustum[5], center) > -radius);

    return visible * u32(view.culling_enabled) + u32(1u - u32(view.culling_enabled));
}

@compute @workgroup_size(64)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let meshlet_instance_index = gid.x;
    if (meshlet_instance_index >= params.meshlet_count || meshlet_instance_index >= arrayLength(&meshlet_instances)) {
        return;
    }

    let meshlet_instance = meshlet_instances[meshlet_instance_index];
    let object_instance_index = meshlet_instance.object_instance_index;
    let global_meshlet_index = meshlet_instance.meshlet_index;
    if (object_instance_index >= arrayLength(&object_instances) || global_meshlet_index >= arrayLength(&meshlets)) {
        return;
    }

    let row = object_instances[object_instance_index].row;
    let entity_row = get_entity_row(row);
    if (entity_row >= arrayLength(&entity_index_lookup)) {
        return;
    }

    let entity_resolved = entity_index_lookup[entity_row];
    if (entity_resolved == INVALID_IDX || entity_resolved >= arrayLength(&entity_transforms)) {
        return;
    }

    let transform = entity_transforms[entity_resolved].transform;
    let scale = transform_max_scale(transform);
    var view = view_buffer[params.view_index];
    let meshlet = meshlets[global_meshlet_index];
    let center_world = transform * vec4<f32>(meshlet.center_radius.xyz, 1.0);
    let radius_world = meshlet.center_radius.w * scale * 1.2;

    if (is_in_frustum(center_world, radius_world, &view) == 0u) {
        return;
    }

    let append_index = atomicAdd(&out_draw_command[0].instance_count, 1u);
    out_visible_meshlets[append_index] = vec4<u32>(object_instance_index, global_meshlet_index, 0u, 0u);
}
