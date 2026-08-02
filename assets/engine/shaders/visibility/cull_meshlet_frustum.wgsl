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
@group(1) @binding(4) var<storage, read> entity_flags: array<u32>;
@group(1) @binding(5) var<storage, read> meshlets: array<MeshletRecord>;
@group(1) @binding(6) var<uniform> params: MeshletCullParams;
@group(1) @binding(7) var<storage, read_write> out_visible_meshlets: array<vec4<u32>>;
@group(1) @binding(8) var<storage, read_write> out_draw_command: array<MeshletDrawCommand>;

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

fn is_meshlet_backfacing(
    meshlet: MeshletRecord,
    object_instance: ObjectInstance,
    entity_resolved: u32,
    center_world: vec3<f32>,
    radius_world: f32,
    view: ptr<function, View>
) -> bool {
    if ((*view).culling_enabled == 0.0 ||
        (object_instance.flags & OIF_DOUBLE_SIDED) != 0u ||
        (entity_flags[entity_resolved] & EF_BILLBOARD) != 0u) {
        return false;
    }

    let transform = entity_transforms[entity_resolved].transform;
    let scale_x = length(transform[0].xyz);
    let scale_y = length(transform[1].xyz);
    let scale_z = length(transform[2].xyz);
    let max_scale = max(max(scale_x, scale_y), scale_z);
    let min_scale = min(min(scale_x, scale_y), scale_z);
    if (max_scale <= MESHLET_EPSILON || max_scale - min_scale > max_scale * 0.001) {
        return false;
    }

    let orientation = determinant(mat3x3<f32>(
        transform[0].xyz,
        transform[1].xyz,
        transform[2].xyz
    ));
    if (orientation <= 0.0) {
        return false;
    }

    let camera_to_center = center_world - (*view).view_position.xyz;
    let camera_distance = length(camera_to_center);
    if (camera_distance <= radius_world + MESHLET_EPSILON) {
        return false;
    }

    let cone_axis_world = safe_normalize(
        (entity_transforms[entity_resolved].transpose_inverse_model_matrix *
            vec4<f32>(meshlet.normal_cone.xyz, 0.0)).xyz
    );
    // meshoptimizer's sphere form avoids storing a separate cone apex. The radius
    // term makes the rejection conservative while still removing the entire draw.
    let conservative_cutoff = meshlet.normal_cone.w + radius_world / camera_distance;
    return dot(camera_to_center / camera_distance, cone_axis_world) >= conservative_cutoff;
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
    if (entity_resolved == INVALID_IDX ||
        entity_resolved >= arrayLength(&entity_transforms) ||
        entity_resolved >= arrayLength(&entity_flags)) {
        return;
    }

    let object_instance = object_instances[object_instance_index];
    let transform = entity_transforms[entity_resolved].transform;
    let scale = transform_max_scale(transform);
    var view = view_buffer[params.view_index];
    let meshlet = meshlets[global_meshlet_index];
    let center_world = transform * vec4<f32>(meshlet.center_radius.xyz, 1.0);
    let radius_world = meshlet.center_radius.w * scale * 1.2;

    if (is_in_frustum(center_world, radius_world, &view) == 0u) {
        return;
    }

    if (is_meshlet_backfacing(
        meshlet,
        object_instance,
        entity_resolved,
        center_world.xyz,
        radius_world,
        &view
    )) {
        return;
    }

    let append_index = atomicAdd(&out_draw_command[0].instance_count, 1u);
    out_visible_meshlets[append_index] = vec4<u32>(object_instance_index, global_meshlet_index, 0u, 0u);
}
