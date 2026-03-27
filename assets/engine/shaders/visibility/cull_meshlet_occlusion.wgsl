#include "common.wgsl"
#include "visibility/visibility_common.wgsl"

struct MeshletOcclusionParams {
    view_index: u32,
    list_capacity: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(1) @binding(0) var input_texture: texture_2d<f32>;
@group(1) @binding(1) var<storage, read> in_visible_meshlets: array<vec4<u32>>;
@group(1) @binding(2) var<storage, read_write> in_draw_command: array<MeshletDrawCommand>;
@group(1) @binding(3) var<storage, read_write> out_visible_meshlets: array<vec4<u32>>;
@group(1) @binding(4) var<storage, read_write> out_draw_command: array<MeshletDrawCommand>;
@group(1) @binding(5) var<storage, read> object_instances: array<ObjectInstance>;
@group(1) @binding(6) var<storage, read> entity_index_lookup: array<u32>;
@group(1) @binding(7) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(8) var<storage, read> meshlets: array<MeshletRecord>;
@group(1) @binding(9) var<uniform> params: MeshletOcclusionParams;

fn project_sphere_rect(
    center_world: vec3<f32>,
    radius: f32,
    view: ptr<function, View>,
    uv_rect: ptr<function, vec4<f32>>,
    sphere_depth: ptr<function, f32>
) -> bool {
    let center_view = (*view).view_matrix * vec4<f32>(center_world, 1.0);
    if (-center_view.z <= radius) {
        return false;
    }

    let clip_center = (*view).projection_matrix * center_view;
    let clip_right = (*view).projection_matrix * vec4<f32>(center_view.x + radius, center_view.y, center_view.z, 1.0);
    let clip_up = (*view).projection_matrix * vec4<f32>(center_view.x, center_view.y + radius, center_view.z, 1.0);

    let ndc_center = clip_center.xy / clip_center.w;
    let ndc_right = clip_right.xy / clip_right.w;
    let ndc_up = clip_up.xy / clip_up.w;

    let radius_ndc = vec2<f32>(
        abs(ndc_right.x - ndc_center.x),
        abs(ndc_up.y - ndc_center.y)
    );

    let ndc_min = ndc_center - radius_ndc;
    let ndc_max = ndc_center + radius_ndc;
    if (ndc_max.x < -1.0 || ndc_min.x > 1.0 || ndc_max.y < -1.0 || ndc_min.y > 1.0) {
        return false;
    }

    let u_min = clamp(ndc_min.x * 0.5 + 0.5, 0.0, 1.0);
    let u_max = clamp(ndc_max.x * 0.5 + 0.5, 0.0, 1.0);
    let v_min = clamp(-ndc_max.y * 0.5 + 0.5, 0.0, 1.0);
    let v_max = clamp(-ndc_min.y * 0.5 + 0.5, 0.0, 1.0);

    if (u_max <= u_min || v_max <= v_min) {
        return false;
    }

    *uv_rect = vec4<f32>(u_min, v_min, u_max, v_max);
    *sphere_depth = -center_view.z - radius;
    return true;
}

fn is_occluded(center_world: vec3<f32>, radius: f32, view: ptr<function, View>) -> bool {
    if ((*view).occlusion_enabled == 0.0) {
        return false;
    }

    var uv_rect: vec4<f32>;
    var sphere_depth = 0.0;
    if (!project_sphere_rect(center_world, radius, view, &uv_rect, &sphere_depth)) {
        return false;
    }

    let hzb_dims = textureDimensions(input_texture);
    let width = (uv_rect.z - uv_rect.x) * f32(hzb_dims.x);
    let height = (uv_rect.w - uv_rect.y) * f32(hzb_dims.y);
    let level_floor = floor(log2(max(max(width, height), 1.0)));
    let level = clamp(level_floor, 0.0, f32(textureNumLevels(input_texture) - 1u));

    var max_depth = 0.0;
    for (var ix = 0u; ix < 3u; ix = ix + 1u) {
        for (var iy = 0u; iy < 3u; iy = iy + 1u) {
            let uv = vec2<f32>(
                mix(uv_rect.x, uv_rect.z, f32(ix) * 0.5),
                mix(uv_rect.y, uv_rect.w, f32(iy) * 0.5)
            );
            let raw_depth = textureSampleLevel(input_texture, non_filtering_sampler, uv, level).r;
            max_depth = max(max_depth, linearize_depth(raw_depth, (*view).near, (*view).far, params.view_index));
        }
    }

    return sphere_depth >= max_depth + 1.0;
}

@compute @workgroup_size(64)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let meshlet_list_index = global_id.x;
    let visible_count = atomicLoad(&in_draw_command[0].instance_count);
    if (meshlet_list_index >= visible_count || meshlet_list_index >= params.list_capacity) {
        return;
    }

    let visible_entry = in_visible_meshlets[meshlet_list_index];
    let object_instance_index = meshlet_object_index(visible_entry);
    let global_meshlet_index = meshlet_index(visible_entry);

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
    let meshlet = meshlets[global_meshlet_index];
    let center_world = (transform * vec4<f32>(meshlet.center_radius.xyz, 1.0)).xyz;
    let radius_world = meshlet.center_radius.w * transform_max_scale(transform);

    var view = view_buffer[params.view_index];
    if (is_occluded(center_world, radius_world, &view)) {
        return;
    }

    let append_index = atomicAdd(&out_draw_command[0].instance_count, 1u);

    out_visible_meshlets[append_index] = visible_entry;
}
