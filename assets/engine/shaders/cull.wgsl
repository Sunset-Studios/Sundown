#include "common.wgsl"

struct CompactedObjectInstance {
    entity: u32,
}

struct DrawCommand {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    vertex_offset: i32,
    first_instance: u32,
}

struct DrawCullConstants {
    z_near: f32,
    p00: f32,
    p11: f32,
    hzb_width: f32,
    hzb_height: f32,
    draw_count: u32,
    culling_enabled: u32,
    occlusion_enabled: u32,
    distance_check: u32
}

@group(1) @binding(0) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(1) var<storage, read> object_instances: array<ObjectInstance>;
@group(1) @binding(2) var<storage, read_write> compacted_object_instances: array<CompactedObjectInstance>;
@group(1) @binding(3) var<storage, read_write> draw_indirect_buffer: array<DrawCommand>;
@group(1) @binding(4) var<uniform> draw_cull_constants: DrawCullConstants;

// 2D Polyhedral Bounds of a Clipped, Perspective-Projected 3D Sphere. Michael Mara, Morgan McGuire. 2013
fn sphere_project(center: vec4<f32>, radius: f32, p00: f32, p11: f32) -> vec4<f32> {
    var center_view = view_buffer[0].view_matrix * center;

    if (-center_view.z - radius < draw_cull_constants.z_near) {
        return vec4<f32>(0.0);
    }

    let cx = vec2<f32>(center_view.x, -center_view.z);
    let vx = vec2<f32>(sqrt(dot(cx, cx) - radius * radius), radius);
    let min_x = mat2x2<f32>(vx.x, vx.y, -vx.y, vx.x) * cx;
    let max_x = mat2x2<f32>(vx.x, -vx.y, vx.y, vx.x) * cx;

    let cy = -center_view.yz;
    let vy = vec2<f32>(sqrt(dot(cy, cy) - radius * radius), radius);
    let min_y = mat2x2<f32>(vy.x, vy.y, -vy.y, vy.x) * cy;
    let max_y = mat2x2<f32>(vy.x, -vy.y, vy.y, vy.x) * cy;

    var aabb = vec4<f32>(min_x.x / min_x.y * p00, min_y.x / min_y.y * p11, max_x.x / max_x.y * p00, max_y.x / max_y.y * p11);
    aabb = aabb.xwzy * vec4<f32>(0.5, -0.5, 0.5, -0.5) + vec4<f32>(0.5);

    return aabb;
}

fn is_occluded(object_index: u32, center: vec4<f32>, radius: f32) -> bool {
    var visible = 1u;

    if (visible * u32(draw_cull_constants.occlusion_enabled != 0u) != 0u) {
        let aabb = sphere_project(center, radius, draw_cull_constants.p00, draw_cull_constants.p11);
        if (any(aabb != vec4<f32>(0.0))) {
            let width = (aabb.z - aabb.x) * draw_cull_constants.hzb_width;
            let height = (aabb.w - aabb.y) * draw_cull_constants.hzb_height;
            let level = floor(log2(max(width, height)));

            let uv = (aabb.xy + aabb.zw) * 0.5;

            var depth = textureSampleLevel(input_texture, global_sampler, uv, level).r;

            depth = max(depth, textureSampleLevel(input_texture, global_sampler, vec2<f32>(aabb.x, aabb.y), level).r);
            depth = max(depth, textureSampleLevel(input_texture, global_sampler, vec2<f32>(aabb.z, aabb.w), level).r);
            depth = max(depth, textureSampleLevel(input_texture, global_sampler, vec2<f32>(aabb.x, aabb.w), level).r);
            depth = max(depth, textureSampleLevel(input_texture, global_sampler, vec2<f32>(aabb.z, aabb.y), level).r);

            let direction = normalize(view_buffer[0].view_direction.xyz - center.xyz);
            let screen_space_center = view_buffer[0].view_projection_matrix * vec4<f32>(center.xyz + direction * radius, 1.0);
            let sphere_depth = screen_space_center.z / screen_space_center.w;

            visible = visible * u32(sphere_depth <= depth);
        }
    }

    return (1u - visible) * u32(draw_cull_constants.culling_enabled);
}

fn is_in_frustum(object_index: u32, center: vec4<f32>, radius: f32) -> bool {
    var visible = 1u;

    visible = visible * u32(dot(center, view_buffer[0].frustum_planes[0]) > -radius;
    visible = visible * u32(dot(center, view_buffer[0].frustum_planes[1]) > -radius);
    visible = visible * u32(dot(center, view_buffer[0].frustum_planes[2]) > -radius);
    visible = visible * u32(dot(center, view_buffer[0].frustum_planes[3]) > -radius);

    visible = visible * u32(dot(center, view_buffer[0].frustum_planes[4]) > -radius) + u32(draw_cull_constants.distance_check == 0u);
    visible = visible * u32(dot(center, view_buffer[0].frustum_planes[5]) > -radius) + u32(draw_cull_constants.distance_check == 0u);

    return visible + u32(draw_cull_constants.culling_enabled == 0u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let g_id = global_id.x;
    if (g_id < draw_cull_constants.draw_count) {
        let entity_id = object_instances[g_id].entity;

        let sphere_bounds = entity_transforms[entity_id].bounds_pos_radius;
        let center = vec4<f32>(sphere_bounds.xyz, 1.0);

        // artificially inflate bounds to be more conservative with culling and to prevent
        // "z-fighting" style cull between bounds and written depth values
        let radius = sphere_bounds.w * entity_transforms[entity_id].bounds_extent_and_custom_scale.w;

        let in_frustum = is_in_frustum(entity_id, center, radius);
        let occluded = is_occluded(entity_id, center, radius);
        if (in_frustum && !occluded) {
            let batch_index = object_instances[g_id].batch;
            let count_index = atomicAdd(&draw_indirect_buffer[batch_index].instance_count, 1u);
            let instance_index = draw_indirect_buffer[batch_index].first_instance + count_index;
            compacted_object_instances[instance_index].entity = entity_id;
        }
    }
}