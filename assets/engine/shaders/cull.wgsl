#include "common.wgsl"

struct DrawCommand {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    vertex_offset: i32,
    first_instance: u32,
}

struct DrawCullConstants {
    draw_count: f32,
    culling_enabled: f32,
    occlusion_enabled: f32,
    distance_check: f32,
    z_near: f32,
    z_far: f32,
    p00: f32,
    p11: f32,
    hzb_width: f32,
    hzb_height: f32,
}

@group(1) @binding(0) var input_texture: texture_2d<f32>;
@group(1) @binding(1) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(2) var<storage, read> object_instances: array<ObjectInstance>;
@group(1) @binding(3) var<storage, read_write> compacted_object_instances: array<CompactedObjectInstance>;
@group(1) @binding(4) var<storage, read_write> draw_indirect_buffer: array<DrawCommand>;
@group(1) @binding(5) var<uniform> draw_cull_constants: DrawCullConstants;

// 2D Polyhedral Bounds of a Clipped, Perspective-Projected 3D Sphere. Michael Mara, Morgan McGuire. 2013
fn sphere_project(center: vec4<f32>, radius: f32, p00: f32, p11: f32, aabb: ptr<function, vec4<f32>>) -> bool {
    var center_view = view_buffer[0].view_matrix * center;

	let cx = vec2f(center_view.x, -center_view.z);
	let vx = vec2f(sqrt(dot(cx, cx) - radius * radius), radius);
	let min_x = mat2x2f(vx.x, vx.y, -vx.y, vx.x) * cx;
	let max_x = mat2x2f(vx.x, -vx.y, vx.y ,vx.x) * cx;

	let cy = vec2f(center_view.y, -center_view.z);
	let vy = vec2f(sqrt(dot(cy, cy) - radius * radius), radius);
	let min_y = mat2x2f(vy.x, vy.y, -vy.y, vy.x) * cy;
	let max_y = mat2x2f(vy.x, -vy.y, vy.y, vy.x) * cy;

	*aabb = vec4f(
        min_x.x / min_x.y * draw_cull_constants.p00, 
        min_y.x / min_y.y * draw_cull_constants.p11, 
        max_x.x / max_x.y * draw_cull_constants.p00, 
        max_y.x / max_y.y * draw_cull_constants.p11
    );
	*aabb = aabb.xwzy * vec4f(0.5, -0.5, 0.5, -0.5) + vec4f(0.5);

	return true;
}

fn is_occluded(center: vec4<f32>, radius: f32) -> u32 {
    var world_center = center;

    if (draw_cull_constants.occlusion_enabled == 0.0) {
        return 0u;
    }

    var aabb: vec4<f32>;
    if (!sphere_project(world_center, radius, draw_cull_constants.p00, draw_cull_constants.p11, &aabb)) {
        return 0u;
    }

    let width = (aabb.z - aabb.x) * draw_cull_constants.hzb_width;
    let height = (aabb.w - aabb.y) * draw_cull_constants.hzb_height;
    let level = floor(log2(max(width, height)));

    let uv = (aabb.xy + aabb.zw) * 0.5;

    // Sample HZB at the coarsest level that fully contains the AABB
    var depth = textureSampleLevel(input_texture, non_filtering_sampler, uv, level).r;
    depth = max(depth, textureSampleLevel(input_texture, non_filtering_sampler, vec2(aabb.x, aabb.y), level).r);
    depth = max(depth, textureSampleLevel(input_texture, non_filtering_sampler, vec2(aabb.z, aabb.w), level).r);
    depth = max(depth, textureSampleLevel(input_texture, non_filtering_sampler, vec2(aabb.x, aabb.w), level).r);
    depth = max(depth, textureSampleLevel(input_texture, non_filtering_sampler, vec2(aabb.z, aabb.y), level).r);

    // Calculate sphere's depth using full view-projection matrix
    let view_proj_matrix = view_buffer[0].view_projection_matrix;
    let sphere_front_point = world_center + (vec4<f32>(normalize(world_center.xyz) * radius, 0.0) - world_center);
    let projected_front = view_proj_matrix * sphere_front_point;
    let sphere_depth = projected_front.z / projected_front.w;

    // if the depth of the sphere is in front of the depth pyramid value, then the object is visible
    let bias = 0.01;
    let visible = u32(sphere_depth < depth + bias);


    return (1u - visible) * u32(draw_cull_constants.culling_enabled);
}

fn is_in_frustum(center: vec4<f32>, radius: f32) -> u32 {
    var visible = 1u;

    // [left, right, bottom, top, near, far] checks
    visible *= u32(dot(view_buffer[0].frustum[0], center) > -radius);
    visible *= u32(dot(view_buffer[0].frustum[1], center) > -radius);
    visible *= u32(dot(view_buffer[0].frustum[2], center) > -radius);
    visible *= u32(dot(view_buffer[0].frustum[3], center) > -radius);
    visible *= u32(dot(view_buffer[0].frustum[4], center) > -radius);
    visible *= u32(dot(view_buffer[0].frustum[5], center) > -radius);

    return visible * u32(draw_cull_constants.culling_enabled) + u32(1 - draw_cull_constants.culling_enabled);
}

@compute @workgroup_size(256)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let g_id = global_id.x;
    if (g_id < u32(draw_cull_constants.draw_count)) {
        let entity_id = object_instances[g_id].entity;

        let sphere_bounds = entity_transforms[entity_id].bounds_pos_radius;
        let center = vec4<f32>(sphere_bounds.xyz, 1.0);

        // artificially inflate bounds to be more conservative with culling and to prevent
        // "z-fighting" style cull between bounds and written depth values
        let radius = sphere_bounds.w * entity_transforms[entity_id].bounds_extent_and_custom_scale.w;

        let in_frustum = is_in_frustum(center, radius);
        let occluded = is_occluded(center, radius);
        if ((in_frustum * (1u - occluded)) > 0u) {
            let batch_index = object_instances[g_id].batch;
            let count_index = atomicAdd(&draw_indirect_buffer[batch_index].instance_count, 1u);
            let instance_index = draw_indirect_buffer[batch_index].first_instance + count_index;
            compacted_object_instances[instance_index].entity = entity_id;
        }
    }
}