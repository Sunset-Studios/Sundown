#include "common.wgsl"

// ------------------------------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------------------------------ 

struct DrawCommand {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    vertex_offset: i32,
    first_instance: u32,
}

struct DrawCullConstants {
    draw_count: f32,
    p00: f32,
    p11: f32,
    hzb_width: f32,
    hzb_height: f32,
}

// ------------------------------------------------------------------------------------
// Buffers
// ------------------------------------------------------------------------------------ 

@group(1) @binding(0) var input_texture: texture_2d<f32>;
@group(1) @binding(1) var<storage, read> aabb_bounds: array<AABBNodeBounds>;
@group(1) @binding(2) var<storage, read> object_instances: array<ObjectInstance>;
@group(1) @binding(3) var<storage, read_write> compacted_object_instances: array<CompactedObjectInstance>;
@group(1) @binding(4) var<storage, read_write> draw_indirect_buffer: array<DrawCommand>;
@group(1) @binding(5) var<storage, read> entity_aabb_node_indices: array<u32>;
@group(1) @binding(6) var<uniform> draw_cull_constants: DrawCullConstants;

// ------------------------------------------------------------------------------------
// Occlusion Helper Functions
// ------------------------------------------------------------------------------------ 

fn sphere_project(center: vec4<f32>, radius: f32, p00: f32, p11: f32, aabb: ptr<function, vec4<f32>>, view: ptr<function, View>) -> bool {
    // Transform center to view space and skip occlusion if inside bounding sphere
    var center_view = view.view_matrix * center;
    let dist2 = dot(center_view.xyz, center_view.xyz);
    if (dist2 <= radius * radius) {
        // viewer is inside the sphere, always visible
        return false;
    }

    let cx = vec2f(center_view.x, -center_view.z);
    let vx = vec2f(sqrt(max(dot(cx, cx) - radius * radius, 0.0)), radius);
    let min_x = mat2x2f(vx.x, vx.y, -vx.y, vx.x) * cx;
    let max_x = mat2x2f(vx.x, -vx.y, vx.y, vx.x) * cx;

    let cy = vec2f(center_view.y, -center_view.z);
    let vy = vec2f(sqrt(max(dot(cy, cy) - radius * radius, 0.0)), radius);
    let min_y = mat2x2f(vy.x, vy.y, -vy.y, vy.x) * cy;
    let max_y = mat2x2f(vy.x, -vy.y, vy.y, vy.x) * cy;

    // Compute AABB in NDC ([-1,1] range)
    *aabb = vec4f(
        (select(0.0, min_x.x / min_x.y, min_x.y != 0.0)) * p00,
        -(select(0.0, min_y.x / min_y.y, min_y.y != 0.0)) * p11,
        (select(0.0, max_x.x / max_x.y, max_x.y != 0.0)) * p00,
        -(select(0.0, max_y.x / max_y.y, max_y.y != 0.0)) * p11
    );

    // Map from NDC [-1,1] to UV [0,1]
    *aabb = (*aabb * 0.5) + 0.5;
    *aabb = clamp(*aabb, vec4f(0.0), vec4f(1.0));

    return true;
}

fn is_occluded(center: vec4<f32>, radius: f32, view: ptr<function, View>) -> u32 {
    if (view.occlusion_enabled == 0.0) {
        return 0u;
    }

    var aabb: vec4<f32>;
    if (!sphere_project(center, radius, draw_cull_constants.p00, draw_cull_constants.p11, &aabb, view)) {
        return 0u;
    }

    let width = (aabb.z - aabb.x) * draw_cull_constants.hzb_width;
    let height = (aabb.w - aabb.y) * draw_cull_constants.hzb_height;
    let non_negative_size = max(1.0, max(width, height));
    let level = max(floor(log2(non_negative_size)), 0.0);

    let uv = (aabb.xy + aabb.zw) * 0.5;

    var depth = textureSampleLevel(input_texture, non_filtering_sampler, uv, level).r;
    depth = max(depth, textureSampleLevel(input_texture, non_filtering_sampler, vec2(aabb.x, aabb.y), level).r);
    depth = max(depth, textureSampleLevel(input_texture, non_filtering_sampler, vec2(aabb.z, aabb.w), level).r);
    depth = max(depth, textureSampleLevel(input_texture, non_filtering_sampler, vec2(aabb.x, aabb.w), level).r);
    depth = max(depth, textureSampleLevel(input_texture, non_filtering_sampler, vec2(aabb.z, aabb.y), level).r);

    let view_matrix = view.view_matrix;
    let projection_matrix = view.projection_matrix;

    let closest_point_ws = center.xyz - (view.view_direction.xyz * radius);
    let view_space_point = (view_matrix * vec4<f32>(closest_point_ws, 1.0));

    let proj_point = projection_matrix * view_space_point;
    var sphere_depth = proj_point.z / proj_point.w; 

    let dist_bias_val = abs(view_space_point.z) * 0.000001;
    let final_bias = max(0.000001, dist_bias_val);

    let visible_u32 = u32(sphere_depth < depth + final_bias);
    return (1u - visible_u32) * u32(view.culling_enabled);
}

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

// ------------------------------------------------------------------------------------
// Compute Shader
// ------------------------------------------------------------------------------------ 

@compute @workgroup_size(256)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let g_id = global_id.x;
    if (g_id < u32(draw_cull_constants.draw_count)) {
        let row = object_instances[g_id].row;
        let entity_resolved = get_entity_row(row);
        let aabb_node_index = entity_aabb_node_indices[entity_resolved];

        let aabb_node = aabb_bounds[aabb_node_index];
        let center = vec4f((aabb_node.min_point.xyz + aabb_node.max_point.xyz) * 0.5, 1.0);
        var radius = length(aabb_node.max_point.xyz - aabb_node.min_point.xyz) * 0.5;
        radius *= 1.05; // Inflate bounds conservatively

        var view = view_buffer[frame_info.view_index];
        let in_frustum = is_in_frustum(center, radius, &view);
        let occluded = is_occluded(center, radius, &view);

        if ((in_frustum * (1u - occluded)) > 0u) {
            let batch_index = object_instances[g_id].batch;
            let first_instance = draw_indirect_buffer[batch_index].first_instance;
            let count_index = atomicAdd(&draw_indirect_buffer[batch_index].instance_count, 1u);
            let instance_index = first_instance + count_index;
            compacted_object_instances[instance_index].row = row;
        }
    }
}
