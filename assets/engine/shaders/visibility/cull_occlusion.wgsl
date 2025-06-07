#include "common.wgsl"

// ------------------------------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------------------------------ 

struct DrawCullData {
    draw_count: u32,
    hzb_width: u32,
    hzb_height: u32,
}

// ------------------------------------------------------------------------------------
// Buffers
// ------------------------------------------------------------------------------------ 

@group(1) @binding(0) var input_texture: texture_2d<f32>;
@group(1) @binding(1) var<storage, read> aabb_bounds: array<AABBNodeBounds>;
@group(1) @binding(2) var<storage, read> visible_object_instances_no_occlusion: array<i32>;
@group(1) @binding(3) var<storage, read_write> visible_object_instances: array<i32>;
@group(1) @binding(4) var<storage, read> object_instances: array<ObjectInstance>;
@group(1) @binding(5) var<storage, read> entity_aabb_node_indices: array<u32>;
@group(1) @binding(6) var<uniform> draw_cull_data: DrawCullData;
@group(1) @binding(7) var<storage, read_write> draw_indirect_buffer: array<DrawCommand>;


// ------------------------------------------------------------------------------------
// Occlusion Helper Functions
// ------------------------------------------------------------------------------------ 

fn sphere_project(center: vec4<f32>, radius: f32, aabb: ptr<function, vec4<f32>>, view: ptr<function, View>) -> bool {
    // Transform the sphere center to view space.
    let center_view = view.view_matrix * center;
    // Compute a positive depth assuming the camera looks along -z.
    let depth = -center_view.z;
    // If the sphere is too close to the camera, skip occlusion.
    if (depth <= radius) {
        return false;
    }

    // Get the projection matrix plane from the view
    let p00 = view.projection_matrix[0][0];
    let p11 = view.projection_matrix[1][1];
    
    let inv_depth = 1.0 / depth;
    // Project the center into normalized device coordinates (NDC).
    let ndc_center = vec2<f32>(center_view.x, center_view.y) * inv_depth * vec2<f32>(p00, p11);
    // Compute the projected radius in NDC for each axis.
    let ndc_radius = vec2<f32>(radius * p00, radius * p11) * inv_depth;
    
    // Calculate the NDC bounding box.
    let ndc_min = ndc_center - ndc_radius;
    let ndc_max = ndc_center + ndc_radius;
    
    // Convert the NDC bounding box to texture coordinates [0, 1].
    let tex_left = 0.5 * (ndc_min.x + 1.0);
    let tex_right = 0.5 * (ndc_max.x + 1.0);
    let tex_top = 0.5 * (1.0 - ndc_max.y);
    let tex_bottom = 0.5 * (1.0 - ndc_min.y);
    
    *aabb = vec4<f32>(tex_left, tex_top, tex_right, tex_bottom);
    return true;
}

fn is_occluded(center: vec4<f32>, radius: f32, view: ptr<function, View>) -> u32 {
    if (view.occlusion_enabled == 0.0) {
        return 0u;
    }
    
    var aabb: vec4<f32>;
    if (!sphere_project(center, radius, &aabb, view)) {
        return 0u;
    }
    
    let width = (aabb.z - aabb.x) * f32(draw_cull_data.hzb_width);
    let height = (aabb.w - aabb.y) * f32(draw_cull_data.hzb_height);
    let non_negative_size = max(width, height);
    let level = max(floor(log2(non_negative_size)), 0.0);
    let uv = (aabb.xy + aabb.zw) * 0.5;
    
    // Sample occluder depth from the hierarchical Z-buffer.
    var depth = textureSampleLevel(input_texture, non_filtering_sampler, uv, level).r;
    depth = max(depth, textureSampleLevel(input_texture, non_filtering_sampler, vec2(aabb.x, aabb.y), level).r);
    depth = max(depth, textureSampleLevel(input_texture, non_filtering_sampler, vec2(aabb.z, aabb.w), level).r);
    depth = max(depth, textureSampleLevel(input_texture, non_filtering_sampler, vec2(aabb.x, aabb.w), level).r);
    depth = max(depth, textureSampleLevel(input_texture, non_filtering_sampler, vec2(aabb.z, aabb.y), level).r);
    
    //let view_direction = normalize(view.view_position.xyz - center.xyz);
    let view_direction = -view.view_direction.xyz;
    let adjusted_center = center.xyz + (view_direction * radius * 1.5);
    let screen_space_center = view.view_matrix * vec4f(adjusted_center, 1.0);
    let sphere_depth = -screen_space_center.z;

    let lin_depth = linearize_depth(depth, view.near, view.far);
    
    let visible = u32(sphere_depth < lin_depth);
    return (1u - visible) * u32(view.culling_enabled);
}

// ------------------------------------------------------------------------------------
// Compute Shader
// ------------------------------------------------------------------------------------ 

@compute @workgroup_size(256)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let g_id = global_id.x;
    if (g_id >= u32(draw_cull_data.draw_count)) {
        return;
    }

    let object_instance_index = visible_object_instances_no_occlusion[g_id];
    if (object_instance_index == -1) {
        return;
    }

    let object_instance = object_instances[object_instance_index];
    let entity_resolved = get_entity_row(object_instance.row);

    let aabb_node_index = entity_aabb_node_indices[entity_resolved];
    let aabb_node = aabb_bounds[aabb_node_index];

    let center = vec4f((aabb_node.min_point.xyz + aabb_node.max_point.xyz) * 0.5, 1.0);
    var radius = length(aabb_node.max_point.xyz - aabb_node.min_point.xyz) * 0.5;
    radius *= 1.2; // Inflate bounds conservatively

    var view = view_buffer[frame_info.view_index];
    let occluded = is_occluded(center, radius, &view);

    if (occluded > 0u) {
        return;
    }

    let batch_index = object_instance.batch;
    let first_instance = draw_indirect_buffer[batch_index].first_instance;
    let count_index = atomicAdd(&draw_indirect_buffer[batch_index].instance_count, 1u);
    let instance_index = first_instance + count_index;
    visible_object_instances[instance_index] = object_instance_index;
}
