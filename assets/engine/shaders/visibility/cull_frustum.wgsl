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

@group(1) @binding(0) var<storage, read> aabb_bounds: array<AABBNodeBounds>;
@group(1) @binding(1) var<storage, read> object_instances: array<ObjectInstance>;
@group(1) @binding(2) var<storage, read_write> visible_object_instances: array<i32>;
@group(1) @binding(3) var<storage, read_write> draw_indirect_buffer: array<DrawCommand>;
@group(1) @binding(4) var<storage, read> entity_aabb_node_indices: array<u32>;
@group(1) @binding(5) var<uniform> draw_cull_data: DrawCullData;


// ------------------------------------------------------------------------------------
// Occlusion Helper Functions
// ------------------------------------------------------------------------------------ 

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
    if (g_id >= u32(draw_cull_data.draw_count)) {
        return;
    }

    let row = object_instances[g_id].row;
    let entity_resolved = get_entity_row(row);
    let aabb_node_index = entity_aabb_node_indices[entity_resolved];

    let aabb_node = aabb_bounds[aabb_node_index];
    let center = vec4f((aabb_node.min_point.xyz + aabb_node.max_point.xyz) * 0.5, 1.0);
    var radius = length(aabb_node.max_point.xyz - aabb_node.min_point.xyz) * 0.5;
    radius *= 1.1; // Inflate bounds conservatively

    var view = view_buffer[frame_info.view_index];
    let in_frustum = is_in_frustum(center, radius, &view);

    if (in_frustum == 0u) {
        return;
    }

    let batch_index = object_instances[g_id].batch;
    let first_instance = draw_indirect_buffer[batch_index].first_instance;
    let count_index = atomicAdd(&draw_indirect_buffer[batch_index].instance_count, 1u);
    let instance_index = first_instance + count_index;
    visible_object_instances[instance_index] = i32(g_id);
}
