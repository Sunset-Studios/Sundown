#include "common.wgsl"
#include "acceleration_common.wgsl"

// ------------------------------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------------------------------ 

struct DrawCullData {
    draw_count: u32,
    view_index: u32,
    clipmap_index: u32,
}

// ------------------------------------------------------------------------------------
// Buffers
// ------------------------------------------------------------------------------------ 

@group(1) @binding(0) var<storage, read> aabb_bounds: array<AABB>;
@group(1) @binding(1) var<storage, read> object_instances: array<ObjectInstance>;
@group(1) @binding(2) var<storage, read_write> visible_object_instances: array<i32>;
@group(1) @binding(3) var<storage, read_write> draw_indirect_buffer: array<DrawCommand>;
@group(1) @binding(4) var<uniform> draw_cull_data: DrawCullData;
@group(1) @binding(5) var<storage, read> entity_index_lookup: array<u32>;


// ------------------------------------------------------------------------------------
// Occlusion Helper Functions
// ------------------------------------------------------------------------------------ 

fn is_in_frustum(center: vec4<f32>, radius: f32, view: ptr<function, View>) -> u32 {
    var visible = 1u;

    var clip_scaling = f32(1u << draw_cull_data.clipmap_index);

    var new_frustum = view.frustum;
    new_frustum[0] = vec4<f32>(view.frustum[0].xyz / clip_scaling, view.frustum[0].w / clip_scaling);
    new_frustum[1] = vec4<f32>(view.frustum[1].xyz / clip_scaling, view.frustum[1].w / clip_scaling);
    new_frustum[2] = vec4<f32>(view.frustum[2].xyz / clip_scaling, view.frustum[2].w / clip_scaling);
    new_frustum[3] = vec4<f32>(view.frustum[3].xyz / clip_scaling, view.frustum[3].w / clip_scaling);
    new_frustum[4] = vec4<f32>(view.frustum[4].xyz / clip_scaling, view.frustum[4].w / clip_scaling);
    new_frustum[5] = vec4<f32>(view.frustum[5].xyz / clip_scaling, view.frustum[5].w / clip_scaling);

    // Check all frustum planes
    visible *= u32(dot(new_frustum[0], center) > -radius);
    visible *= u32(dot(new_frustum[1], center) > -radius);
    visible *= u32(dot(new_frustum[2], center) > -radius);
    visible *= u32(dot(new_frustum[3], center) > -radius);
    visible *= u32(dot(new_frustum[4], center) > -radius);
    visible *= u32(dot(new_frustum[5], center) > -radius);

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
    let entity_row = get_entity_row(row);
    if (entity_row >= arrayLength(&entity_index_lookup)) {
        return;
    }

    let entity_resolved = entity_index_lookup[entity_row];
    if (entity_resolved == 0xffffffffu || entity_resolved >= arrayLength(&aabb_bounds)) {
        return;
    }

    let aabb_node = aabb_bounds[entity_resolved];
    let center = vec4<f32>((aabb_node.min.xyz + aabb_node.max.xyz) * 0.5, 1.0);
    var radius = length(aabb_node.max.xyz - aabb_node.min.xyz) * 0.5;
    radius *= 1.2; // Inflate bounds conservatively

    var view = view_buffer[draw_cull_data.view_index];
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
