#include "common.wgsl"

// ------------------------------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------------------------------ 
struct VertexOutput {
    @builtin(position) @invariant position: vec4f,
};

// ------------------------------------------------------------------------------------
// Buffers
// ------------------------------------------------------------------------------------ 

@group(1) @binding(0) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(1) var<storage, read> entity_flags: array<u32>;
@group(1) @binding(2) var<storage, read> object_instances: array<ObjectInstance>;
@group(1) @binding(3) var<storage, read> visible_object_instances_no_occlusion: array<i32>;

// ------------------------------------------------------------------------------------
// Vertex Shader
// ------------------------------------------------------------------------------------ 
@vertex fn vs(
    @builtin(vertex_index) vi : u32,
    @builtin(instance_index) ii: u32
) -> VertexOutput {
    let instance_vertex         = vertex_buffer[vi];
    let object_instance_index   = visible_object_instances_no_occlusion[ii];
    let entity_resolved         = get_entity_row(object_instances[object_instance_index].row);
    let transform               = entity_transforms[entity_resolved].transform;
    let view_index              = frame_info.view_index;
    let view_proj_mat           = view_buffer[view_index].view_projection_matrix;

    let world_position = select(
        transform * vec4<f32>(instance_vertex.position),
        billboard_vertex_local(
            instance_vertex.uv,
            transform
        ),
        (entity_flags[entity_resolved] & EF_BILLBOARD) != 0
    );

    var output : VertexOutput;
    output.position = view_proj_mat * world_position;
    return output;
}