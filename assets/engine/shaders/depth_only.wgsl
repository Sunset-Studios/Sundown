#include "common.wgsl"
#include "lighting_common.wgsl"

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
#ifndef CUSTOM_VS
fn vertex(v_out: ptr<function, VertexOutput>) -> VertexOutput {
    return *v_out;
}
#endif

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
    output.uv = instance_vertex.uv;
    output.instance_id = entity_resolved;
    output.world_position = world_position;

    output = vertex(&output);

#ifndef FINAL_POSITION_WRITE
    output.position = view_proj_mat * output.world_position;
#endif

    return output;
}

@fragment
fn fs(input: VertexOutput) {
#if MASKED
    let mask = fragment_mask(input);
    if (mask <= 0.0) {
        discard;
    } 
#endif
}