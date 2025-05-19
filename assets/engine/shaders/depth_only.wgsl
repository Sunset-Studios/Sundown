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
@group(1) @binding(2) var<storage, read> compacted_object_instances: array<CompactedObjectInstance>;

// ------------------------------------------------------------------------------------
// Vertex Shader
// ------------------------------------------------------------------------------------ 
@vertex fn vs(
    @builtin(vertex_index) vi : u32,
    @builtin(instance_index) ii: u32
) -> VertexOutput {
    let instance_vertex = vertex_buffer[vi];
    let entity_resolved = get_entity_row(compacted_object_instances[ii].row);
    let transform         = entity_transforms[entity_resolved].transform;
    let view_proj_mat     = view_buffer[0].view_projection_matrix;

    let is_bill    = (entity_flags[entity_resolved] & EF_BILLBOARD) != 0;

    let world_pos = select(
        view_proj_mat * transform * instance_vertex.position,
        view_proj_mat * billboard_vertex_local(
                 instance_vertex.uv,
                 transform
             ),
        is_bill
    );

    var output : VertexOutput;
    output.position = world_pos;
    return output;
}