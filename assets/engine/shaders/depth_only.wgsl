#include "common.wgsl"

// ------------------------------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------------------------------ 
struct VertexOutput {
    @builtin(position) position: vec4f,
};

struct FragmentOutput {
    @builtin(frag_depth) depth: f32,
};

// ------------------------------------------------------------------------------------
// Buffers
// ------------------------------------------------------------------------------------ 

@group(1) @binding(0) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(1) var<storage, read> compacted_object_instances: array<CompactedObjectInstance>;

// ------------------------------------------------------------------------------------
// Vertex Shader
// ------------------------------------------------------------------------------------ 
@vertex fn vs(
    @builtin(vertex_index) vi : u32,
    @builtin(instance_index) ii: u32
) -> VertexOutput {
    let instance_vertex = vertex_buffer[vi];
    let entity = compacted_object_instances[ii].entity;
    let entity_resolved = entity_metadata[entity].offset + compacted_object_instances[ii].entity_instance;

    let entity_transform = entity_transforms[entity_resolved];
    let view_proj_mat = view_buffer[0].view_projection_matrix;
    let view_mat = view_buffer[0].view_matrix;

    var output : VertexOutput;
    
    output.position = view_proj_mat * entity_transform.transform * vec4<f32>(instance_vertex.position);
    
    return output;
}

// ------------------------------------------------------------------------------------
// Fragment Shader
// ------------------------------------------------------------------------------------ 
@fragment fn fs(v_out: VertexOutput) -> FragmentOutput {
    var output: FragmentOutput;
    
    output.depth = log_depth(v_out.position);
    
    return output;
}