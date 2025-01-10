#include "common.wgsl"

// ------------------------------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------------------------------ 

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(flat) entity_id: u32,
    @location(1) @interpolate(flat) base_entity_id: u32,
};

struct FragmentOutput {
    @location(0) entity_id: vec2<u32>,
}

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
    let entity = compacted_object_instances[ii].entity;
    let entity_resolved = entity_metadata[entity].offset + compacted_object_instances[ii].entity_instance;

    let model_matrix = entity_transforms[entity_resolved].transform;
    let mvp = view_buffer[0].view_projection_matrix * model_matrix;

    var output : VertexOutput;

    output.position = mvp * vertex_buffer[vi].position;

    output.base_entity_id = entity;
    output.entity_id = entity_resolved;

    return output;
}

// ------------------------------------------------------------------------------------
// Fragment Shader
// ------------------------------------------------------------------------------------ 

@fragment fn fs(v_out: VertexOutput) -> FragmentOutput {
    var output : FragmentOutput;

    output.entity_id = vec2<u32>(v_out.base_entity_id, v_out.entity_id);

    return output;
}