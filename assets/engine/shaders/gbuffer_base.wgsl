#include "common.wgsl"

struct VertexOutput {
    @builtin(position) @invariant position: vec4f,
    @location(0) world_position: vec4f,
    @location(1) color: vec4f,
    @location(2) uv: vec2f,
    @location(3) normal: vec4f,
    @location(4) tangent: vec4f,
    @location(5) bitangent: vec4f,
    @location(6) @interpolate(flat) instance_id: u32,
};

struct FragmentOutput {
    @location(0) albedo: vec4f,
    @location(1) smra: vec4f,
    @location(2) position: vec4f,
    @location(3) normal: vec4f,
}

@group(1) @binding(0) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(1) var<storage, read> compacted_object_instances: array<CompactedObjectInstance>;

#ifndef CUSTOM_VS
fn vertex(v_out: VertexOutput) -> VertexOutput {
    return v_out;
}
#endif

@vertex fn vs(
    @builtin(vertex_index) vi : u32,
    @builtin(instance_index) ii: u32
) -> VertexOutput {
    let entity = compacted_object_instances[ii].entity;
    let entity_transform = entity_transforms[entity];
    let instance_vertex = vertex_buffer[vi];

    let model_matrix = entity_transform.transform;
    let inverse_model_matrix = entity_transform.inverse_model_matrix;
    let mvp = view_buffer[0].view_projection_matrix * model_matrix;

	var transpose_inverse_model_matrix = transpose(inverse_model_matrix);
    var normal_matrix = mat3x3f(
        transpose_inverse_model_matrix[0].xyz,
        transpose_inverse_model_matrix[1].xyz,
        transpose_inverse_model_matrix[2].xyz
    );

    var output : VertexOutput;

    output.position = mvp * instance_vertex.position;
    output.world_position = model_matrix * instance_vertex.position;
    output.color = instance_vertex.color;
    output.uv = instance_vertex.uv;
    output.normal = normalize(vec4f(normal_matrix * instance_vertex.normal.rgb, 1.0));
    output.tangent = model_matrix * instance_vertex.tangent;
    output.bitangent = model_matrix * instance_vertex.bitangent;
    output.instance_id = entity;

    return vertex(output);
}

#ifndef CUSTOM_FS
fn fragment(v_out: VertexOutput, f_out: FragmentOutput) -> FragmentOutput {
    return f_out;
}
#endif

@fragment fn fs(v_out: VertexOutput) -> FragmentOutput {
    var output : FragmentOutput;

    output.position = v_out.world_position;
    output.normal = v_out.normal;

    return fragment(v_out, output);
}