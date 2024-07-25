#include "common.wgsl"

struct VertexOutput {
    @builtin(position) position: vec4f,
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

struct EntityTransform {
    model_matrix: mat4x4f,
    inverse_model_matrix: mat4x4f
}

struct ObjectInstance {
    batch: u32,
    entity: u32
}

@group(1) @binding(0) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(1) var<storage, read> object_instances: array<ObjectInstance>;

#ifndef CUSTOM_VS
fn vertex(v_out: VertexOutput) -> VertexOutput {
    return v_out;
}
#endif

@vertex fn vs(
    @builtin(vertex_index) vi : u32,
    @builtin(instance_index) ii: u32
) -> VertexOutput {
    var model_matrix = entity_transforms[object_instances[ii].entity].model_matrix;
    var inverse_model_matrix = entity_transforms[object_instances[ii].entity].inverse_model_matrix;
    var mvp = view_buffer[0].view_projection_matrix * model_matrix;

	var transpose_inverse_model_matrix = transpose(inverse_model_matrix);
    transpose_inverse_model_matrix[3] = vec4f(0.0, 0.0, 0.0, 1.0);

    var output : VertexOutput;

    output.world_position = model_matrix * vertex_buffer[vi].position;
    output.position = mvp * vertex_buffer[vi].position;
    output.color = vertex_buffer[vi].color;
    output.uv = vertex_buffer[vi].uv;
    output.normal = transpose_inverse_model_matrix * vertex_buffer[vi].normal;
    output.tangent = model_matrix * vertex_buffer[vi].tangent;
    output.bitangent = model_matrix * vertex_buffer[vi].bitangent;
    output.instance_id = ii;

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
    output.normal = normalize(vec4f(v_out.normal.xyz, 1.0));

    return fragment(v_out, output);
}