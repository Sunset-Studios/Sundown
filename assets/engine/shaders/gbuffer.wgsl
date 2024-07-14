#include "common.wgsl"

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
    @location(1) uv: vec2f,
    @location(2) normal: vec4f,
    @location(3) tangent: vec4f,
    @location(4) bitangent: vec4f,
    @location(5) @interpolate(flat) instance_index: u32,
    @location(6) @interpolate(flat) material_index: u32,
};

struct FragmentOutput {
    @location(0) albedo: vec4f,
    @location(1) smra: vec4f,
    @location(2) normal: vec4f,
    @location(3) position: vec4f,
}

struct EntityTransform {
    model_matrix: mat4x4f,
    inverse_model_matrix: mat4x4f
}

struct ObjectInstance {
    batch: u32,
    entity: u32,
    material: u32,
}

@group(1) @binding(0) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(1) var<storage, read> object_instances: array<ObjectInstance>;

@vertex fn vs(
    @builtin(vertex_index) vi : u32,
    @builtin(instance_index) ii: u32
) -> VertexOutput {
    var model_matrix = entity_transforms[object_instances[ii].entity].model_matrix;
    var inverse_model_matrix = entity_transforms[object_instances[ii].entity].inverse_model_matrix;
    var mvp = view_buffer[0].projection_matrix * view_buffer[0].view_matrix * model_matrix;
	var transpose_inverse_model_matrix = transpose(inverse_model_matrix);

	var n = normalize(transpose_inverse_model_matrix * vertex_buffer[vi].normal);
	var t = normalize(model_matrix * vertex_buffer[vi].tangent);
	var b = normalize(model_matrix * vertex_buffer[vi].bitangent);

    var output : VertexOutput;
    output.position = mvp * vertex_buffer[vi].position;
    output.color = vertex_buffer[vi].color;
    output.uv = vertex_buffer[vi].uv;
    output.normal = n;
    output.tangent = t;
    output.bitangent = b;
    output.instance_index = ii;
    output.material_index = object_instances[ii].material;

    return output;
}

@fragment fn fs(v_out: VertexOutput) -> FragmentOutput {
    var output : FragmentOutput;
    output.albedo = v_out.color;
    output.normal = v_out.normal;
    output.position = v_out.position;
    return output;
}