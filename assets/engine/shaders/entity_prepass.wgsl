#include "common.wgsl"

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(flat) entity_id: u32,
};

struct FragmentOutput {
    @location(0) entity_id: u32,
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

@vertex fn vs(
    @builtin(vertex_index) vi : u32,
    @builtin(instance_index) ii: u32
) -> VertexOutput {
    var model_matrix = entity_transforms[object_instances[ii].entity].model_matrix;
    var mvp = view_buffer[0].view_projection_matrix * model_matrix;

    var output : VertexOutput;
    output.position = mvp * vertex_buffer[vi].position;
    output.entity_id = object_instances[ii].entity;

    return output;
}

@fragment fn fs(v_out: VertexOutput) -> FragmentOutput {
    var output : FragmentOutput;

    output.entity_id = v_out.entity_id;

    return output;
}