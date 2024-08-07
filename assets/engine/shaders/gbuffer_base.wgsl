#include "common.wgsl"

struct VertexOutput {
    @builtin(position) @invariant position: vec4f,
    @location(0) view_position: vec4f,
    @location(1) world_position: vec4f,
    @location(2) color: vec4f,
    @location(3) uv: vec2f,
    @location(4) normal: vec4f,
    @location(5) tangent: vec4f,
    @location(6) bitangent: vec4f,
    @location(7) @interpolate(flat) instance_id: u32,
};

struct FragmentOutput {
    @location(0) albedo: vec4f,
    @location(1) smra: vec4f,
    @location(2) position: vec4f,
    @location(3) normal: vec4f,
    @location(4) entity_id: u32,
}

@group(1) @binding(0) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(1) var<storage, read> compacted_object_instances: array<CompactedObjectInstance>;

#ifndef CUSTOM_VS
fn vertex(v_out: ptr<function, VertexOutput>) -> VertexOutput {
    return *v_out;
}
#endif

@vertex fn vs(
    @builtin(vertex_index) vi : u32,
    @builtin(instance_index) ii: u32
) -> VertexOutput {
    let instance_vertex = vertex_buffer[vi];
    let entity = compacted_object_instances[ii].entity;
    let entity_transform = entity_transforms[entity];

    var output : VertexOutput;

    output.position = view_buffer[0].view_projection_matrix * entity_transform.transform * instance_vertex.position;
    output.view_position = view_buffer[0].view_matrix * entity_transform.transform * instance_vertex.position;
    output.world_position = entity_transform.transform * instance_vertex.position;
    output.color = instance_vertex.color;
    output.uv = instance_vertex.uv;
    output.normal = normalize(vec4f((entity_transform.transpose_inverse_model_matrix * instance_vertex.normal).xyz, 1.0));
    output.instance_id = entity;

    return vertex(&output);
}

#ifndef CUSTOM_FS
fn fragment(v_out: VertexOutput, f_out: ptr<function, FragmentOutput>) -> FragmentOutput {
    return *f_out;
}
#endif

@fragment fn fs(v_out: VertexOutput) -> FragmentOutput {
    var output : FragmentOutput;

    output.position = v_out.world_position;
    // Last component of normal is deferred standard lighting factor. Set to 0 if custom lighting is used when using custom FS / VS.
    output.normal = vec4f(v_out.normal.xyz, 1.0);
    output.entity_id = v_out.instance_id;

    return fragment(v_out, &output);
}