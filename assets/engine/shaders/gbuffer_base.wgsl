#include "common.wgsl"

struct VertexOutput {
    @builtin(position) @invariant position: vec4f,
    @location(0) local_position: vec4f,
    @location(1) view_position: vec4f,
    @location(2) world_position: vec4f,
    @location(3) color: vec4f,
    @location(4) uv: vec2f,
    @location(5) normal: vec4f,
    @location(6) tangent: vec4f,
    @location(7) bitangent: vec4f,
    @location(8) @interpolate(flat) instance_index: u32,
    @location(9) @interpolate(flat) instance_id: u32,
};

#ifndef SKIP_ENTITY_WRITES
const transparency_reveal_location = 6;
#else
const transparency_reveal_location = 5;
#endif

struct FragmentOutput {
    @location(0) albedo: vec4f,
    @location(1) emissive: vec4f,
    @location(2) smra: vec4f,
    @location(3) position: vec4f,
    @location(4) normal: vec4f,
#ifndef SKIP_ENTITY_WRITES
    @location(5) entity_id: u32,
#endif
#if TRANSPARENT
    @location(transparency_reveal_location) transparency_reveal: f32,
#endif
}

@group(1) @binding(0) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(1) var<storage, read> entity_inverse_transforms: array<EntityInverseTransform>;
@group(1) @binding(2) var<storage, read> entity_bounds_data: array<EntityBoundsData>;
@group(1) @binding(3) var<storage, read> compacted_object_instances: array<CompactedObjectInstance>;

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
    let view_mat = view_buffer[0].view_matrix;
    let view_proj_mat = view_buffer[0].view_projection_matrix;

    var output : VertexOutput;

    output.local_position = instance_vertex.position;
    output.world_position = entity_transform.transform * output.local_position;
    output.color = instance_vertex.color;
    output.uv = instance_vertex.uv;
    output.normal = normalize(vec4f((entity_inverse_transforms[entity].transpose_inverse_model_matrix * instance_vertex.normal).xyz, 1.0));
    output.instance_index = ii;
    output.instance_id = entity;

    output = vertex(&output);

#ifndef FINAL_POSITION_WRITE
    output.view_position = view_mat * output.world_position;
    output.position = view_proj_mat * output.world_position;
#endif

    return output;
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

#ifndef SKIP_ENTITY_WRITES
    output.entity_id = v_out.instance_id;
#endif

    var post_material_output = fragment(v_out, &output);

#if TRANSPARENT
    let weight = clamp(pow(min(1.0, post_material_output.albedo.a * 10.0) + 0.01, 3.0) * 1e8 * pow(1.0 - v_out.position.z * 0.9, 3.0), 1e-2, 3e3); 
    post_material_output.transparency_reveal = post_material_output.albedo.a;
    post_material_output.albedo = vec4f(post_material_output.albedo.rgb * post_material_output.albedo.a, post_material_output.albedo.a) * weight;
    post_material_output.normal.w = 0.0; // Disable deferred standard lighting for transparent objects
    // TODO: Run forward transparent lighting instead for this pixel 
#endif

    return post_material_output;
}