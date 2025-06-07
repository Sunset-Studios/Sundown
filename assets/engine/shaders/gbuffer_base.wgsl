#include "common.wgsl"
#include "lighting_common.wgsl"

// ------------------------------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------------------------------ 
struct VertexOutput {
    @builtin(position) @invariant position: vec4f,
    @location(0) local_position: vec4<precision_float>,
    @location(1) view_position: vec4<f32>,
    @location(2) world_position: vec4<f32>,
    @location(3) color: vec4<precision_float>,
    @location(4) uv: vec2<precision_float>,
    @location(5) normal: vec4<precision_float>,
    @location(6) tangent: vec4<precision_float>,
    @location(7) bitangent: vec4<precision_float>,
    @location(8) @interpolate(flat) instance_index: u32,
    @location(9) @interpolate(flat) instance_id: u32,
    @location(10) @interpolate(flat) vertex_index: u32,
};

struct FragmentOutput {
    @location(0) albedo: vec4<precision_float>,
    @location(1) emissive: vec4<precision_float>,
    @location(2) smra: vec4<precision_float>,
    @location(3) position: vec4<f32>,
    @location(4) normal: vec4<precision_float>,
#ifndef SKIP_ENTITY_WRITES
    @location(5) entity_id: vec2<u32>,
#endif
#if TRANSPARENT
    @location(transparency_reveal_location) transparency_reveal: f32,
#endif
}

// ------------------------------------------------------------------------------------
// Constants
// ------------------------------------------------------------------------------------ 

#ifndef SKIP_ENTITY_WRITES
const transparency_reveal_location = 6;
#else
const transparency_reveal_location = 5;
#endif

// ------------------------------------------------------------------------------------
// Buffers
// ------------------------------------------------------------------------------------ 

@group(1) @binding(0) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(1) var<storage, read> entity_flags: array<u32>;
@group(1) @binding(2) var<storage, read> object_instances: array<ObjectInstance>;
@group(1) @binding(3) var<storage, read> visible_object_instances: array<i32>;

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
    let instance_vertex = vertex_buffer[vi];
    let object_instance_index = visible_object_instances[ii];
    let entity_resolved = get_entity_row(object_instances[object_instance_index].row);

    let entity_transform = entity_transforms[entity_resolved];
    let view_index = frame_info.view_index;
    let view_mat = view_buffer[view_index].view_matrix;
    let view_proj_mat = view_buffer[view_index].view_projection_matrix;

    var output : VertexOutput;

    output.uv = instance_vertex.uv;
    output.instance_index = ii;
    output.instance_id = entity_resolved;
    output.vertex_index = vi;
    output.local_position = instance_vertex.position;

    output.world_position = select(
        entity_transform.transform * vec4<f32>(output.local_position),
        billboard_vertex_local(
            output.uv,
            entity_transform.transform
        ),
        (entity_flags[entity_resolved] & EF_BILLBOARD) != 0
    );

    let n = normalize((entity_transform.transpose_inverse_model_matrix * vec4<f32>(instance_vertex.normal)).xyz);
    let t = normalize((entity_transform.transform * vec4<f32>(instance_vertex.tangent.xyz, 0.0)).xyz);
    let b = normalize((entity_transform.transform * vec4<f32>(instance_vertex.bitangent.xyz, 0.0)).xyz);

    output.normal = vec4<precision_float>(n, 0.0);
    output.tangent = vec4<precision_float>(t, 0.0);
    output.bitangent = vec4<precision_float>(b, 0.0);

    output = vertex(&output);

#ifndef FINAL_POSITION_WRITE
    output.view_position = view_mat * output.world_position;
    output.position = view_proj_mat * output.world_position;
#endif

    return output;
}

// ------------------------------------------------------------------------------------
// Fragment Shader
// ------------------------------------------------------------------------------------ 

#ifndef CUSTOM_FS
fn fragment(v_out: VertexOutput, f_out: ptr<function, FragmentOutput>) -> FragmentOutput {
    return *f_out;
}
#endif

@fragment fn fs(v_out: VertexOutput) -> FragmentOutput {
    var output : FragmentOutput;

    output.position = v_out.world_position;
    // Last component of normal is deferred standard lighting factor. Set to 0 if custom lighting is used when using custom FS / VS.
    output.normal = vec4<precision_float>(v_out.normal.xyz, 1.0);

#ifndef SKIP_ENTITY_WRITES
    output.entity_id = vec2<u32>(v_out.instance_id, v_out.instance_id);
#endif

    var post_material_output = fragment(v_out, &output);

#if TRANSPARENT
    if (post_material_output.albedo.a <= 0.0) {
        discard;
    } 

    let color = (post_material_output.emissive.r * post_material_output.albedo.rgb);

    let weight = clamp(pow(min(1.0, post_material_output.albedo.a * 10.0) + 0.01, 3.0) * 1e8 * pow(1.0 - v_out.position.z * 0.9, 3.0), 1e-2, 3e3); 
    post_material_output.transparency_reveal = post_material_output.albedo.a;
    post_material_output.albedo = vec4f(color * post_material_output.albedo.a, post_material_output.albedo.a) * weight;
    post_material_output.normal = vec4f(0.0); // Treat transparency as unlit in deferred lighting pass; We've already done lighting here
#endif

    return post_material_output;
}