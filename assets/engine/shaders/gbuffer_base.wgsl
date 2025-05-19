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
@group(1) @binding(2) var<storage, read> compacted_object_instances: array<CompactedObjectInstance>;
@group(1) @binding(3) var<storage, read> lights_buffer: array<Light>; // Used for forward shading if necessary

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
    let entity_resolved = get_entity_row(compacted_object_instances[ii].row);

    let entity_transform = entity_transforms[entity_resolved];
    let view_mat = view_buffer[0].view_matrix;
    let view_proj_mat = view_buffer[0].view_projection_matrix;

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

    var view_dir = normalize(-view_buffer[0].view_direction.xyz);
    var color = vec3f(0.0);
    let num_lights = arrayLength(&lights_buffer) * min(1u, u32(post_material_output.normal.w));

    for (var i = 0u; i < num_lights; i++) {
        var light = lights_buffer[i];
        if (light.activated <= 0.0) {
            continue;
        }
        color += calculate_brdf(
            light,
            post_material_output.normal.xyz,
            view_dir,
            post_material_output.position.xyz,
            post_material_output.albedo.rgb,
            post_material_output.smra.r * 0.0009765625 /* 1.0f / 1024 */,
            post_material_output.smra.g,
            post_material_output.smra.b,
            0.0, // clear coat
            1.0, // clear coat roughness 
            post_material_output.smra.a,
            vec3f(1.0, 1.0, 1.0), // irradiance
            vec3f(1.0, 1.0, 1.0), // prefilter color 
            vec2f(1.0, 1.0), // env brdf
            0, // shadow map index
        );
    }

    color += (post_material_output.emissive.r * post_material_output.albedo.rgb);

    let weight = clamp(pow(min(1.0, post_material_output.albedo.a * 10.0) + 0.01, 3.0) * 1e8 * pow(1.0 - v_out.position.z * 0.9, 3.0), 1e-2, 3e3); 
    post_material_output.transparency_reveal = post_material_output.albedo.a;
    post_material_output.albedo = vec4f(color * post_material_output.albedo.a, post_material_output.albedo.a) * weight;
    post_material_output.normal = vec4f(0.0); // Treat transparency as unlit in deferred lighting pass; We've already done lighting here
#endif

    return post_material_output;
}