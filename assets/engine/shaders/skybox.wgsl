#include "common.wgsl"

struct SkyboxData {
    @location(0) color: vec4<f32>,
};

@group(1) @binding(0) var skybox_texture: texture_cube<f32>;
@group(1) @binding(1) var<uniform> skybox_data: SkyboxData;

// ------------------------------------------------------------------------------------
// Data Structures

// ------------------------------------------------------------------------------------ 
struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) pos: vec4f,
};

struct FragmentOutput {
    @location(0) color: vec4<precision_float>,
};

// ------------------------------------------------------------------------------------
// Vertex Shader
// ------------------------------------------------------------------------------------ 
@vertex fn vs(
    @builtin(vertex_index) vi : u32,
    @builtin(instance_index) ii: u32
) -> VertexOutput {
    var output : VertexOutput;
    
    // Extract the rotation part of the view matrix (3x3 upper-left part)
    var rotation_view = mat3x3f(
        view_buffer[0].view_matrix[0].xyz,
        view_buffer[0].view_matrix[1].xyz,
        view_buffer[0].view_matrix[2].xyz
    );
    
    // Apply rotation to the vertex position
    var rotated_pos = rotation_view * vec3<f32>(vertex_buffer[vi].position.xyz);
    
    // Apply projection matrix
    output.position = view_buffer[0].projection_matrix * vec4f(rotated_pos, 1.0);
    
    // Store the rotated position for fragment shader
    output.pos = 0.5 * (vec4<f32>(vertex_buffer[vi].position) + vec4<f32>(1.0, 1.0, 1.0, 1.0)); 
    
    return output;
}

// ------------------------------------------------------------------------------------
// Fragment Shader
// ------------------------------------------------------------------------------------ 
@fragment fn fs(v_out: VertexOutput) -> FragmentOutput {
    var dir = v_out.pos.xyz - vec3(0.5);
    dir.z *= -1;
    var color = textureSample(skybox_texture, global_sampler, dir) * skybox_data.color;
    return FragmentOutput(vec4<precision_float>(color));
}