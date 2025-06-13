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
    let view_index = frame_info.view_index;
    var rotation_view = mat3x3f(
        view_buffer[view_index].view_matrix[0].xyz,
        view_buffer[view_index].view_matrix[1].xyz,
        view_buffer[view_index].view_matrix[2].xyz
    );
    
    // Apply rotation to the cube vertex so we get
    // a direction vector that follows the camera
    var rotated_pos = rotation_view * vec3<f32>(vertex_buffer[vi].position.xyz);
    
    // Full-screen quad in clip-space, pushed to the far plane
    output.position = vec4f(vertex_buffer[vi].position.xy, 1.0, 1.0);
    output.position.z = 1.0;               // (= far plane)

    // Pass the rotated direction to the fragment stage
    output.pos = vec4f(rotated_pos, 0.0);
    
    return output;
}

// ------------------------------------------------------------------------------------
// Fragment Shader
// ------------------------------------------------------------------------------------ 
@fragment fn fs(v_out: VertexOutput) -> FragmentOutput {
    let dir   = normalize(v_out.pos.xyz);            // view-space direction
    let color = textureSample(skybox_texture, global_sampler, dir) *
                skybox_data.color;
    return FragmentOutput(vec4<precision_float>(color));
}