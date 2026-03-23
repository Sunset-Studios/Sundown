#include "common.wgsl"
#include "lighting_common.wgsl"
#include "postprocess_common.wgsl"
#include "sky_common.wgsl"

@group(1) @binding(0) var skybox_texture: texture_cube<f32>;
@group(1) @binding(1) var<uniform> scene_lighting_data: SceneLightingData;

// ------------------------------------------------------------------------------------
// Data Structures

// ------------------------------------------------------------------------------------ 
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) pos: vec4<f32>,
    @location(1) world_position: vec4<f32>,
    @location(2) sun_dir: vec3<f32>,
};

struct FragmentOutput {
    @location(0) color: vec4<f32>,
};

// ------------------------------------------------------------------------------------
// Vertex Shader
// ------------------------------------------------------------------------------------ 
@vertex fn vs(
    @builtin(vertex_index) vi : u32,
    @builtin(instance_index) ii: u32
) -> VertexOutput {
    var output : VertexOutput;
    
    let view_index = u32(frame_info.view_index);
    let local_position = vertex_position4(vertex_buffer[vi]);

    if (scene_lighting_data.sky_type < 1.0) {
        // Extract the rotation part of the view matrix (3x3 upper-left part)
        var rotation_view = mat3x3<f32>(
            view_buffer[view_index].view_matrix[0].xyz,
            view_buffer[view_index].view_matrix[1].xyz,
            view_buffer[view_index].view_matrix[2].xyz
        );
        
        // Apply rotation to the cube vertex so we get
        // a direction vector that follows the camera
        var rotated_pos = rotation_view * vec3<f32>(local_position.xyz);
        
        // Full-screen quad in clip-space, pushed to the far plane
        output.position = vec4<f32>(local_position.xy, 1.0, 1.0);
        output.position.z = 1.0;               // (= far plane)

        // Pass the rotated direction to the fragment stage
        output.pos = vec4<f32>(rotated_pos, 0.0);
    } else {
        // Get camera position (fallback if camera_data is not available)
        let camera_pos = view_buffer[view_index].view_position.xyz;
        let light_view = view_buffer[u32(scene_lighting_data.view_index)];
        
        // Create model matrix positioned at camera
        let model_matrix = mat4x4<f32>(
            vec4<f32>(1.0, 0.0, 0.0, 0.0),
            vec4<f32>(0.0, 1.0, 0.0, 0.0),
            vec4<f32>(0.0, 0.0, 1.0, 0.0),
            vec4<f32>(camera_pos.x, camera_pos.y, camera_pos.z, 1.0)
        );

        output.world_position = model_matrix * vec4<f32>(local_position.xyz, 1.0);
        // Calculate sun direction and intensity
        output.sun_dir = normalize(-light_view.view_direction.xyz);
        // Project to clip space
        let view_proj = view_buffer[view_index].view_projection_matrix;
        // Project to clip space
        output.position = view_proj * output.world_position;
        output.position.z = output.position.w;
    }
    
    return output;
}

// ------------------------------------------------------------------------------------
// Fragment Shader
// ------------------------------------------------------------------------------------ 
@fragment fn fs(v_out: VertexOutput) -> FragmentOutput {
    var output: FragmentOutput;

    if (scene_lighting_data.sky_type < 1.0) {
        let dir   = normalize(v_out.pos.xyz);            // view-space direction
        output.color = textureSample(skybox_texture, global_sampler, dir) *
                        scene_lighting_data.skybox_color;
    } else {
        // Get camera position and calculate view direction
        let camera_pos = view_buffer[u32(frame_info.view_index)].view_position.xyz;
        let view_dir = normalize(v_out.world_position.xyz - camera_pos);
        // Evaluate sky
        output.color = vec4<f32>(evaluate_sky(view_dir, v_out.sun_dir, scene_lighting_data), 1.0);
    }

    return output;
}