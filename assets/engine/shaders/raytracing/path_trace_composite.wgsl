// =============================================================================
// Path Trace Composite Shader
// - Composites path traced output over skybox background
// - Uses G-buffer normal to determine where geometry exists
// =============================================================================
#include "common.wgsl"

@group(1) @binding(0) var skybox_texture: texture_2d<f32>;
@group(1) @binding(1) var path_trace_texture: texture_2d<f32>;
@group(1) @binding(2) var normal_texture: texture_2d<f32>;
@group(1) @binding(3) var output_tex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(output_tex);
    
    if (gid.x >= res.x || gid.y >= res.y) { return; }
    
    let pixel_coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) / vec2<f32>(f32(res.x), f32(res.y));
    
    // Read skybox background
    let skybox_color = textureSampleLevel(skybox_texture, global_sampler, uv, 0.0);
    
    // Read path traced result
    let path_trace_color = textureSampleLevel(path_trace_texture, global_sampler, uv, 0.0);
    
    // Read normal from G-buffer to determine if there's geometry
    let normal_data = textureSampleLevel(normal_texture, global_sampler, uv, 0.0);
    let normal = normal_data.xyz;
    let normal_length = length(normal);
    
    // If there's no geometry (normal length ~= 0), use skybox as background
    // Otherwise, use the path traced result
    // We blend based on a threshold to avoid harsh edges
    let has_geometry = select(0.0, 1.0, normal_length > 0.01);
    
    // Composite: path trace over skybox
    // Where there's geometry, show path trace. Where there's none, show skybox
    var final_color = mix(skybox_color.rgb, path_trace_color.rgb, has_geometry);
    
    textureStore(output_tex, pixel_coord, vec4<f32>(final_color, 1.0));
}

