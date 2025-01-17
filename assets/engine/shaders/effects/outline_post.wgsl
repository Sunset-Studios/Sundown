#include "postprocess_common.wgsl"
#include "common.wgsl"

// ------------------------------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------------------------------

struct OutlineParams {
    outline_thickness: f32,
    depth_threshold: f32,
    normal_threshold: f32,
    depth_scale: f32,
    outline_color: vec4f,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
    @location(1) uv: vec2f,
    @location(2) @interpolate(flat) instance_index: u32,
};

struct FragmentOutput {
    @location(0) color: vec4f,
};

// ------------------------------------------------------------------------------------
// Buffers
// ------------------------------------------------------------------------------------

@group(1) @binding(0) var<uniform> outline_params: OutlineParams;
@group(1) @binding(1) var color_texture: texture_2d<f32>;
@group(1) @binding(2) var depth_texture: texture_2d<f32>;
@group(1) @binding(3) var normal_texture: texture_2d<f32>;

// ------------------------------------------------------------------------------------
// Helper Functions
// ------------------------------------------------------------------------------------

fn sample_depth(uv: vec2f) -> f32 {
    return textureSample(depth_texture, non_filtering_sampler, uv).r;
}

fn sample_normal(uv: vec2f) -> vec3f {
    return textureSample(normal_texture, global_sampler, uv).xyz * 2.0 - 1.0;
}

fn detect_edge(uv: vec2f) -> f32 {
    let pixel_size = vec2f(1.0) / vec2f(textureDimensions(depth_texture));
    
    // Sample the 4 adjacent pixels (cross pattern)
    let offsets = array<vec2f, 4>(
        vec2f(1.0, 0.0),  // right
        vec2f(-1.0, 0.0), // left
        vec2f(0.0, 1.0),  // up
        vec2f(0.0, -1.0)  // down
    );
    
    // Sample center pixel
    let depth_center = sample_depth(uv);
    let normal_center = sample_normal(uv);
    
    var edge = 0.0;
    
    // Check each adjacent pixel
    for (var i = 0; i < 4; i++) {
        let uv_offset = uv + offsets[i] * pixel_size * outline_params.outline_thickness;
        
        // Sample neighbor
        let depth_sample = sample_depth(uv_offset);
        let normal_sample = sample_normal(uv_offset);
        
        // Compute depth difference
        let depth_diff = abs(depth_center - depth_sample) * outline_params.depth_scale;
        
        // Compute normal difference
        let normal_diff = 1.0 - max(dot(normal_center, normal_sample), 0.0);
        
        // Combine depth and normal edges
        let depth_edge = step(outline_params.depth_threshold, depth_diff);
        let normal_edge = step(outline_params.normal_threshold, normal_diff);
        
        edge = max(edge, max(depth_edge, normal_edge));
    }
    
    return edge;
}

// ------------------------------------------------------------------------------------
// Fragment Shader
// ------------------------------------------------------------------------------------

@fragment
fn fs(v_out: VertexOutput) -> FragmentOutput {
    let edge = detect_edge(v_out.uv);
    
    // Sample original color
    let original_color = textureSample(color_texture, global_sampler, v_out.uv);
    
    // Only show outline (no white dots)
    let final_color = select(original_color, outline_params.outline_color, edge > 0.1);
    
    return FragmentOutput(final_color);
} 