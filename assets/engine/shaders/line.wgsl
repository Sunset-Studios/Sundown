#include "common.wgsl"

struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
    @builtin(instance_index) instance_index: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<precision_float>,
    @location(0) color: vec4<precision_float>,
    @location(1) uv: vec2<precision_float>,
};

struct FragmentOutput {
    @location(0) color: vec4<precision_float>,
    @location(1) emissive: vec4<precision_float>,
    @location(2) smra: vec4<precision_float>,
    @location(3) position: vec4<f32>,
    @location(4) normal: vec4<precision_float>,
}

struct TransformData {
    transform: mat4x4f,
};

struct LineData {
    color_and_width: vec4f,
};

@group(1) @binding(0) var<storage, read> transform_data: array<TransformData>;
@group(1) @binding(1) var<storage, read> line_data: array<LineData>;

@vertex
fn vs(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    let vertex = vertex_buffer[input.vertex_index];
    let color_and_width = line_data[input.instance_index].color_and_width;
    let model_transform = transform_data[input.instance_index].transform;

    let model_view_transform = view_buffer[0].view_matrix * model_transform;
    
    // Calculate line direction in view space
    let view_start = (model_view_transform * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
    let view_end = (model_view_transform * vec4<f32>(1.0, 0.0, 0.0, 1.0)).xyz;
    let line_dir = normalize(view_end - view_start);
    
    // Calculate camera-facing perpendicular direction
    // This ensures consistent width regardless of viewing angle
    let view_dir = normalize(view_start); // Direction from camera to line start
    var perp_dir = normalize(cross(line_dir, view_dir));
    
    // If line_dir and view_dir are parallel, use a different vector
    if (length(perp_dir) < 0.001) {
        let alt_up = vec3<f32>(0.0, 1.0, 0.0);
        perp_dir = normalize(cross(line_dir, alt_up));
    }
    
    // The quad vertices are at UV coordinates (0,0), (1,0), (0,1), (1,1)
    let uv = vertex.uv;
    
    // Determine position along the line (x-coordinate of UV)
    let pos_along_line = mix(view_start, view_end, uv.x);
    
    // Determine offset from line center (y-coordinate of UV, remapped from [0,1] to [-0.5,0.5])
    let offset = perp_dir * (uv.y - 0.5) * color_and_width.w;
    
    // Final position in view space
    let final_view_pos = pos_along_line + offset;
    
    // Convert directly to clip space from view space
    output.position = view_buffer[0].projection_matrix * vec4<f32>(final_view_pos, 1.0);
    
    // Pass color and UV to fragment shader
    output.color = vec4<f32>(color_and_width.x, color_and_width.y, color_and_width.z, 1.0);
    output.uv = uv;
    
    return output;
}

@fragment
fn fs(input: VertexOutput) -> FragmentOutput {
    var output: FragmentOutput;

    // Calculate distance from center line for anti-aliasing
    let distance_from_center = abs(input.uv.y - 0.5) * 2.0; // Range 0 to 1
    
    // Discard pixels that are too far from the center
    if (distance_from_center > 0.95) {
        discard;
    }

    output.color = input.color;
    output.position = input.position;
    output.emissive = vec4f(1.0, 0.0, 0.0, 0.0);
    //output.smra = vec4f(0.0, 0.0, 0.0, 0.0);
    output.normal = vec4f(0.0, 0.0, 0.0, 0.0);
    
    return output;
} 