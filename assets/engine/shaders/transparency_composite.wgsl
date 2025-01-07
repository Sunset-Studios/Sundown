#include "common.wgsl"

@group(1) @binding(0) var accumulation_texture: texture_2d<f32>;
@group(1) @binding(1) var reveal_texture: texture_2d<f32>;

// ------------------------------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------------------------------ 
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
// Vertex Shader
// ------------------------------------------------------------------------------------ 
@vertex fn vs(
    @builtin(vertex_index) vi : u32,
    @builtin(instance_index) ii: u32
) -> VertexOutput {
    var output : VertexOutput;
    output.position = vertex_buffer[vi].position;
    output.color = vertex_buffer[vi].color;
    output.uv = vertex_buffer[vi].uv;
    output.instance_index = ii;
    return output;
}

// ------------------------------------------------------------------------------------
// Fragment Shader
// ------------------------------------------------------------------------------------ 
@fragment fn fs(v_out: VertexOutput) -> FragmentOutput {
    var reveal = textureSample(reveal_texture, global_sampler, v_out.uv);
    var accum = textureSample(accumulation_texture, global_sampler, v_out.uv);
    var average_color = accum.rgb / max(accum.a, epsilon);
    return FragmentOutput(vec4f(average_color, reveal.r));
}