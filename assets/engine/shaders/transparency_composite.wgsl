#include "common.wgsl"

@group(1) @binding(0) var accumulation_texture: texture_2d<f32>;

// ------------------------------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------------------------------ 
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) instance_index: u32,
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
    output.position = vertex_position4(vertex_buffer[vi]);
    output.uv = vertex_uv(vertex_buffer[vi]);
    output.instance_index = ii;
    return output;
}

// ------------------------------------------------------------------------------------
// Fragment Shader
// ------------------------------------------------------------------------------------ 
@fragment fn fs(v_out: VertexOutput) -> FragmentOutput {
    let accum = vec4<f32>(textureSample(accumulation_texture, global_sampler, vec2<f32>(v_out.uv)));
    var average_color = accum.rgb / max(accum.a, epsilon);
    return FragmentOutput(vec4<f32>(average_color, accum.a));

}