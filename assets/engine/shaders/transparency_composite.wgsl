#include "common.wgsl"

@group(1) @binding(0) var accumulation_texture: texture_2d<f32>;
@group(1) @binding(1) var reveal_texture: texture_2d<f32>;

// ------------------------------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------------------------------ 
struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2<precision_float>,
    @location(1) @interpolate(flat) instance_index: u32,
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
    output.position = vec4<f32>(vertex_buffer[vi].position);
    output.uv = vertex_buffer[vi].uv;
    output.instance_index = ii;
    return output;
}

// ------------------------------------------------------------------------------------
// Fragment Shader
// ------------------------------------------------------------------------------------ 
@fragment fn fs(v_out: VertexOutput) -> FragmentOutput {
    var reveal = vec4<precision_float>(textureSample(reveal_texture, global_sampler, vec2<f32>(v_out.uv)));
    var accum = vec4<precision_float>(textureSample(accumulation_texture, global_sampler, vec2<f32>(v_out.uv)));
    var average_color = accum.rgb / max(accum.a, epsilon);
    return FragmentOutput(vec4<precision_float>(average_color, reveal.r));

}