#include "common.wgsl"

@group(1) @binding(0) var input_texture: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
    @location(1) uv: vec2f,
    @location(2) @interpolate(flat) instance_index: u32,
};

struct FragmentOutput {
    @location(0) color: vec4f,
};

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

@fragment fn fs(v_out: VertexOutput) -> FragmentOutput {
    var color = textureSample(input_texture, global_sampler, v_out.uv);
    return FragmentOutput(color);
}