#include "common.wgsl"

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
    @location(1) uv: vec2f,
};

@vertex fn vs(
    @builtin(vertex_index) vi : u32,
    @builtin(instance_index) ii: u32
) -> VertexOutput {
    var mvp = view_buffer[0].projection_matrix * view_buffer[0].view_matrix;
    var output : VertexOutput;
    output.position = mvp * vertex_buffer[vi].position;
    output.color = vertex_buffer[vi].color;
    output.uv = vertex_buffer[vi].uv;
    return output;
}

@fragment fn fs(v_out: VertexOutput) -> @location(0) vec4f {
    return v_out.color;
}