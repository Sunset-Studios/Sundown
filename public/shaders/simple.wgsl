struct Vertex {
    position: vec4f,
    normal: vec4f,
    color: vec4f,
    uv: vec2f,
};

struct View {
    view_matrix: mat4x4f,
    prev_view_matrix: mat4x4f,
    projection_matrix: mat4x4f,
    prev_projection_matrix: mat4x4f
};

struct FrameInfo {
    view_index: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
    @location(1) uv: vec2f,
};

@group(0) @binding(0) var<storage, read> vertex_buffer: array<Vertex>;
@group(0) @binding(1) var<storage, read> view_buffer: array<View>;
@group(0) @binding(2) var<uniform> frame_info: FrameInfo;

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
    return v_out.position;
}