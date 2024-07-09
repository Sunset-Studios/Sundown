struct Vertex {
    position: vec4f,
    normal: vec4f,
    color: vec4f,
    uv: vec2f,
    tangent: vec4f,
    bitangent: vec4f,
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

@group(0) @binding(0) var<storage, read> vertex_buffer: array<Vertex>;
@group(0) @binding(1) var<storage, read> view_buffer: array<View>;
@group(0) @binding(2) var global_sampler: sampler;
@group(0) @binding(3) var<uniform> frame_info: FrameInfo;