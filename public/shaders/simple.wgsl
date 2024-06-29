struct Vertex {
    position: vec4f,
    color: vec4f,
    uv: vec2f,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
    @location(1) uv: vec2f,
};

@group(1) @binding(0) var<storage, read> vertices: array<Vertex>;

@vertex fn vs(
    @builtin(vertex_index) vi : u32,
    @builtin(instance_index) ii: u32
) -> VertexOutput {
    var output : VertexOutput;
    output.position = vertices[vi].position;
    output.color = vertices[vi].color;
    output.uv = vertices[vi].uv;
    return output;
}

@fragment fn fs(v_out: VertexOutput) -> @location(0) vec4f {
    return v_out.color;
}