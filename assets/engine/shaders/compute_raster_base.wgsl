#include "common.wgsl"

// ------------------------------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------------------------------ 

struct Vertex {
    position: vec4f,
    uv: vec2f,
    normal: vec4f,
};

struct Fragment {
    color: vec4f,
    depth: f32,
};

// ------------------------------------------------------------------------------------
// Constants
// ------------------------------------------------------------------------------------ 

const workgroup_size = 256;

// ------------------------------------------------------------------------------------
// Buffers
// ------------------------------------------------------------------------------------ 

@group(0) @binding(0) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;

@group(0) @binding(2) var<storage, write> framebuffer: array<vec4f>;
@group(0) @binding(3) var<storage, write> depthbuffer: array<f32>;

// ------------------------------------------------------------------------------------
// Rasterization Methods
// ------------------------------------------------------------------------------------ 

fn rasterize_point(position: vec2f, color: vec4f) {
    let index = u32(position.y) * workgroup_size + u32(position.x);
    if (index < framebuffer.length()) {
        framebuffer[index] = color;
        depthbuffer[index] = 0.0; // Example depth
    }
}

fn rasterize_line(start: vec2f, end: vec2f, color: vec4f) {
    // Implement Bresenham's line algorithm or similar
    // ...
}

fn rasterize_triangle(v0: vec2f, v1: vec2f, v2: vec2f, color: vec4f) {
    // Implement triangle rasterization (e.g., barycentric coordinates)
    // ...
}

fn rasterize_quad(v0: vec2f, v1: vec2f, v2: vec2f, v3: vec2f, color: vec4f) {
    // Rasterize two triangles to form a quad
    rasterize_triangle(v0, v1, v2, color);
    rasterize_triangle(v2, v3, v0, color);
}

// ------------------------------------------------------------------------------------
// Compute Shader Entry
// ------------------------------------------------------------------------------------ 

#ifndef CUSTOM_RASTER
fn rasterize(v_out: VertexOutput, f_out: ptr<function, FragmentOutput>) -> FragmentOutput {
    return *f_out;
}
#endif

@compute @workgroup_size(workgroup_size)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    rasterize(vec2f(x, y), vec4f(1.0, 0.0, 0.0, 1.0));
}


