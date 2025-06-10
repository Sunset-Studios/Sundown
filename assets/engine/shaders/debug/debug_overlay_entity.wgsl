#include "common.wgsl"

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2<precision_float>,
    @location(1) @interpolate(flat) instance_index: u32,
};

@group(1) @binding(0) var debug_texture: texture_2d<u32>;

@fragment
fn fs(input: VertexOutput) -> @location(0) vec4<f32> {
    let size = textureDimensions(debug_texture);
    let packed_id = textureLoad(debug_texture,
                               vec2<u32>(input.uv * vec2<f32>(size)),
                               0).r;
    let color = get_entity_color(packed_id);
    return vec4<f32>(color, 1.0);
}

// ----------------------------------------------------------------
// new helper: turn a u32 entity ID into a "random" but stable color
fn get_entity_color(id: u32) -> vec3<f32> {
    // 32-bit LCG hash constants
    let hash = id * 1664525u + 1013904223u;
    // pull out three bytes
    let r = f32((hash >>  0) & 0xFFu) / 255.0;
    let g = f32((hash >>  8) & 0xFFu) / 255.0;
    let b = f32((hash >> 16) & 0xFFu) / 255.0;
    return vec3<f32>(r, g, b);
} 