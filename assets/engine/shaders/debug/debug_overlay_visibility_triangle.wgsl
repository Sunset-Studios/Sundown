#include "common.wgsl"

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(1) @binding(0) var visibility_entity_texture: texture_2d<u32>;
@group(1) @binding(1) var visibility_surface_texture: texture_2d<u32>;

@fragment
fn fs(input: VertexOutput) -> @location(0) vec4<f32> {
    let size = textureDimensions(visibility_entity_texture);
    let coord = vec2<i32>(input.uv * vec2<f32>(size));
    let entity_id = textureLoad(visibility_entity_texture, coord, 0).x;
    if (entity_id == INVALID_IDX) {
        return vec4<f32>(0.0);
    }

    let surface = textureLoad(visibility_surface_texture, coord, 0).x;
    return vec4<f32>(id_to_color(surface), 1.0);
}
