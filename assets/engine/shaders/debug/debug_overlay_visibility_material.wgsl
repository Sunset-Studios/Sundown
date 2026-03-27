#include "common.wgsl"
#include "visibility/visibility_common.wgsl"

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(1) @binding(0) var visibility_entity_texture: texture_2d<u32>;
@group(1) @binding(1) var visibility_surface_texture: texture_2d<u32>;
@group(1) @binding(2) var<storage, read> meshlets: array<MeshletRecord>;
@group(1) @binding(3) var<storage, read> meshlet_vertices: array<u32>;
@group(1) @binding(4) var<storage, read> meshlet_triangles: array<u32>;
@group(1) @binding(5) var<storage, read> material_table_offset: array<u32>;
@group(1) @binding(6) var<storage, read> material_palette: array<u32>;

@fragment
fn fs(input: VertexOutput) -> @location(0) vec4<f32> {
    let size = textureDimensions(visibility_entity_texture);
    let coord = vec2<i32>(input.uv * vec2<f32>(size));
    let entity_id = textureLoad(visibility_entity_texture, coord, 0).x;
    if (entity_id == INVALID_IDX) {
        return vec4<f32>(0.0);
    }

    let surface = textureLoad(visibility_surface_texture, coord, 0).x;
    let meshlet = meshlets[unpack_surface_meshlet(surface)];
    let tri_offset = meshlet.triangle_offset + unpack_surface_triangle(surface) * 3u;
    let local_index = meshlet_triangles[tri_offset];
    let global_vertex_index = meshlet_vertices[meshlet.vertex_offset + local_index];
    let section_index = u32(max(vertex_section_index(vertex_buffer[global_vertex_index]), 0.0));
    let palette_offset = material_table_offset[entity_id];
    let material_id = material_palette[palette_offset + section_index];

    return vec4<f32>(id_to_color(material_id), 1.0);
}
