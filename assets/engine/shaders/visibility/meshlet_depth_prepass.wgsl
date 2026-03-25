#include "common.wgsl"
#include "visibility/meshlet_common.wgsl"

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) material_index: u32,
}

@group(1) @binding(0) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(1) var<storage, read> object_instances: array<ObjectInstance>;
@group(1) @binding(2) var<storage, read> visible_meshlets: array<vec4<u32>>;
@group(1) @binding(3) var<storage, read> meshlets: array<MeshletRecord>;
@group(1) @binding(4) var<storage, read> meshlet_vertices: array<u32>;
@group(1) @binding(5) var<storage, read> meshlet_triangles: array<u32>;
@group(1) @binding(6) var<storage, read> entity_index_lookup: array<u32>;
@group(1) @binding(7) var<storage, read> material_params: array<StandardMaterialParams>;
@group(1) @binding(8) var<storage, read> material_table_offset: array<u32>;
@group(1) @binding(9) var<storage, read> material_palette: array<u32>;
@group(1) @binding(10) var texture_pool_albedo: texture_2d_array<f32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
    let visible_entry = visible_meshlets[ii];
    let global_meshlet_index = meshlet_index(visible_entry);
    let meshlet = meshlets[global_meshlet_index];

    let triangle_index = vi / 3u;
    let corner_index = vi % 3u;

    var output: VertexOutput;
    output.position = vec4<f32>(-2.0, -2.0, 1.0, 1.0);
    output.uv = vec2<f32>(0.0);
    output.material_index = 0u;

    if (triangle_index >= meshlet.triangle_count) {
        return output;
    }

    let object_instance_index = meshlet_object_index(visible_entry);

    let local_triangle_index = meshlet.triangle_offset + triangle_index * 3u + corner_index;
    let local_vertex_index = meshlet_triangles[local_triangle_index];
    let global_vertex_index = meshlet_vertices[meshlet.vertex_offset + local_vertex_index];
    let section_index = u32(max(vertex_section_index(vertex_buffer[global_vertex_index]), 0.0));

    let entity_row = get_entity_row(object_instances[object_instance_index].row);
    let entity_resolved = entity_index_lookup[entity_row];
    let transform = entity_transforms[entity_resolved].transform;
    let world_position = transform * vertex_position4(vertex_buffer[global_vertex_index]);

    let entity_palette_offset = material_table_offset[entity_resolved];
    let material_params_index = material_palette[entity_palette_offset + section_index];

    output.position = view_buffer[u32(frame_info.view_index)].view_projection_matrix * world_position;
    output.uv = vertex_uv(vertex_buffer[global_vertex_index]);
    output.material_index = material_params_index;

    return output;
}

fn fragment_mask(input: VertexOutput) -> f32 {
    let material = material_params[input.material_index];
    let base_uv = input.uv * material.emission_roughness_metallic_tiling.w;
    let tex_size = vec2<f32>(textureDimensions(texture_pool_albedo).xy);
    let lod = compute_lod_from_uv(base_uv, tex_size);
    let albedo = sample_texture_or_vec4_param_handle(
        u32(material.albedo_handle),
        base_uv,
        material.albedo,
        u32(material.texture_flags1.x),
        texture_pool_albedo,
        lod
    );
    return albedo.a;
}

@fragment
fn fs(input: VertexOutput) {
    let mask = fragment_mask(input);
    if (mask <= 0.0) {
        discard;
    }
}
