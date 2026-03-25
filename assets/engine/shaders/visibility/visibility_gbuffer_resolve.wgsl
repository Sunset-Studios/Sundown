#include "common.wgsl"
#include "visibility/meshlet_common.wgsl"

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct FragmentOutput {
    @location(0) albedo: vec4<f32>,
    @location(1) smra: vec4<f32>,
    @location(2) normal: vec4<f32>,
    @location(3) motion_emissive: vec4<f32>,
};

@group(1) @binding(0) var visibility_entity_texture: texture_2d<u32>;
@group(1) @binding(1) var visibility_surface_texture: texture_2d<u32>;
@group(1) @binding(2) var visibility_barycentric_texture: texture_2d<u32>;
@group(1) @binding(3) var depth_texture: texture_2d<f32>;
@group(1) @binding(4) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(5) var<storage, read> meshlets: array<MeshletRecord>;
@group(1) @binding(6) var<storage, read> meshlet_vertices: array<u32>;
@group(1) @binding(7) var<storage, read> meshlet_triangles: array<u32>;
@group(1) @binding(8) var<storage, read> material_params: array<StandardMaterialParams>;
@group(1) @binding(9) var<storage, read> material_table_offset: array<u32>;
@group(1) @binding(10) var<storage, read> material_palette: array<u32>;
@group(1) @binding(11) var texture_pool_albedo: texture_2d_array<f32>;
@group(1) @binding(12) var texture_pool_normal: texture_2d_array<f32>;
@group(1) @binding(13) var texture_pool_roughness: texture_2d_array<f32>;
@group(1) @binding(14) var texture_pool_metallic: texture_2d_array<f32>;
@group(1) @binding(15) var texture_pool_ao: texture_2d_array<f32>;
@group(1) @binding(16) var texture_pool_height: texture_2d_array<f32>;
@group(1) @binding(17) var texture_pool_specular: texture_2d_array<f32>;
@group(1) @binding(18) var texture_pool_emission: texture_2d_array<f32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var output: VertexOutput;
    output.position = vertex_position4(vertex_buffer[vi]);
    output.uv = vertex_uv(vertex_buffer[vi]);
    return output;
}

fn interpolate_vec2(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, bary: vec3<f32>) -> vec2<f32> {
    return a * bary.x + b * bary.y + c * bary.z;
}

fn interpolate_vec3(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, bary: vec3<f32>) -> vec3<f32> {
    return a * bary.x + b * bary.y + c * bary.z;
}

@fragment
fn fs(input: VertexOutput) -> FragmentOutput {
    let resolution = vec2<u32>(u32(frame_info.resolution.x), u32(frame_info.resolution.y));
    let pixel_coord = uv_to_coord(input.uv, resolution);

    let entity_id = textureLoad(visibility_entity_texture, pixel_coord, 0).x;
    if (entity_id == INVALID_IDX) {
        discard;
    }

    let surface = textureLoad(visibility_surface_texture, pixel_coord, 0).x;
    let bary_xy = unpack2x16unorm(textureLoad(visibility_barycentric_texture, pixel_coord, 0).x);
    let bary = vec3<f32>(bary_xy.x, bary_xy.y, max(1.0 - bary_xy.x - bary_xy.y, 0.0));

    let meshlet_index_value = unpack_surface_meshlet(surface);
    let triangle_index = unpack_surface_triangle(surface);
    let meshlet = meshlets[meshlet_index_value];

    let tri_offset = meshlet.triangle_offset + triangle_index * 3u;
    let local_index0 = meshlet_triangles[tri_offset + 0u];
    let local_index1 = meshlet_triangles[tri_offset + 1u];
    let local_index2 = meshlet_triangles[tri_offset + 2u];

    let global_index0 = meshlet_vertices[meshlet.vertex_offset + local_index0];
    let global_index1 = meshlet_vertices[meshlet.vertex_offset + local_index1];
    let global_index2 = meshlet_vertices[meshlet.vertex_offset + local_index2];

    let decoded0 = decode_vertex(vertex_buffer[global_index0]);
    let decoded1 = decode_vertex(vertex_buffer[global_index1]);
    let decoded2 = decode_vertex(vertex_buffer[global_index2]);

    let section_index = u32(max(decoded0.section_index, 0.0));
    let entity_transform = entity_transforms[entity_id];

    let local_position = interpolate_vec3(
        decoded0.position.xyz,
        decoded1.position.xyz,
        decoded2.position.xyz,
        bary
    );
    let prev_local_position = local_position;
    let uv = interpolate_vec2(decoded0.uv, decoded1.uv, decoded2.uv, bary);
    let normal_os = safe_normalize(interpolate_vec3(
        decoded0.normal.xyz,
        decoded1.normal.xyz,
        decoded2.normal.xyz,
        bary
    ));
    let tangent_os = safe_normalize(interpolate_vec3(
        decoded0.tangent.xyz,
        decoded1.tangent.xyz,
        decoded2.tangent.xyz,
        bary
    ));
    let bitangent_os = safe_normalize(interpolate_vec3(
        decoded0.bitangent.xyz,
        decoded1.bitangent.xyz,
        decoded2.bitangent.xyz,
        bary
    ));

    let world_position = entity_transform.transform * vec4<f32>(local_position, 1.0);
    let prev_world_position = entity_transform.prev_transform * vec4<f32>(prev_local_position, 1.0);

    let view_index = u32(frame_info.view_index);
    let view_proj = view_buffer[view_index].view_projection_matrix;
    let prev_view_proj = view_buffer[view_index].prev_projection_matrix * view_buffer[view_index].prev_view_matrix;
    let current_clip_pos = view_proj * world_position;
    let prev_clip_pos = prev_view_proj * prev_world_position;

    let normal_ws = safe_normalize((entity_transform.transform * vec4<f32>(normal_os, 0.0)).xyz);
    let tangent_ws = safe_normalize((entity_transform.transform * vec4<f32>(tangent_os, 0.0)).xyz);
    let bitangent_ws = safe_normalize((entity_transform.transform * vec4<f32>(bitangent_os, 0.0)).xyz);

    let entity_palette_offset = material_table_offset[entity_id];
    let material_index = material_palette[entity_palette_offset + section_index];
    let material = material_params[material_index];

    let tiling = material.emission_roughness_metallic_tiling.w;
    let base_uv = uv * tiling;
    let tex_size = vec2<f32>(textureDimensions(texture_pool_albedo).xy);
    let lod = compute_lod_from_uv(base_uv, tex_size);

    var sample_uv = base_uv;
    let height_flag = u32(material.texture_flags2.y);
    if ((height_flag & 1u) != 0u) {
        let view_dir = normalize(view_buffer[view_index].view_position.xyz - world_position.xyz);
        let tbn_matrix = mat3x3<f32>(tangent_ws, bitangent_ws, normal_ws);
        let view_tangent = normalize(tbn_matrix * view_dir);
        let height_scale = material.ao_height_specular.y;
        let height_value = sample_texture_or_float_param_handle(
            u32(material.height_handle),
            base_uv,
            0.0,
            height_flag,
            texture_pool_height,
            lod
        ) * height_scale - height_scale * 0.5;
        sample_uv = base_uv + view_tangent.xy * height_value / (view_tangent.z + 0.0001) * 0.05;
    }

    let albedo = sample_texture_or_vec4_param_handle(
        u32(material.albedo_handle),
        sample_uv,
        material.albedo,
        u32(material.texture_flags1.x),
        texture_pool_albedo,
        lod
    );
    let roughness = sample_texture_or_float_param_handle(
        u32(material.roughness_handle),
        sample_uv,
        material.emission_roughness_metallic_tiling.y,
        u32(material.texture_flags1.z),
        texture_pool_roughness,
        lod
    );
    let metallic = sample_texture_or_float_param_handle(
        u32(material.metallic_handle),
        sample_uv,
        material.emission_roughness_metallic_tiling.z,
        u32(material.texture_flags1.w),
        texture_pool_metallic,
        lod
    );
    let ao = sample_texture_or_float_param_handle(
        u32(material.ao_handle),
        sample_uv,
        material.ao_height_specular.x,
        u32(material.texture_flags2.x),
        texture_pool_ao,
        lod
    );
    let emissive = sample_texture_or_float_param_handle(
        u32(material.emission_handle),
        sample_uv,
        material.emission_roughness_metallic_tiling.x,
        u32(material.texture_flags2.w),
        texture_pool_emission,
        lod
    );
    let specular = sample_texture_or_float_param_handle(
        u32(material.specular_handle),
        sample_uv,
        material.ao_height_specular.z,
        u32(material.texture_flags2.z),
        texture_pool_specular,
        lod
    );

    var final_normal = normal_ws;
    if ((u32(material.texture_flags1.y) & 1u) != 0u) {
        let tbn_matrix = mat3x3<f32>(tangent_ws, bitangent_ws, normal_ws);
        let normal_sample =
            sample_handle_rgba(u32(material.normal_handle), sample_uv, texture_pool_normal, lod).xyz * 2.0 - 1.0;
        final_normal = safe_normalize(tbn_matrix * normal_sample);
    }

    let current_ndc = current_clip_pos.xy / current_clip_pos.w;
    let prev_ndc = prev_clip_pos.xy / prev_clip_pos.w;
    let motion = current_ndc - prev_ndc;

    var output: FragmentOutput;
    output.albedo = albedo;
    output.smra = vec4<f32>(specular, roughness, metallic, ao);
    output.normal = vec4<f32>(final_normal, 1.0);
    output.motion_emissive = vec4<f32>(motion, 0.0, emissive);
    return output;
}
