#define CUSTOM_FS
#define MASKED

#include "gbuffer_base.wgsl"

@group(2) @binding(0) var<storage, read> material_params: array<StandardMaterialParams>;
@group(2) @binding(1) var<storage, read> material_table_offset: array<u32>;
@group(2) @binding(2) var<storage, read> material_palette: array<u32>;
@group(2) @binding(3) var texture_pool_albedo: texture_2d_array<f32>;
@group(2) @binding(4) var texture_pool_normal: texture_2d_array<f32>;
@group(2) @binding(5) var texture_pool_roughness: texture_2d_array<f32>;
@group(2) @binding(6) var texture_pool_metallic: texture_2d_array<f32>;
@group(2) @binding(7) var texture_pool_ao: texture_2d_array<f32>;
@group(2) @binding(8) var texture_pool_height: texture_2d_array<f32>;
@group(2) @binding(9) var texture_pool_specular: texture_2d_array<f32>;
@group(2) @binding(10) var texture_pool_emission: texture_2d_array<f32>;

// ------------------------------------------------------------------------------------
// Fragment Shader
// ------------------------------------------------------------------------------------ 
fn fragment(v_out: VertexOutput, f_out: ptr<function, FragmentOutput>) -> FragmentOutput {
    let section_index = u32(vertex_section_index(vertex_buffer[v_out.vertex_index]));
    let entity_palette_offset = material_table_offset[v_out.instance_id];
    let material_params_index = material_palette[entity_palette_offset + section_index];
    let material_params = material_params[material_params_index];

    let tiling = material_params.emission_roughness_metallic_tiling.w;
    var base_uv = v_out.uv * tiling;

    let tex_size = vec2<f32>(textureDimensions(texture_pool_albedo).xy);
    let lod = compute_lod_from_uv(base_uv, tex_size);
    
    // Simple parallax offset
    var sample_uv = base_uv;
    let height_flag = u32(material_params.texture_flags2.y);
    if ((height_flag & 1u) != 0u) {
        let view_data = view_buffer[u32(frame_info.view_index)];
        let view_dir = normalize(view_data.view_position.xyz - v_out.world_position.xyz);
        let tbn_matrix = mat3x3<f32>(
            v_out.tangent.xyz,
            v_out.bitangent.xyz,
            v_out.normal.xyz
        );
        let view_tangent = normalize(tbn_matrix * view_dir);
        let height_scale = material_params.ao_height_specular.y;
        let height_value = sample_texture_or_float_param_handle(
            u32(material_params.height_handle),
            base_uv,
            0.0,
            height_flag,
            texture_pool_height,
            lod
        ) * height_scale - height_scale * 0.5;
        let parallax_offset = view_tangent.xy * height_value / (view_tangent.z + 0.0001) * 0.05; // Fixed scale 0.05
        sample_uv = base_uv + parallax_offset;
    }
    
    let albedo = sample_texture_or_vec4_param_handle(
        u32(material_params.albedo_handle),
        sample_uv,
        material_params.albedo,
        u32(material_params.texture_flags1.x),
        texture_pool_albedo,
        lod
    );
    let roughness = sample_texture_or_float_param_handle(
        u32(material_params.roughness_handle),
        sample_uv,
        material_params.emission_roughness_metallic_tiling.y,
        u32(material_params.texture_flags1.z),
        texture_pool_roughness,
        lod
    );
    let metallic = sample_texture_or_float_param_handle(
        u32(material_params.metallic_handle),
        sample_uv,
        material_params.emission_roughness_metallic_tiling.z,
        u32(material_params.texture_flags1.w),
        texture_pool_metallic,
        lod
    );
    let ao = sample_texture_or_float_param_handle(
        u32(material_params.ao_handle),
        sample_uv,
        material_params.ao_height_specular.x,
        u32(material_params.texture_flags2.x),
        texture_pool_ao,
        lod
    );
    let emissive = sample_texture_or_float_param_handle(
        u32(material_params.emission_handle),
        sample_uv,
        material_params.emission_roughness_metallic_tiling.x,
        u32(material_params.texture_flags2.w),
        texture_pool_emission,
        lod
    );
    let specular = sample_texture_or_float_param_handle(
        u32(material_params.specular_handle),
        sample_uv,
        material_params.ao_height_specular.z,
        u32(material_params.texture_flags2.z),
        texture_pool_specular,
        lod
    );
    
    // Apply normal mapping if enabled
    if ((u32(material_params.texture_flags1.y) & 1u) != 0u) {
        let tbn_matrix = mat3x3<f32>(
            v_out.tangent.xyz,
            v_out.bitangent.xyz,
            v_out.normal.xyz
        );
        let nm_sample = sample_handle_rgba(u32(material_params.normal_handle), sample_uv, texture_pool_normal, lod).xyz * 2.0 - 1.0;
        let normal_map_vec = normalize(tbn_matrix * nm_sample);
        f_out.normal = vec4<f32>(normal_map_vec, 1.0);
    }
    
    f_out.albedo = albedo;
    f_out.smra.r = specular;
    f_out.smra.g = roughness;
    f_out.smra.b = metallic;
    f_out.smra.a = ao;
    f_out.motion_emissive.a = emissive;
    
    return *f_out;
}

fn fragment_mask(v_out: VertexOutput) -> f32 {
    let section_index = u32(vertex_section_index(vertex_buffer[v_out.vertex_index]));
    let entity_palette_offset = material_table_offset[v_out.instance_id];
    let material_params_index = material_palette[entity_palette_offset + section_index];
    let material_params = material_params[material_params_index];

    let tiling = material_params.emission_roughness_metallic_tiling.w;
    var base_uv = v_out.uv * tiling;

    let tex_size = vec2<f32>(textureDimensions(texture_pool_albedo).xy);
    let lod = compute_lod_from_uv(base_uv, tex_size);

    // Simple parallax offset
    var sample_uv = base_uv;

    let albedo = sample_texture_or_vec4_param_handle(
        u32(material_params.albedo_handle),
        sample_uv,
        material_params.albedo,
        u32(material_params.texture_flags1.x),
        texture_pool_albedo,
        lod
    );

    return albedo.a;
}
