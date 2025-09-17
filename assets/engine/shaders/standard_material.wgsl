#define CUSTOM_FS

#include "gbuffer_base.wgsl"

@group(2) @binding(0) var<uniform> material_params: StandardMaterialParams;
@group(2) @binding(1) var texture_pool_albedo: texture_2d_array<f32>;
@group(2) @binding(2) var texture_pool_normal: texture_2d_array<f32>;
@group(2) @binding(3) var texture_pool_roughness: texture_2d_array<f32>;
@group(2) @binding(4) var texture_pool_metallic: texture_2d_array<f32>;
@group(2) @binding(5) var texture_pool_ao: texture_2d_array<f32>;
@group(2) @binding(6) var texture_pool_height: texture_2d_array<f32>;
@group(2) @binding(7) var texture_pool_specular: texture_2d_array<f32>;
@group(2) @binding(8) var texture_pool_emission: texture_2d_array<f32>;

fn sample_texture_or_vec4_param_handle(
    tex_handle: u32,
    uv_coords: vec2<precision_float>,
    param_val: vec4<precision_float>,
    flag: u32,
    pool: texture_2d_array<f32>
) -> vec4<precision_float> {
    if ((flag & 1u) != 0u) {
        return sample_handle_rgba(tex_handle, uv_coords, pool);
    }
    return param_val;
}

fn sample_texture_or_float_param_handle(
    tex_handle: u32,
    uv_coords: vec2<precision_float>,
    param_val: precision_float,
    flag: u32,
    pool: texture_2d_array<f32>
) -> precision_float {
    if ((flag & 1u) != 0u) {
        let sampled_val = sample_handle_rgba(tex_handle, uv_coords, pool);
        let channel_index = (flag >> 1u) & 3u;
        return select(select(select(sampled_val.r, sampled_val.g, channel_index == 1u), sampled_val.b, channel_index == 2u), sampled_val.a, channel_index == 3u);
    }
    return param_val;
}

// ------------------------------------------------------------------------------------
// Fragment Shader
// ------------------------------------------------------------------------------------ 
fn fragment(v_out: VertexOutput, f_out: ptr<function, FragmentOutput>) -> FragmentOutput {
    let tiling = material_params.emission_roughness_metallic_tiling.w;
    var base_uv = v_out.uv * tiling;
    
    // Simple parallax offset
    var sample_uv = base_uv;
    let height_flag = material_params.texture_flags2.y;
    if ((height_flag & 1u) != 0u) {
        let view_data = view_buffer[u32(frame_info.view_index)];
        let view_dir = normalize(view_data.view_position.xyz - v_out.world_position.xyz);
        let tbn_matrix = mat3x3<precision_float>(
            v_out.tangent.xyz,
            v_out.bitangent.xyz,
            v_out.normal.xyz
        );
        let view_tangent = normalize(tbn_matrix * view_dir);
        let height_scale = material_params.ao_height_specular.y;
        let height_value = sample_texture_or_float_param_handle(
            material_params.height_handle,
            base_uv,
            0.0,
            height_flag,
            texture_pool_height) * height_scale - height_scale * 0.5;
        let parallax_offset = view_tangent.xy * height_value / (view_tangent.z + 0.0001) * 0.05; // Fixed scale 0.05
        sample_uv = base_uv + parallax_offset;
    }
    
    let albedo = sample_texture_or_vec4_param_handle(
        material_params.albedo_handle,
        sample_uv,
        material_params.albedo,
        material_params.texture_flags1.x,
        texture_pool_albedo
    );
    let roughness = sample_texture_or_float_param_handle(
        material_params.roughness_handle,
        sample_uv,
        material_params.emission_roughness_metallic_tiling.y,
        material_params.texture_flags1.z,
        texture_pool_roughness
    );
    let metallic = sample_texture_or_float_param_handle(
        material_params.metallic_handle,
        sample_uv,
        material_params.emission_roughness_metallic_tiling.z,
        material_params.texture_flags1.w,
        texture_pool_metallic
    );
    let ao = sample_texture_or_float_param_handle(
        material_params.ao_handle,
        sample_uv,
        material_params.ao_height_specular.x,
        material_params.texture_flags2.x,
        texture_pool_ao
    );
    let emissive = sample_texture_or_float_param_handle(
        material_params.emission_handle,
        sample_uv,
        material_params.emission_roughness_metallic_tiling.x,
        material_params.texture_flags2.w,
        texture_pool_emission
    );
    let specular = sample_texture_or_float_param_handle(
        material_params.specular_handle,
        sample_uv,
        material_params.ao_height_specular.z,
        material_params.texture_flags2.z,
        texture_pool_specular
    );
    
    // Apply normal mapping if enabled
    if ((material_params.texture_flags1.y & 1u) != 0u) {
        let tbn_matrix = mat3x3<precision_float>(
            v_out.tangent.xyz,
            v_out.bitangent.xyz,
            v_out.normal.xyz
        );
        let nm_sample = sample_handle_rgba(material_params.normal_handle, sample_uv, texture_pool_normal).xyz * 2.0 - 1.0;
        let normal_map_vec = normalize(tbn_matrix * nm_sample);
        f_out.normal = vec4<precision_float>(normal_map_vec, 1.0);
    }
    
    f_out.albedo = albedo;
    f_out.smra.r = specular;
    f_out.smra.g = roughness;
    f_out.smra.b = metallic;
    f_out.smra.a = ao;
    f_out.emissive.r = emissive;
    
    return *f_out;
}