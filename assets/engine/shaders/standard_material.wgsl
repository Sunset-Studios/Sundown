#define CUSTOM_FS

#include "gbuffer_base.wgsl"

struct MaterialParams {
    albedo: vec4<precision_float>,
    normal: vec4<precision_float>,
    emission_roughness_metallic_tiling: vec4<precision_float>,
    ao_height_specular: vec4<precision_float>,
    texture_flags1: vec4<u32>, // x: albedo, y: normal, z: roughness, w: metallic
    texture_flags2: vec4<u32>, // x: ao, y: height, z: specular, w: emission 
}

@group(2) @binding(0) var<uniform> material_params: MaterialParams;
@group(2) @binding(1) var albedo: texture_2d<precision_float>;
@group(2) @binding(2) var normal: texture_2d<precision_float>;
@group(2) @binding(3) var roughness: texture_2d<precision_float>;
@group(2) @binding(4) var metallic: texture_2d<precision_float>;
@group(2) @binding(5) var emission: texture_2d<precision_float>;
@group(2) @binding(6) var ao: texture_2d<precision_float>;
@group(2) @binding(7) var height_tex: texture_2d<precision_float>;
@group(2) @binding(8) var specular_tex: texture_2d<precision_float>;

fn sample_texture_or_vec4_param(
    tex: texture_2d<precision_float>,
    uv_coords: vec2<precision_float>,
    param_val: vec4<precision_float>,
    flag: u32
) -> vec4<precision_float> {
    if ((flag & 1u) != 0u) {
        return textureSample(tex, global_sampler, uv_coords);
    }
    return param_val;
}

fn sample_texture_or_float_param(
    tex: texture_2d<precision_float>,
    uv_coords: vec2<precision_float>,
    param_val: precision_float,
    flag: u32
) -> precision_float {
    if ((flag & 1u) != 0u) {
        let sampled_val = textureSample(tex, global_sampler, uv_coords);
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
        let height_value = sample_texture_or_float_param(height_tex, base_uv, 0.0, height_flag) * height_scale - height_scale * 0.5;
        let parallax_offset = view_tangent.xy * height_value / (view_tangent.z + 0.0001) * 0.05; // Fixed scale 0.05
        sample_uv = base_uv + parallax_offset;
    }
    
    let albedo = sample_texture_or_vec4_param(
        albedo,
        sample_uv,
        material_params.albedo,
        material_params.texture_flags1.x
    );
    let roughness = sample_texture_or_float_param(
        roughness,
        sample_uv,
        material_params.emission_roughness_metallic_tiling.y,
        material_params.texture_flags1.z
    );
    let metallic = sample_texture_or_float_param(
        metallic,
        sample_uv,
        material_params.emission_roughness_metallic_tiling.z,
        material_params.texture_flags1.w
    );
    let ao = sample_texture_or_float_param(
        ao,
        sample_uv,
        material_params.ao_height_specular.x,
        material_params.texture_flags2.x
    );
    let emissive = sample_texture_or_float_param(
        emission,
        sample_uv,
        material_params.emission_roughness_metallic_tiling.x,
        material_params.texture_flags2.w
    );
    let specular = sample_texture_or_float_param(
        specular_tex,
        sample_uv,
        material_params.ao_height_specular.z,
        material_params.texture_flags2.z
    );
    
    // Apply normal mapping if enabled
    if ((material_params.texture_flags1.y & 1u) != 0u) {
        let tbn_matrix = mat3x3<precision_float>(
            v_out.tangent.xyz,
            v_out.bitangent.xyz,
            v_out.normal.xyz
        );
        let normal_map = get_normal_from_normal_map(
            normal,
            sample_uv,
            tbn_matrix
        );
        f_out.normal = vec4<precision_float>(normal_map, 1.0);
    }
    
    f_out.albedo = albedo;
    f_out.smra.r = specular;
    f_out.smra.g = roughness;
    f_out.smra.b = metallic;
    f_out.smra.a = ao;
    f_out.emissive.r = emissive;
    
    return *f_out;
}