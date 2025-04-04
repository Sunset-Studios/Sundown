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

fn sample_texture_or_vec4_param(
    texture: texture_2d<precision_float>,
    uv: vec2<precision_float>,
    param: vec4<precision_float>,
    use_texture: u32
) -> vec4<precision_float> {
    if (use_texture != 0u) {
        return textureSample(texture, global_sampler, uv);
    }
    return param;
}

fn sample_texture_or_float_param(
    texture: texture_2d<precision_float>,
    uv: vec2<precision_float>,
    param: precision_float,
    use_texture: u32
) -> precision_float {
    if (use_texture != 0u) {
        return textureSample(texture, global_sampler, uv).r;
    }
    return param;
}

// ------------------------------------------------------------------------------------
// Fragment Shader
// ------------------------------------------------------------------------------------ 
fn fragment(v_out: VertexOutput, f_out: ptr<function, FragmentOutput>) -> FragmentOutput {
    let tiling = material_params.emission_roughness_metallic_tiling.w;
    var uv = v_out.uv * tiling;
    
    let albedo = sample_texture_or_vec4_param(
        albedo,
        uv,
        material_params.albedo,
        material_params.texture_flags1.x
    );
    let roughness = sample_texture_or_float_param(
        roughness,
        uv,
        material_params.emission_roughness_metallic_tiling.y,
        material_params.texture_flags1.z
    );
    let metallic = sample_texture_or_float_param(
        metallic,
        uv,
        material_params.emission_roughness_metallic_tiling.z,
        material_params.texture_flags1.w
    );
    let ao = sample_texture_or_float_param(
        ao,
        uv,
        material_params.ao_height_specular.x,
        material_params.texture_flags2.x
    );
    let emissive = sample_texture_or_float_param(
        emission,
        uv,
        material_params.emission_roughness_metallic_tiling.x,
        material_params.texture_flags2.w
    );
    
    // Apply normal mapping if enabled
    if (material_params.texture_flags1.y != 0u) {
        let tbn_matrix = mat3x3<precision_float>(
            v_out.tangent.xyz,
            v_out.bitangent.xyz,
            v_out.normal.xyz
        );
        let normal_map = get_normal_from_normal_map(
            normal,
            uv,
            tbn_matrix
        );
        f_out.normal = vec4<precision_float>(normal_map, 1.0);
    }
    
    f_out.albedo = albedo;
    f_out.smra.r = material_params.ao_height_specular.z;
    f_out.smra.g = roughness;
    f_out.smra.b = metallic;
    f_out.smra.a = ao;
    f_out.emissive.r = emissive;
    
    return *f_out;
}