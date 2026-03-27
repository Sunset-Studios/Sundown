#define CUSTOM_DEPTH_FRAGMENT_MASK
#define CUSTOM_RASTER_FRAGMENT_MASK
#define CUSTOM_RESOLVE_FRAGMENT

#include "visibility/visibility_draw_base.wgsl"

#if MESHLET_RESOLVE_PASS
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
#else
@group(2) @binding(0) var<storage, read> material_params: array<StandardMaterialParams>;
@group(2) @binding(1) var<storage, read> material_table_offset: array<u32>;
@group(2) @binding(2) var<storage, read> material_palette: array<u32>;
@group(2) @binding(3) var texture_pool_albedo: texture_2d_array<f32>;
#endif

fn resolve_material(entity_id: u32, section_index: u32) -> StandardMaterialParams {
    let entity_palette_offset = material_table_offset[entity_id];
    let material_id = material_palette[entity_palette_offset + section_index];
    return material_params[material_id];
}

fn fragment_mask(entity_id: u32, section_index: u32, uv: vec2<f32>) -> f32 {
    let material = resolve_material(entity_id, section_index);
    let base_uv = uv * material.emission_roughness_metallic_tiling.w;
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

fn depth_fragment_mask(input: DepthVertexOutput) -> f32 {
    return fragment_mask(input.entity_id, input.section_index, input.uv);
}

fn raster_fragment_mask(input: RasterVertexOutput) -> f32 {
    return fragment_mask(input.entity_id, input.section_index, input.uv);
}

fn resolve_fragment(
    input: ResolveFragmentInput,
    f_out: ptr<function, ResolveFragmentOutput>
) -> ResolveFragmentOutput {
#if MESHLET_RESOLVE_PASS
    let material = resolve_material(input.entity_id, input.section_index);

    let tiling = material.emission_roughness_metallic_tiling.w;
    let base_uv = input.uv * tiling;
    let tex_size = vec2<f32>(textureDimensions(texture_pool_albedo).xy);
    let lod = compute_lod_from_uv(base_uv, tex_size);

    let view_index = u32(frame_info.view_index);
    var sample_uv = base_uv;
    let height_flag = u32(material.texture_flags2.y);
    if ((height_flag & 1u) != 0u) {
        let view_dir = normalize(view_buffer[view_index].view_position.xyz - input.world_position.xyz);
        let tbn_matrix = mat3x3<f32>(
            input.tangent.xyz,
            input.bitangent.xyz,
            input.normal.xyz
        );
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

    if ((u32(material.texture_flags1.y) & 1u) != 0u) {
        let tbn_matrix = mat3x3<f32>(
            input.tangent.xyz,
            input.bitangent.xyz,
            input.normal.xyz
        );
        let normal_sample =
            sample_handle_rgba(u32(material.normal_handle), sample_uv, texture_pool_normal, lod).xyz * 2.0 - 1.0;
        f_out.normal = vec4<f32>(safe_normalize(tbn_matrix * normal_sample), 1.0);
    }

    f_out.albedo = albedo;
    f_out.smra = vec4<f32>(specular, roughness, metallic, ao);
    f_out.motion_emissive.a = emissive;
#endif

    return *f_out;
}
