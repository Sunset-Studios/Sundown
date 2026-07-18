#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "postprocess_common.wgsl"
#include "sky_common.wgsl"
#include "gi/surface_cache_common.wgsl"

@group(1) @binding(0) var<uniform> surface_cache_params: SurfaceCacheParams;
@group(1) @binding(1) var<uniform> scene_lighting_data: SceneLightingData;
@group(1) @binding(2) var<storage, read_write> surface_cache: array<SurfacePatchReadOnly>;
@group(1) @binding(3) var<storage, read_write> surface_cache_sh: array<u32>;
@group(1) @binding(4) var<storage, read> active_indices: array<u32>;
@group(1) @binding(5) var<storage, read> counters: SurfaceCacheCountersReadOnly;
@group(1) @binding(6) var<storage, read_write> hit_info: array<SurfaceCacheHitInfo>;
@group(1) @binding(7) var<storage, read> material_params: array<StandardMaterialParams>;
@group(1) @binding(8) var<storage, read> material_table_offset: array<u32>;
@group(1) @binding(9) var<storage, read> material_palette: array<u32>;
@group(1) @binding(10) var<storage, read> entity_index_lookup: array<u32>;
@group(1) @binding(11) var<storage, read> dense_lights_buffer: DenseLightsBuffer;
@group(1) @binding(12) var texture_pool_albedo: texture_2d_array<f32>;
@group(1) @binding(13) var texture_pool_normal: texture_2d_array<f32>;
@group(1) @binding(14) var texture_pool_roughness: texture_2d_array<f32>;
@group(1) @binding(15) var texture_pool_metallic: texture_2d_array<f32>;
@group(1) @binding(16) var texture_pool_ao: texture_2d_array<f32>;
@group(1) @binding(17) var texture_pool_height: texture_2d_array<f32>;
@group(1) @binding(18) var texture_pool_specular: texture_2d_array<f32>;
@group(1) @binding(19) var texture_pool_emission: texture_2d_array<f32>;
@group(1) @binding(20) var skybox_texture: texture_cube<f32>;
@group(1) @binding(21) var<storage, read_write> radiance_info: array<SurfaceCacheRadianceInfo>;

#include "gi/surface_cache_lookup.wgsl"

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let active_index = gid.x;
    if (active_index >= counters.active_patch_count) {
        return;
    }

    let path = hit_info[active_index];
    let view_index = u32(frame_info.view_index);
    let camera_position = view_buffer[view_index].view_position.xyz;
    let light_view_index = u32(scene_lighting_data.view_index);
    let sun_direction = normalize(-view_buffer[light_view_index].view_direction.xyz);
    radiance_info[active_index].sample_radiance = vec4<f32>(0.0);
    hit_info[active_index].shadow_origin = vec4<f32>(0.0);
    hit_info[active_index].shadow_direction = vec4<f32>(0.0);
    radiance_info[active_index].shadow_radiance = vec4<f32>(0.0);

    if (path.ray_direction_primitive.w < 0.0) {
        let environment_radiance = evaluate_environment(
            path.ray_direction_primitive.xyz,
            sun_direction,
            scene_lighting_data,
            skybox_texture
        );
        radiance_info[active_index].sample_radiance = vec4<f32>(
            safe_clamp_vec3_max(environment_radiance, SURFACE_CACHE_MAX_RADIANCE),
            1.0
        );
        return;
    }

    let prim_store = u32(path.ray_direction_primitive.w);
    let entity_resolved = entity_index_lookup[prim_store];
    let palette_base = material_table_offset[entity_resolved];
    let material_index = material_palette[palette_base + u32(path.normal_section_index.w)];
    let material = material_params[material_index];
    let base_uv = vec2<f32>(path.hit_attr0.w, path.hit_attr1.w)
        * material.emission_roughness_metallic_tiling.w;
    let lod = 0.0;
    let albedo = sample_texture_or_vec4_param_handle(
        u32(material.albedo_handle), base_uv, material.albedo,
        u32(material.texture_flags1.x), texture_pool_albedo, lod
    ).xyz;
    let roughness = sample_texture_or_float_param_handle(
        u32(material.roughness_handle), base_uv, material.emission_roughness_metallic_tiling.y,
        u32(material.texture_flags1.z), texture_pool_roughness, lod
    );
    let metallic = sample_texture_or_float_param_handle(
        u32(material.metallic_handle), base_uv, material.emission_roughness_metallic_tiling.z,
        u32(material.texture_flags1.w), texture_pool_metallic, lod
    );
    let emissive = sample_texture_or_float_param_handle(
        u32(material.emission_handle), base_uv, material.emission_roughness_metallic_tiling.x,
        u32(material.texture_flags2.w), texture_pool_emission, lod
    );
    let reflectance = sample_texture_or_float_param_handle(
        u32(material.specular_handle), base_uv, material.ao_height_specular.z,
        u32(material.texture_flags2.z), texture_pool_specular, lod
    );

    let world_normal = safe_normalize(path.normal_section_index.xyz);
    var shading_normal = world_normal;
    let normal_flags = u32(material.texture_flags1.y);
    if ((normal_flags & NORMAL_TEXTURE_FLAG_PRESENT) != 0u) {
        let tangent_frame = mat3x3<f32>(path.hit_attr0.xyz, path.hit_attr1.xyz, world_normal);
        let normal_sample = decode_normal_texture_sample(
            sample_handle_rgba(u32(material.normal_handle), base_uv, texture_pool_normal, lod).xyz,
            normal_flags
        );
        shading_normal = normalize(tangent_frame * normal_sample);
    }

    let hit_position = path.hit_position_sampling_weight.xyz;
    let view_direction = normalize(-path.ray_direction_primitive.xyz);
    let recurrent_irradiance = surface_cache_sample(
        hit_position,
        shading_normal,
        camera_position
    ).xyz;
    radiance_info[active_index].sample_radiance = vec4<f32>(
        safe_clamp_vec3_max(
            emissive * albedo + recurrent_irradiance,
            SURFACE_CACHE_MAX_RADIANCE
        ),
        1.0
    );

    let light_count = dense_lights_buffer.header.light_count;
    if (light_count == 0u) {
        return;
    }

    let patch_index = active_indices[active_index];
    var rng = random_seed(surface_cache_patch_rng(
        patch_index,
        surface_cache[patch_index].fingerprint
    ) ^ hash(
        (u32(surface_cache[patch_index].history.y) ^
        u32(surface_cache_params.frame_index)) * 0x9e3779b9u
    ));
    let selected_light_index = u32(rand_float(rng) * f32(light_count)) % light_count;
    let light = dense_lights_buffer.lights[selected_light_index];
    let light_direction = get_light_dir(light, hit_position);
    let attenuation = get_light_attenuation(light, hit_position);
    let brdf = calculate_brdf_rt(
        shading_normal, view_direction, light_direction, albedo, roughness,
        metallic, reflectance, 0.0, 0.0
    );
    let direct_radiance = brdf * light.color.rgb * light.intensity * attenuation
        * f32(light_count);
    let light_distance = select(
        surface_cache_params.max_ray_length,
        length(light.position.xyz - hit_position),
        light.light_type != 0.0
    );
    hit_info[active_index].shadow_origin = vec4<f32>(
        hit_position + shading_normal * 0.001,
        0.0001
    );
    hit_info[active_index].shadow_direction = vec4<f32>(
        light_direction,
        min(light_distance * 0.999, surface_cache_params.max_ray_length)
    );
    // w = 1 means pending visibility; the shadow pass changes it to 2 only
    // when the light is visible.
    radiance_info[active_index].shadow_radiance = vec4<f32>(
        safe_clamp_vec3_max(direct_radiance, SURFACE_CACHE_MAX_RADIANCE),
        1.0
    );
}
