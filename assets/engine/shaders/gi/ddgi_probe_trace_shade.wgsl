// =============================================================================
// DDGI Probe Ray Trace - Shade Pass
// - Shades the hit attributes written by ddgi_probe_trace_hit
// - Evaluates sky/environment for ray misses
// - Queries the world cache for cached indirect radiance
// - Intentionally avoids ALL BVH bindings to reduce binding count
// =============================================================================
#include "common.wgsl"
#include "postprocess_common.wgsl"
#include "sky_common.wgsl"
#include "gi/ddgi_common.wgsl"

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<uniform> scene_lighting_data: SceneLightingData;
@group(1) @binding(2) var<storage, read_write> probe_ray_data: DDGIProbeRayDataBufferReadOnlyHeader;
@group(1) @binding(3) var<storage, read_write> probe_states: array<ProbeStateData>;
@group(1) @binding(4) var<storage, read> material_params: array<StandardMaterialParams>;
@group(1) @binding(5) var<storage, read> material_table_offset: array<u32>;
@group(1) @binding(6) var<storage, read> material_palette: array<u32>;
@group(1) @binding(7) var<storage, read_write> sh_probes: array<u32>;
@group(1) @binding(8) var<storage, read> probe_depth_moments: array<u32>;
@group(1) @binding(9) var texture_pool_albedo: texture_2d_array<f32>;
@group(1) @binding(10) var texture_pool_normal: texture_2d_array<f32>;
@group(1) @binding(11) var texture_pool_roughness: texture_2d_array<f32>;
@group(1) @binding(12) var texture_pool_metallic: texture_2d_array<f32>;
@group(1) @binding(13) var texture_pool_ao: texture_2d_array<f32>;
@group(1) @binding(14) var texture_pool_height: texture_2d_array<f32>;
@group(1) @binding(15) var texture_pool_specular: texture_2d_array<f32>;
@group(1) @binding(16) var texture_pool_emission: texture_2d_array<f32>;
@group(1) @binding(17) var skybox_texture: texture_cube<f32>;

const EMISSIVE_HIT_LUMA_SOFT_CAP: f32 = 2.0;
const EMISSIVE_HIT_OVERFLOW_SCALE: f32 = 0.1;

fn stabilize_emissive_hit_radiance(raw_emissive_radiance: vec3<f32>) -> vec3<f32> {
    let clamped_emissive_radiance = safe_clamp_vec3_max(raw_emissive_radiance, MAX_RADIANCE_LUMINANCE);
    let emissive_luma = max(luminance(clamped_emissive_radiance), 1e-6);
    let compressed_luma = select(
        emissive_luma,
        EMISSIVE_HIT_LUMA_SOFT_CAP + (emissive_luma - EMISSIVE_HIT_LUMA_SOFT_CAP) * EMISSIVE_HIT_OVERFLOW_SCALE,
        emissive_luma > EMISSIVE_HIT_LUMA_SOFT_CAP
    );
    let emissive_scale = compressed_luma / emissive_luma;
    return clamped_emissive_radiance * emissive_scale;
}

@compute @workgroup_size(256, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let active_ray_count = probe_ray_data.header.active_ray_count;

    if (gid.x >= active_ray_count || probe_ray_data.rays[gid.x].state_u32.y == 0u) {
        return;
    }

    let hit = probe_ray_data.rays[gid.x];
    let ray_dir = hit.ray_dir_prim.xyz;
    let ray_index_in_probe = hit.meta_u32.y;
    let probe_index = hit.meta_u32.x;

    let light_view_index = u32(scene_lighting_data.view_index);
    let sun_dir = normalize(-view_buffer[light_view_index].view_direction.xyz);

    // Backface leak reduction:
    // - Backface hits are encoded with negative t values.
    // - We record 0 radiance to avoid lighting surfaces that should be shadowed.
    if (hit.hit_pos_t.w < 0.0) {
        probe_ray_data.rays[gid.x].radiance = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Backface hits are encoded as negative t values in hit_pos_t.w.
    let is_miss = hit.state_u32.w == 0xffffffffu;
    if (is_miss) {
        // Ray miss: evaluate environment radiance.
        let env_radiance = evaluate_environment(ray_dir, sun_dir, scene_lighting_data, skybox_texture);
        probe_ray_data.rays[gid.x].radiance = vec4<f32>(safe_clamp_vec3_max(env_radiance, MAX_RADIANCE_LUMINANCE), 1.0);
    }
    
    if (!is_miss) {
        // Ray hit: shade the hit.
        let prim_store = hit.state_u32.x;
        let section_index = u32(hit.world_n_section.w);

        let entity_palette_base = material_table_offset[prim_store];
        let mat_params_index = material_palette[entity_palette_base + section_index];
        let material = material_params[mat_params_index];

        let tiling = material.emission_roughness_metallic_tiling.w;
        let base_uv = vec2<f32>(hit.world_t_uvx.w, hit.world_b_uvy.w) * tiling;
        let lod = 0.0;

        let albedo = sample_texture_or_vec4_param_handle(
            u32(material.albedo_handle), base_uv, material.albedo,
            u32(material.texture_flags1.x), texture_pool_albedo, lod
        ).xyz;
        let roughness = sample_texture_or_float_param_handle(
            u32(material.roughness_handle), base_uv,
            material.emission_roughness_metallic_tiling.y,
            u32(material.texture_flags1.z), texture_pool_roughness, lod
        );
        let metallic = sample_texture_or_float_param_handle(
            u32(material.metallic_handle), base_uv,
            material.emission_roughness_metallic_tiling.z,
            u32(material.texture_flags1.w), texture_pool_metallic, lod
        );
        let emissive = sample_texture_or_float_param_handle(
            u32(material.emission_handle), base_uv,
            material.emission_roughness_metallic_tiling.x,
            u32(material.texture_flags2.w), texture_pool_emission, lod
        );
        let reflectance = sample_texture_or_float_param_handle(
            u32(material.specular_handle), base_uv,
            material.ao_height_specular.z,
            u32(material.texture_flags2.z), texture_pool_specular, lod
        );

        let world_n = hit.world_n_section.xyz;
        let world_t = hit.world_t_uvx.xyz;
        let world_b = hit.world_b_uvy.xyz;

        var n = world_n;
        if ((u32(material.texture_flags1.y) & 1u) != 0u) {
            let tbn = mat3x3<f32>(world_t, world_b, world_n);
            let nm = sample_handle_rgba(
                u32(material.normal_handle), base_uv,
                texture_pool_normal, lod
            ).xyz * 2.0 - 1.0;
            n = normalize(tbn * nm);
        }

        // -----------------------------------------------------------------------------
        // Direct lighting seed (NEE at the PRIMARY HIT POINT)
        // - Hit pass chooses one analytic/emissive candidate light and computes
        //   its radiance scale + direction.
        // - This pass only applies visibility and the shaded normal response.
        // -----------------------------------------------------------------------------
        var radiance = vec3<f32>(0.0);
        let light_dir = hit.nee_light_dir_type.xyz;
        let visibility = select(0.0, 1.0, hit.state_u32.z == 1u);
        radiance += safe_clamp_vec3_max(hit.nee_light_radiance.xyz * visibility, MAX_RADIANCE_LUMINANCE);

        // Reseed multi-bounce using last frame's DDGI SH field.
        // Treat SH as incident diffuse irradiance at the hit point.
        let sh_irradiance = ddgi_sample_sh_irradiance_with_states(
            &ddgi_params,
            &sh_probes,
            &probe_states,
            &probe_depth_moments,
            hit.hit_pos_t.xyz,
            n
        );
        radiance += safe_clamp_vec3_max(sh_irradiance, MAX_RADIANCE_LUMINANCE);

        // Apply Lambertian BRDF to reflected light (NEE + multi-bounce SH)
        radiance *= albedo * (1.0 / (2.0 * PI));

        if (emissive > 0.0) {
            radiance += stabilize_emissive_hit_radiance(emissive * albedo);
        }

        probe_ray_data.rays[gid.x].radiance = vec4<f32>(radiance, 1.0);
    }
}
