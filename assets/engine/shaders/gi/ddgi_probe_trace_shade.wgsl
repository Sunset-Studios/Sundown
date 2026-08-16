// =============================================================================
// DDGI Probe Ray Trace - Shade Pass
// - Shades the hit attributes written by ddgi_probe_trace_hit
// - Evaluates sky/environment for ray misses
// - Queries the surface cache for cached indirect radiance
// - Intentionally avoids ALL BVH bindings to reduce binding count
// =============================================================================
#include "common.wgsl"
#include "postprocess_common.wgsl"
#include "sky_common.wgsl"
#include "gi/ddgi_common.wgsl"

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<uniform> scene_lighting_data: SceneLightingData;
@group(1) @binding(2) var<storage, read_write> probe_ray_data: DDGIProbeRayDataBufferReadOnlyHeader;
@group(1) @binding(3) var<storage, read> probe_update_indices: array<u32>;
@group(1) @binding(4) var<storage, read_write> probe_states: array<ProbeStateData>;
@group(1) @binding(5) var<storage, read> material_params: array<StandardMaterialParams>;
@group(1) @binding(6) var<storage, read> material_table_offset: array<u32>;
@group(1) @binding(7) var<storage, read> material_palette: array<u32>;
@group(1) @binding(8) var<storage, read_write> sh_probes: array<u32>;
@group(1) @binding(9) var<storage, read> probe_depth_moments: array<u32>;
@group(1) @binding(10) var<storage, read> entity_index_lookup: array<u32>;
@group(1) @binding(11) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(12) var texture_pool_albedo: texture_2d_array<f32>;
@group(1) @binding(13) var texture_pool_normal: texture_2d_array<f32>;
@group(1) @binding(14) var texture_pool_roughness: texture_2d_array<f32>;
@group(1) @binding(15) var texture_pool_metallic: texture_2d_array<f32>;
@group(1) @binding(16) var texture_pool_ao: texture_2d_array<f32>;
@group(1) @binding(17) var texture_pool_height: texture_2d_array<f32>;
@group(1) @binding(18) var texture_pool_specular: texture_2d_array<f32>;
@group(1) @binding(19) var texture_pool_emission: texture_2d_array<f32>;
@group(1) @binding(20) var skybox_texture: texture_cube<f32>;
@group(1) @binding(21) var<storage, read> probe_depth_slots: array<u32>;

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

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let active_ray_count = probe_ray_data.header.active_ray_count;

    if (gid.x >= active_ray_count) {
        return;
    }

    let hit = probe_ray_data.rays[gid.x];
    let ray_dir = ddgi_probe_ray_stored_direction(hit);
    let rays_per_probe = ddgi_max_rays_per_probe(&ddgi_params);
    let probe_slot = gid.x / rays_per_probe;
    let probe_index = probe_update_indices[probe_slot];

    let light_view_index = u32(scene_lighting_data.view_index);
    let sun_dir = normalize(-view_buffer[light_view_index].view_direction.xyz);

    // Backface leak reduction:
    // - Backface hits are encoded with negative t values.
    // - We record 0 radiance to avoid lighting surfaces that should be shadowed.
    if (hit.hit_distance < 0.0) {
        ddgi_probe_ray_set_radiance(&probe_ray_data.rays[gid.x], vec3<f32>(0.0));
        return;
    }

    let is_miss = hit.prim_store == INVALID_IDX;
    if (is_miss) {
        // Ray miss: evaluate environment radiance.
        let env_radiance = evaluate_environment(ray_dir, sun_dir, scene_lighting_data, skybox_texture);
        ddgi_probe_ray_set_radiance(
            &probe_ray_data.rays[gid.x],
            safe_clamp_vec3_max(env_radiance, MAX_RADIANCE_LUMINANCE)
        );
    }
    
    if (!is_miss) {
        // Ray hit: shade the hit.
        let prim_store = hit.prim_store;
        let entity_resolved = entity_index_lookup[prim_store];
        let entity_transform = entity_transforms[entity_resolved];
        let vertex0 = decode_vertex(vertex_buffer[hit.vertex_index_0]);
        let vertex1 = decode_vertex(vertex_buffer[hit.vertex_index_1]);
        let vertex2 = decode_vertex(vertex_buffer[hit.vertex_index_2]);
        let u_bc = hit.barycentric_u;
        let v_bc = hit.barycentric_v;
        let w_bc = 1.0 - u_bc - v_bc;
        let section_index = u32(vertex0.section_index);

        let entity_palette_base = material_table_offset[entity_resolved];
        let mat_params_index = material_palette[entity_palette_base + section_index];
        let material = material_params[mat_params_index];

        let tiling = material.emission_roughness_metallic_tiling.w;
        let base_uv = (
            vertex0.uv * w_bc +
            vertex1.uv * u_bc +
            vertex2.uv * v_bc
        ) * tiling;
        let lod = 0.0;

        let albedo = sample_texture_or_vec4_param_handle(
            u32(material.albedo_handle), base_uv, material.albedo,
            u32(material.texture_flags1.x), texture_pool_albedo, lod
        ).xyz;
        let emissive = sample_emission_handle(
            u32(material.emission_handle), base_uv,
            material.emission_roughness_metallic_tiling.x,
            u32(material.texture_flags2.w), texture_pool_emission, lod
        );

        let n_local = vertex0.normal.xyz * w_bc +
            vertex1.normal.xyz * u_bc +
            vertex2.normal.xyz * v_bc;
        let world_n = safe_normalize(
            (entity_transform.transpose_inverse_model_matrix * vec4<f32>(n_local, 0.0)).xyz
        );
        let t_local = vertex0.tangent.xyz * w_bc +
            vertex1.tangent.xyz * u_bc +
            vertex2.tangent.xyz * v_bc;
        let world_t = safe_normalize(
            (entity_transform.transpose_inverse_model_matrix * vec4<f32>(t_local, 0.0)).xyz
        );
        let b_local = vertex0.bitangent.xyz * w_bc +
            vertex1.bitangent.xyz * u_bc +
            vertex2.bitangent.xyz * v_bc;
        let world_b = safe_normalize(
            (entity_transform.transpose_inverse_model_matrix * vec4<f32>(b_local, 0.0)).xyz
        );
        let p_local = vertex0.position.xyz * w_bc +
            vertex1.position.xyz * u_bc +
            vertex2.position.xyz * v_bc;
        let hit_position = (entity_transform.transform * vec4<f32>(p_local, 1.0)).xyz;

        var n = world_n;
        let normal_flags = u32(material.texture_flags1.y);
        if ((normal_flags & NORMAL_TEXTURE_FLAG_PRESENT) != 0u) {
            let tbn = mat3x3<f32>(world_t, world_b, world_n);
            let nm = decode_normal_texture_sample(
                sample_handle_rgba(
                    u32(material.normal_handle), base_uv,
                    texture_pool_normal, lod
                ).xyz,
                normal_flags
            );
            n = normalize(tbn * nm);
        }

        // -----------------------------------------------------------------------------
        // Direct lighting seed (NEE at the PRIMARY HIT POINT)
        // - Hit pass retains the selected light's radiance only when visible.
        // - This pass applies the shaded material response.
        // -----------------------------------------------------------------------------
        var radiance = safe_clamp_vec3_max(
            ddgi_probe_ray_radiance(hit),
            MAX_RADIANCE_LUMINANCE
        );

        // Reseed multi-bounce using last frame's DDGI SH field.
        // Treat SH as incident diffuse irradiance at the hit point.
        let sh_irradiance = ddgi_sample_sh_irradiance_with_states(
            &ddgi_params,
            &sh_probes,
            &probe_states,
            &probe_depth_moments,
            &probe_depth_slots,
            hit_position,
            n
        );
        radiance += safe_clamp_vec3_max(sh_irradiance, MAX_RADIANCE_LUMINANCE);

        // Apply Lambertian BRDF to reflected light (NEE + multi-bounce SH)
        radiance *= albedo * (1.0 / (2.0 * PI));

        if (emissive > 0.0) {
            radiance += stabilize_emissive_hit_radiance(emissive * albedo);
        }

        ddgi_probe_ray_set_radiance(&probe_ray_data.rays[gid.x], radiance);
    }
}
