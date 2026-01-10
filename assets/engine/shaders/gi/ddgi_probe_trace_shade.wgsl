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
@group(1) @binding(2) var<storage, read> probe_update_indices: array<u32>;
@group(1) @binding(3) var<storage, read> probe_ray_hits: array<DDGIProbeRayHit>;
@group(1) @binding(4) var<storage, read_write> probe_ray_radiance: array<vec4<f32>>;
@group(1) @binding(5) var<storage, read> material_params: array<StandardMaterialParams>;
@group(1) @binding(6) var<storage, read> material_table_offset: array<u32>;
@group(1) @binding(7) var<storage, read> material_palette: array<u32>;
@group(1) @binding(8) var<storage, read> dense_lights_buffer: DenseLightsBuffer;
@group(1) @binding(9) var<storage, read> sh_probes_prev: array<u32>;
@group(1) @binding(10) var probe_depth_atlas: texture_2d_array<f32>;
@group(1) @binding(11) var texture_pool_albedo: texture_2d_array<f32>;
@group(1) @binding(12) var texture_pool_normal: texture_2d_array<f32>;
@group(1) @binding(13) var texture_pool_roughness: texture_2d_array<f32>;
@group(1) @binding(14) var texture_pool_metallic: texture_2d_array<f32>;
@group(1) @binding(15) var texture_pool_ao: texture_2d_array<f32>;
@group(1) @binding(16) var texture_pool_height: texture_2d_array<f32>;
@group(1) @binding(17) var texture_pool_specular: texture_2d_array<f32>;
@group(1) @binding(18) var texture_pool_emission: texture_2d_array<f32>;
@group(1) @binding(19) var skybox_texture: texture_cube<f32>;

fn ddgi_sample_prev_sh_irradiance(position: vec3<f32>, normal_ws: vec3<f32>) -> vec3<f32> {
    let spacing = ddgi_params.probe_counts.w;
    let dims = vec3<u32>(
        u32(ddgi_params.probe_grid_dims.x),
        u32(ddgi_params.probe_grid_dims.y),
        u32(ddgi_params.probe_grid_dims.z)
    );
    let origin = ddgi_params.probe_grid_origin.xyz;
    let probe_radius = ddgi_params.probe_grid_dims.w;

    let rel = (position - origin) / spacing;
    let base_f = floor(rel);
    let frac = rel - base_f;

    let max_base = vec3<f32>(
        f32(select(0u, dims.x - 2u, dims.x > 1u)),
        f32(select(0u, dims.y - 2u, dims.y > 1u)),
        f32(select(0u, dims.z - 2u, dims.z > 1u))
    );
    let base_clamped = clamp(base_f, vec3<f32>(0.0), max_base);
    let base = vec3<u32>(base_clamped);
    let frac_clamped = clamp(frac, vec3<f32>(0.0), vec3<f32>(1.0));

    // Visibility/occlusion bias parameters (DDGI paper-style).
    let normal_bias = probe_radius * 0.20;
    let view_bias = probe_radius * 0.10;
    let mean_bias = probe_radius * 0.20;
    let variance_bias_sq = (probe_radius * 0.20) * (probe_radius * 0.20);

    let perceptual_threshold = max(0.05 * MAX_RADIANCE_LUMINANCE, 1e-4);

    var sh_sum = sh_l1_rgb_zero();
    var weight_sum = 0.0;

    for (var z = 0u; z < 2u; z = z + 1u) {
        for (var y = 0u; y < 2u; y = y + 1u) {
            for (var x = 0u; x < 2u; x = x + 1u) {
                let coord = base + vec3<u32>(x, y, z);
                let clamped_coord = clamp(coord, vec3<u32>(0u), dims - vec3<u32>(1u));
                let probe_index = ddgi_probe_index_from_coord(&ddgi_params, clamped_coord);

                let tri_weight =
                    select(1.0 - frac_clamped.x, frac_clamped.x, x == 1u) *
                    select(1.0 - frac_clamped.y, frac_clamped.y, y == 1u) *
                    select(1.0 - frac_clamped.z, frac_clamped.z, z == 1u);

                let probe_pos = ddgi_probe_world_position_from_coord(&ddgi_params, clamped_coord);
                let dir_to_probe = safe_normalize(probe_pos - position);

                let backface = clamp(dot(normal_ws, dir_to_probe), 0.0, 1.0);
                let backface_weight = backface * backface;

                let biased_pos = position + normal_ws * normal_bias + dir_to_probe * view_bias;
                let dir_from_probe = safe_normalize(biased_pos - probe_pos);
                let dist = length(biased_pos - probe_pos);

                let moments = ddgi_sample_probe_depth_moments(&ddgi_params, probe_depth_atlas, probe_index, dir_from_probe);
                // If the depth atlas is uninitialized / missing data, moments will be ~0.
                // In that case treat the probe as visible rather than collapsing weights to 0.
                let moments_are_valid = (moments.x > 1e-5) || (moments.y > 1e-5);
                let visibility_weight = select(
                    1.0,
                    ddgi_visibility_chebyshev(moments, dist, mean_bias, variance_bias_sq),
                    moments_are_valid
                );

                let probe_sh = ddgi_sh_probe_read(&sh_probes_prev, probe_index);
                let preview_irradiance = max(ddgi_sh_evaluate_irradiance(probe_sh, normal_ws), vec3<f32>(0.0));
                let probe_luma = dot(preview_irradiance, vec3<f32>(0.2126, 0.7152, 0.0722));
                let perceptual_linear = clamp(probe_luma / perceptual_threshold, 0.0, 1.0);
                // Avoid "black holes" from perceptual weight reaching 0.0 everywhere.
                let perceptual_weight = max(perceptual_linear * perceptual_linear, 0.05);

                let weight = tri_weight * backface_weight * visibility_weight * perceptual_weight;

                sh_sum = sh_l1_rgb_add(sh_sum, sh_l1_rgb_multiply_scalar(probe_sh, weight));
                weight_sum = weight_sum + weight;
            }
        }
    }

    // Fallback: if all weights collapse (e.g. outside grid, empty depth atlas, etc.),
    // sample the nearest probe with no visibility/perceptual weighting.
    if (weight_sum <= 1e-6) {
        let nearest_offset = vec3<u32>(
            select(0u, 1u, frac_clamped.x > 0.5),
            select(0u, 1u, frac_clamped.y > 0.5),
            select(0u, 1u, frac_clamped.z > 0.5)
        );
        let nearest_coord = clamp(base + nearest_offset, vec3<u32>(0u), dims - vec3<u32>(1u));
        let nearest_index = ddgi_probe_index_from_coord(&ddgi_params, nearest_coord);
        let nearest_sh = ddgi_sh_probe_read(&sh_probes_prev, nearest_index);
        var irradiance = ddgi_sh_evaluate_irradiance(nearest_sh, normal_ws) * ddgi_params.indirect_boost;
        return max(irradiance, vec3<f32>(0.0));
    }

    let inv_weight_sum = 1.0 / weight_sum;
    let sh_interpolated = sh_l1_rgb_multiply_scalar(sh_sum, inv_weight_sum);
    var irradiance = ddgi_sh_evaluate_irradiance(sh_interpolated, normal_ws) * ddgi_params.indirect_boost;
    return max(irradiance, vec3<f32>(0.0));
}

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rays_per_probe = u32(ddgi_params.probe_counts.y);
    let probes_per_frame = u32(ddgi_params.probe_counts.z);
    let total_primary_rays = probes_per_frame * rays_per_probe;
    let total_rays = probes_per_frame + total_primary_rays;

    if (gid.x >= total_rays) {
        return;
    }

    let hit = probe_ray_hits[gid.x];
    let ray_dir = hit.ray_dir_prim.xyz;
    let probe_index = hit.state_u32.x;

    let light_view_index = u32(scene_lighting_data.view_index);
    let sun_dir = normalize(-view_buffer[light_view_index].view_direction.xyz);

    // -----------------------------------------------------------------------------
    // Shadow ray (1 per probe) for direct lighting via NEE.
    // Layout: [0..probes_per_frame) are shadow rays, [probes_per_frame..) are primary rays.
    // -----------------------------------------------------------------------------
    if (gid.x < probes_per_frame) {
        // Invalid or disabled shadow sample.
        if (hit.state_u32.y == 0u) {
            probe_ray_radiance[gid.x] = vec4f(0.0, 0.0, 0.0, 0.0);
            return;
        }

        let num_lights = dense_lights_buffer.header.light_count;
        if (num_lights == 0u) {
            probe_ray_radiance[gid.x] = vec4f(0.0, 0.0, 0.0, 0.0);
            return;
        }

        let probe_position = ddgi_probe_world_position_from_index(&ddgi_params, probe_index);
        let light_idx = min(u32(hit.ray_dir_prim.w), num_lights - 1u);
        let light = dense_lights_buffer.lights[light_idx];

        // Unbiased w.r.t. uniform light selection: multiply by num_lights (1 / p(light)).
        let attenuation = get_light_attenuation(light, probe_position);
        let raw_light_contrib = light.color.rgb * light.intensity * attenuation * f32(num_lights);
        let light_contrib = safe_clamp_vec3_max(raw_light_contrib, MAX_NEE_LUMINANCE);

        let is_visible = hit.state_u32.z == 1u;
        probe_ray_radiance[gid.x] = vec4f(select(vec3<f32>(0.0), light_contrib, is_visible), 1.0);
    } else {
        // Ray miss: evaluate environment radiance.
        if (hit.state_u32.y != 0u && hit.state_u32.w == 0xffffffffu) {
            let env_radiance = evaluate_environment(ray_dir, sun_dir, scene_lighting_data, skybox_texture);
            probe_ray_radiance[gid.x] = vec4f(safe_clamp_vec3_max(env_radiance, MAX_RADIANCE_LUMINANCE), 1.0);
        }

        // Ray hit: shade the hit.
        if (hit.state_u32.y != 0u && hit.state_u32.w != 0xffffffffu) {
            let prim_store = u32(hit.ray_dir_prim.w);
            let section_index = u32(hit.world_n_section.w);

            let entity_palette_base = material_table_offset[prim_store];
            let mat_params_index = material_palette[entity_palette_base + section_index];
            let material = material_params[mat_params_index];

            let tiling = material.emission_roughness_metallic_tiling.w;
            let base_uv = vec2f(hit.world_t_uvx.w, hit.world_b_uvy.w) * tiling;
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

            var radiance = vec3<f32>(0.0);

            if (emissive > 0.0) {
                radiance += emissive * albedo;
            }

            // Reseed multi-bounce using last frame's DDGI SH field.
            // Treat SH as incident diffuse irradiance at the hit point.
            let sh_irradiance = ddgi_sample_prev_sh_irradiance(hit.hit_pos_t.xyz, n);
            let bounce_radiance = sh_irradiance * albedo * (1.0 / PI);
            radiance += safe_clamp_vec3_max(bounce_radiance, MAX_RADIANCE_LUMINANCE);

            probe_ray_radiance[gid.x] = vec4f(radiance, 1.0);
        }
    }
}
