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
#include "gi/world_cache_common.wgsl"

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(2) var<uniform> scene_lighting_data: SceneLightingData;
@group(1) @binding(3) var<storage, read> probe_update_indices: array<u32>;
@group(1) @binding(4) var<storage, read> probe_ray_hits: array<DDGIProbeRayHit>;
@group(1) @binding(5) var<storage, read_write> probe_ray_radiance: array<vec4<f32>>;
@group(1) @binding(6) var<storage, read> material_params: array<StandardMaterialParams>;
@group(1) @binding(7) var<storage, read> material_table_offset: array<u32>;
@group(1) @binding(8) var<storage, read> material_palette: array<u32>;
@group(1) @binding(9) var<storage, read_write> world_cache: array<WorldCacheCell>;
@group(1) @binding(10) var texture_pool_albedo: texture_2d_array<f32>;
@group(1) @binding(11) var texture_pool_normal: texture_2d_array<f32>;
@group(1) @binding(12) var texture_pool_roughness: texture_2d_array<f32>;
@group(1) @binding(13) var texture_pool_metallic: texture_2d_array<f32>;
@group(1) @binding(14) var texture_pool_ao: texture_2d_array<f32>;
@group(1) @binding(15) var texture_pool_height: texture_2d_array<f32>;
@group(1) @binding(16) var texture_pool_specular: texture_2d_array<f32>;
@group(1) @binding(17) var texture_pool_emission: texture_2d_array<f32>;
@group(1) @binding(18) var skybox_texture: texture_cube<f32>;

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rays_per_probe = u32(ddgi_params.probe_counts.y);
    let probes_per_frame = u32(ddgi_params.probe_counts.z);
    let total_rays = probes_per_frame * rays_per_probe;

    if (gid.x >= total_rays) {
        return;
    }

    let hit = probe_ray_hits[gid.x];
    let ray_dir = hit.ray_dir_prim.xyz;

    let light_view_index = u32(scene_lighting_data.view_index);
    let sun_dir = normalize(-view_buffer[light_view_index].view_direction.xyz);

    // Ray miss: evaluate environment radiance.
    if (hit.state_u32.y != 0u && hit.state_u32.w == 0xffffffffu) {
        let env_radiance = evaluate_environment(ray_dir, sun_dir, scene_lighting_data, skybox_texture);
        probe_ray_radiance[gid.x] = vec4f(safe_clamp_vec3_max(env_radiance, MAX_RADIANCE_LUMINANCE), 1.0);
        return;
    }

    // Ray hit: shade the hit.
    if (hit.state_u32.y != 0u && hit.state_u32.w != 0xffffffffu) {
        let view_index = u32(frame_info.view_index);
        let camera_position = view_buffer[view_index].view_position.xyz;

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

        let probe_slot = gid.x / rays_per_probe;
        let probe_index = probe_update_indices[probe_slot];

        var rng = hash(probe_index ^ u32(gi_params.frame_index) ^ gid.x);

        let cached_radiance = query_world_cache_cell_probabilistic(
            hit.hit_pos_t.xyz,
            n,
            albedo,
            roughness,
            metallic,
            reflectance,
            emissive,
            camera_position,
            u32(gi_params.world_cache_size),
            gi_params.world_cache_cell_size,
            u32(gi_params.world_cache_lod_count),
            50.0,
            rand_float(rng),
            hit.hit_pos_t.w,
            0u
        );

        let cached_luminance = luminance(cached_radiance);
        if (cached_luminance > 0.0001) {
            radiance += safe_clamp_vec3_max(cached_radiance, MAX_RADIANCE_LUMINANCE);
        }

        probe_ray_radiance[gid.x] = vec4f(radiance, 1.0);
    }
}


