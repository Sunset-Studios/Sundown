// =============================================================================
// GI-1.0 Screen Probe Ray Tracing - Shade Pass (ReSTIR GI)
// - Implements ReSTIR GI for probe path reuse
// - Combines direct and indirect lighting with reservoir sampling
// - Performs temporal + spatial reuse across probes for noise reduction
// =============================================================================
#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "postprocess_common.wgsl"
#include "sky_common.wgsl"
#include "gi/gi_common.wgsl"
#include "gi/world_cache_common.wgsl"
#include "raytracing/restir_common.wgsl"

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<uniform> scene_lighting_data: SceneLightingData;
@group(1) @binding(2) var<storage, read_write> gi_counters: GICounters;
@group(1) @binding(3) var<storage, read_write> probe_path_state: array<ProbePathState>;
@group(1) @binding(4) var<storage, read_write> world_cache: array<WorldCacheCell>;
@group(1) @binding(5) var<storage, read> material_params: array<StandardMaterialParams>;
@group(1) @binding(6) var<storage, read> material_table_offset: array<u32>;
@group(1) @binding(7) var<storage, read> material_palette: array<u32>;
@group(1) @binding(8) var<storage, read> dense_lights_buffer: array<Light>;
@group(1) @binding(9) var texture_pool_albedo: texture_2d_array<f32>;
@group(1) @binding(10) var texture_pool_normal: texture_2d_array<f32>;
@group(1) @binding(11) var texture_pool_roughness: texture_2d_array<f32>;
@group(1) @binding(12) var texture_pool_metallic: texture_2d_array<f32>;
@group(1) @binding(13) var texture_pool_ao: texture_2d_array<f32>;
@group(1) @binding(14) var texture_pool_height: texture_2d_array<f32>;
@group(1) @binding(15) var texture_pool_specular: texture_2d_array<f32>;
@group(1) @binding(16) var texture_pool_emission: texture_2d_array<f32>;
@group(1) @binding(17) var skybox_texture: texture_cube<f32>;

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let probe_count = u32(gi_params.total_screen_probes);
    let rays_per_probe = u32(gi_params.screen_ray_count);
    let total_rays = probe_count * rays_per_probe;
    
    if (gid.x >= total_rays) {
        return;
    }
    
    var path = probe_path_state[gid.x];
    let tri_id = path.state_u32.w;
    
    let light_view_index = u32(scene_lighting_data.view_index);
    let light_view = view_buffer[light_view_index];
    let camera_position = view_buffer[u32(frame_info.view_index)].view_position.xyz;
    let sun_dir = normalize(-light_view.view_direction.xyz);

    // === Handle primary vertex visibility ray throughput ===
    if (path.shadow_origin.w >= 0.0 && path.state_u32.z == 1u) {
        path.throughput += vec4f(path.shadow_radiance.rgb * path.path_weight.xyz, 0.0);
        path.shadow_origin.w = -1.0;
        path.state_u32.z = 0u;
    }
    
    // === Handle Ray Miss (Sky) ===
    if (tri_id == 0xffffffffu && path.state_u32.y != 0u) {
        let ray_dir = path.direction_tmax.xyz;
        // Evaluate environment radiance
        let sky_radiance = evaluate_environment(
            ray_dir, 
            sun_dir, 
            scene_lighting_data,
            skybox_texture
        );
        // Add sky contribution weighted by path throughput
        path.throughput += vec4<f32>(sky_radiance * gi_params.indirect_boost * path.path_weight.xyz, 0.0);
        // Mark path as dead
        path.state_u32.y = 0u;
    }
    
    // === Handle Ray Hit ===
    if (tri_id != 0xffffffffu && path.state_u32.y != 0u) {
        let hit_pos = path.origin_tmin.xyz;
        let world_n = path.normal_section_index.xyz;
        
        // Sample material properties from textures
        let prim_store = u32(path.direction_tmax.w);
        let entity_palette_base = material_table_offset[prim_store];
        let section_index = u32(path.normal_section_index.w);
        let mat_params_index = material_palette[entity_palette_base + section_index];
        let material = material_params[mat_params_index];

        let tiling = material.emission_roughness_metallic_tiling.w;
        let base_uv = vec2f(path.hit_attr0.w, path.hit_attr1.w) * tiling;
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
        let specular = sample_texture_or_float_param_handle(
            u32(material.specular_handle), base_uv,
            material.ao_height_specular.z,
            u32(material.texture_flags2.z), texture_pool_specular, lod
        );
        let reflectance = specular * 0.0009765625;

        // Normal mapping
        let world_t = path.hit_attr0.xyz;
        let world_b = path.hit_attr1.xyz;
        var n = world_n;
        if ((u32(material.texture_flags1.y) & 1u) != 0u) {
            let tbn = mat3x3<f32>(world_t, world_b, world_n);
            let nm = sample_handle_rgba(
                u32(material.normal_handle), base_uv,
                texture_pool_normal, lod
            ).xyz * 2.0 - 1.0;
            n = normalize(tbn * nm);
        }

        // === EMISSIVE CONTRIBUTION ===
        if (emissive > 0.0) {
            let emissive_radiance = emissive * albedo;
            // The formula is deterministic: same distance + PDF -> same result every frame
            let hit_distance = max(path.origin_tmin.w, 0.01); // Clamp to avoid division by zero
            let ray_source_pdf = path.path_weight.w;
            let raw_contribution = emissive_radiance * path.path_weight.xyz;
            let contribution_luminance = raw_contribution.x * 0.2126 + raw_contribution.y * 0.7152 + raw_contribution.z * 0.0722;
                
            // Distance-based maximum: closer emissives can contribute more
            // This naturally reduces fireflies from distant small emissives
            let distance_factor = 1.0 / hit_distance;
            let max_contribution = emissive * PI * distance_factor;
            
            // Compute stable scale factor (deterministic for same inputs)
            let scale = min(1.0, (max_contribution * ray_source_pdf) / max(contribution_luminance, 0.001));
             
            let emissive_contribution = safe_clamp_vec3(raw_contribution * scale);
            path.throughput += vec4f(emissive_contribution, 0.0);
        }

        // =====================================================================
        // === World Cache Query - Early termination with cached irradiance ===
        // =====================================================================
        // Query world cache at hit point to check if we have cached radiance
        // NOTE: Use geometric normal (world_n) not shading normal (n) for consistent cache lookups
        let cached_radiance = query_world_cache_cell(
            hit_pos,
            world_n,
            albedo,
            roughness,
            metallic,
            reflectance,
            emissive,
            camera_position,
            u32(gi_params.world_cache_size),
            gi_params.world_cache_cell_size,
            u32(gi_params.world_cache_lod_count)
        );
        
        // Check if we got valid cached data (non-zero radiance)
        let cached_luminance = cached_radiance.x * 0.2126 + cached_radiance.y * 0.7152 + cached_radiance.z * 0.0722;
        if (cached_luminance > 0.0001) {
            // Found valid cached radiance! Apply it and terminate path
            let cached_contribution = cached_radiance * gi_params.indirect_boost * path.path_weight.xyz;
            path.throughput += vec4f(cached_contribution, 0.0);
        }
        
        path.rng_sample_count_frame_stamp.y += 1.0;
    }
    
    probe_path_state[gid.x] = path;
}

