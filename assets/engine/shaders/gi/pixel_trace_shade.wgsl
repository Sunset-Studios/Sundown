// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║               PER-PIXEL PATH TRACING - SHADING PASS                       ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Evaluates material properties at ray hit points:                         ║
// ║  • Samples material textures (albedo, normal, roughness, etc.)            ║
// ║  • Handles emissive surfaces                                              ║
// ║  • Queries world cache for multi-bounce irradiance                        ║
// ║  • Evaluates sky/environment for ray misses                               ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "postprocess_common.wgsl"
#include "sky_common.wgsl"
#include "gi/gi_common.wgsl"
#include "gi/world_cache_common.wgsl"
#include "raytracing/restir_common.wgsl"

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<uniform> scene_lighting_data: SceneLightingData;
@group(1) @binding(2) var<storage, read_write> pixel_path_state: array<PixelPathState>;
@group(1) @binding(3) var<storage, read> material_params: array<StandardMaterialParams>;
@group(1) @binding(4) var<storage, read> material_table_offset: array<u32>;
@group(1) @binding(5) var<storage, read> material_palette: array<u32>;
@group(1) @binding(6) var texture_pool_albedo: texture_2d_array<f32>;
@group(1) @binding(7) var texture_pool_normal: texture_2d_array<f32>;
@group(1) @binding(8) var texture_pool_roughness: texture_2d_array<f32>;
@group(1) @binding(9) var texture_pool_metallic: texture_2d_array<f32>;
@group(1) @binding(10) var texture_pool_ao: texture_2d_array<f32>;
@group(1) @binding(11) var texture_pool_height: texture_2d_array<f32>;
@group(1) @binding(12) var texture_pool_specular: texture_2d_array<f32>;
@group(1) @binding(13) var texture_pool_emission: texture_2d_array<f32>;
@group(1) @binding(14) var skybox_texture: texture_cube<f32>;
@group(1) @binding(15) var<storage, read_write> world_cache: array<WorldCacheCell>;

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

// =============================================================================
// MAIN COMPUTE SHADER
// =============================================================================

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Compute total rays from GI internal resolution (trace every GI pixel each frame)
    let rays_per_pixel = u32(gi_params.screen_ray_count);
    let gi_resolution = vec2<u32>(u32(gi_params.gi_resolution_x), u32(gi_params.gi_resolution_y));
    let total_pixels = gi_resolution.x * gi_resolution.y;
    let total_rays = total_pixels * rays_per_pixel;
    
    if (gid.x >= total_rays) {
        return;
    }
    
    let light_view_index = u32(scene_lighting_data.view_index);
    let camera_position = view_buffer[u32(frame_info.view_index)].view_position.xyz;
    let sun_dir = normalize(-view_buffer[light_view_index].view_direction.xyz);

    // We only handle NEE direct lighting when using the radiance cache as deferred lighting
#if USE_RADIANCE_CACHE_AS_DEFERRED_LIGHTING
    // ─────────────────────────────────────────────────────────────────────────
    // Handle Direct Light Visibility (NEE result from hit pass)
    // Clamp NEE contribution to prevent fireflies from bright lights
    // ─────────────────────────────────────────────────────────────────────────
    if (pixel_path_state[gid.x].shadow_origin.w >= 0.0 && pixel_path_state[gid.x].state_u32.z == 1u) {
        let nee_radiance = safe_clamp_vec3_max(pixel_path_state[gid.x].shadow_radiance.rgb, MAX_RADIANCE_LUMINANCE);
        pixel_path_state[gid.x].throughput_direct += vec4<f32>(nee_radiance, 0.0);
        pixel_path_state[gid.x].shadow_origin.w = -1.0;
        pixel_path_state[gid.x].state_u32.z = 0u;
    }
#endif
    

    // ─────────────────────────────────────────────────────────────────────────
    // Handle Ray Miss (Sky/Environment)
    // Clamp sky contribution to prevent sun disc fireflies on specular bounces
    // ─────────────────────────────────────────────────────────────────────────
    if (pixel_path_state[gid.x].state_u32.w == 0xffffffffu && pixel_path_state[gid.x].state_u32.y != 0u) {
        let is_specular_lobe = pixel_path_state[gid.x].state_u32.x == 1u;

        // Evaluate environment radiance
        let sky_radiance = evaluate_environment(
            pixel_path_state[gid.x].direction_tmax.xyz, 
            sun_dir, 
            scene_lighting_data,
            skybox_texture
        );
        
        // Clamp sky radiance. This prevents sun disc from causing fireflies on specular surfaces.
        let sky_contribution = safe_clamp_vec3_max(sky_radiance, MAX_RADIANCE_LUMINANCE);
        let indirect_add = vec4<f32>(sky_contribution, 0.0);
        pixel_path_state[gid.x].throughput_indirect_diffuse += select(indirect_add, vec4<f32>(0.0), is_specular_lobe);
        pixel_path_state[gid.x].throughput_indirect_specular += select(vec4<f32>(0.0), indirect_add, is_specular_lobe);

        pixel_path_state[gid.x].rng_sample_count_frame_stamp.y += 1.0;
        pixel_path_state[gid.x].state_u32.y = 0u; // Mark path as dead
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Handle Ray Hit
    // ─────────────────────────────────────────────────────────────────────────
    if (pixel_path_state[gid.x].state_u32.w != 0xffffffffu && pixel_path_state[gid.x].state_u32.y != 0u) {
        let hit_pos = pixel_path_state[gid.x].origin_tmin.xyz;
        let world_n = pixel_path_state[gid.x].normal_section_index.xyz;
        
        // ─────────────────────────────────────────────────────────────────────
        // Sample Material Properties
        // ─────────────────────────────────────────────────────────────────────
        let prim_store = u32(pixel_path_state[gid.x].direction_tmax.w);
        let entity_palette_base = material_table_offset[prim_store];
        let section_index = u32(pixel_path_state[gid.x].normal_section_index.w);
        let mat_params_index = material_palette[entity_palette_base + section_index];
        let material = material_params[mat_params_index];

        let tiling = material.emission_roughness_metallic_tiling.w;
        let base_uv = vec2<f32>(pixel_path_state[gid.x].hit_attr0.w, pixel_path_state[gid.x].hit_attr1.w) * tiling;
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

        // Normal mapping
        let world_t = pixel_path_state[gid.x].hit_attr0.xyz;
        let world_b = pixel_path_state[gid.x].hit_attr1.xyz;
        var n = world_n;
        if ((u32(material.texture_flags1.y) & 1u) != 0u) {
            let tbn = mat3x3<f32>(world_t, world_b, world_n);
            let nm = sample_handle_rgba(
                u32(material.normal_handle), base_uv,
                texture_pool_normal, lod
            ).xyz * 2.0 - 1.0;
            n = normalize(tbn * nm);
        }

        if (emissive > 0.0) {
            let emissive_radiance = stabilize_emissive_hit_radiance(emissive * albedo);
            let is_specular_lobe = pixel_path_state[gid.x].state_u32.x == 1u;
            let indirect_add = vec4<f32>(emissive_radiance, 0.0) * pixel_path_state[gid.x].path_weight;
            pixel_path_state[gid.x].throughput_indirect_diffuse += select(indirect_add, vec4<f32>(0.0), is_specular_lobe);
            pixel_path_state[gid.x].throughput_indirect_specular += select(vec4<f32>(0.0), indirect_add, is_specular_lobe);
        }

        // ─────────────────────────────────────────────────────────────────────
        // World Cache Query (Multi-Bounce Irradiance)
        // ─────────────────────────────────────────────────────────────────────
        let hit_distance_for_cache = pixel_path_state[gid.x].origin_tmin.w;
        var cached_radiance = vec3<f32>(0.0);
        if (hit_distance_for_cache >= gi_params.world_cache_cell_size * 0.5) {
            cached_radiance = query_world_cache_cell(
                hit_pos,
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
                hit_distance_for_cache,
                0u // Screen space traces rank at 0 (first hit)
            );
        }
        
        // Apply cached radiance if valid, with firefly clamping
        let cached_luminance = luminance(cached_radiance);
        if (cached_luminance > 0.0001) {
            let is_specular_lobe = pixel_path_state[gid.x].state_u32.x == 1u;
            let indirect_add = vec4<f32>(safe_clamp_vec3_max(cached_radiance, MAX_RADIANCE_LUMINANCE), 0.0);
            pixel_path_state[gid.x].throughput_indirect_diffuse += select(indirect_add * pixel_path_state[gid.x].path_weight, vec4<f32>(0.0), is_specular_lobe);
            pixel_path_state[gid.x].throughput_indirect_specular += select(vec4<f32>(0.0), indirect_add * pixel_path_state[gid.x].path_weight, is_specular_lobe);
        }
        
        pixel_path_state[gid.x].rng_sample_count_frame_stamp.y += 1.0;
    }
}

