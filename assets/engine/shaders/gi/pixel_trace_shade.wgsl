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
// FIREFLY SUPPRESSION CONSTANTS
// =============================================================================

// Maximum luminance for any single radiance contribution
// Tune based on your HDR range - lower = more aggressive firefly removal
const MAX_RADIANCE_LUMINANCE = 10.0;

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<uniform> scene_lighting_data: SceneLightingData;
@group(1) @binding(2) var<storage, read_write> gi_counters: GICounters;
@group(1) @binding(3) var<storage, read_write> pixel_path_state: array<PixelPathState>;
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

// =============================================================================
// MAIN COMPUTE SHADER
// =============================================================================

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Compute total rays from tile grid (not total pixels)
    let rays_per_tile = u32(gi_params.screen_ray_count);
    let resolution = vec2<u32>(u32(gi_params.resolution_x), u32(gi_params.resolution_y));
    let upscale_factor = u32(gi_params.upscale_factor);
    let tile_grid_dims = vec2<u32>(resolution.x / upscale_factor, resolution.y / upscale_factor);
    let total_tiles = tile_grid_dims.x * tile_grid_dims.y;
    let total_rays = total_tiles * rays_per_tile;
    
    if (gid.x >= total_rays) {
        return;
    }
    
    var path = pixel_path_state[gid.x];
    let tri_id = path.state_u32.w;
    
    let light_view_index = u32(scene_lighting_data.view_index);
    let light_view = view_buffer[light_view_index];
    let camera_position = view_buffer[u32(frame_info.view_index)].view_position.xyz;
    let sun_dir = normalize(-light_view.view_direction.xyz);

    // ─────────────────────────────────────────────────────────────────────────
    // Handle Direct Light Visibility (NEE result from hit pass)
    // Clamp NEE contribution to prevent fireflies from bright lights
    // ─────────────────────────────────────────────────────────────────────────
    if (path.shadow_origin.w >= 0.0 && path.state_u32.z == 1u) {
        let nee_radiance = safe_clamp_vec3_max(path.shadow_radiance.rgb, MAX_RADIANCE_LUMINANCE);
        path.throughput += vec4f(nee_radiance, 0.0);
        path.shadow_origin.w = -1.0;
        path.state_u32.z = 0u;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Handle Ray Miss (Sky/Environment)
    // Clamp sky contribution to prevent sun disc fireflies on specular bounces
    // ─────────────────────────────────────────────────────────────────────────
    if (tri_id == 0xffffffffu && path.state_u32.y != 0u) {
        let ray_dir = path.direction_tmax.xyz;
        
        // Evaluate environment radiance
        let sky_radiance = evaluate_environment(
            ray_dir, 
            sun_dir, 
            scene_lighting_data,
            skybox_texture
        );
        
        // Clamp sky radiance before multiplying by path weight
        // This prevents sun disc from causing fireflies on specular surfaces
        let sky_contribution = safe_clamp_vec3_max(sky_radiance, MAX_RADIANCE_LUMINANCE);
        path.throughput += vec4<f32>(sky_contribution, 0.0);
        
        // Mark path as dead
        path.state_u32.y = 0u;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Handle Ray Hit
    // ─────────────────────────────────────────────────────────────────────────
    if (tri_id != 0xffffffffu && path.state_u32.y != 0u) {
        let hit_pos = path.origin_tmin.xyz;
        let world_n = path.normal_section_index.xyz;
        
        // ─────────────────────────────────────────────────────────────────────
        // Sample Material Properties
        // ─────────────────────────────────────────────────────────────────────
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
        let reflectance = specular;

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

        // ─────────────────────────────────────────────────────────────────────
        // Emissive Contribution
        // ─────────────────────────────────────────────────────────────────────
        if (emissive > 0.0) {
            let emissive_radiance = emissive * albedo;
            let hit_distance = max(path.origin_tmin.w, 0.01);
            let ray_source_pdf = path.path_weight.w;
            let raw_contribution = emissive_radiance;
            let contribution_luminance = raw_contribution.x * 0.2126 + raw_contribution.y * 0.7152 + raw_contribution.z * 0.0722;
                
            // Distance-based maximum for firefly reduction
            let distance_factor = 1.0 / hit_distance;
            let max_contribution = emissive * PI * distance_factor;
            
            let scale = min(1.0, (max_contribution * ray_source_pdf) / max(contribution_luminance, 0.001));
             
            let emissive_contribution = safe_clamp_vec3(raw_contribution * scale);
            path.throughput += vec4f(emissive_contribution, 0.0);
        }

        // ─────────────────────────────────────────────────────────────────────
        // World Cache Query (Multi-Bounce Irradiance)
        // ─────────────────────────────────────────────────────────────────────
        let cached_radiance = query_world_cache_cell(
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
            path.origin_tmin.w,
            0u // Screen space traces rank at 0 (first hit)
        );
        
        // Apply cached radiance if valid, with firefly clamping
        let cached_luminance = cached_radiance.x * 0.2126 + cached_radiance.y * 0.7152 + cached_radiance.z * 0.0722;
        if (cached_luminance > 0.0001) {
            let cached_contribution = safe_clamp_vec3_max(cached_radiance, MAX_RADIANCE_LUMINANCE);
            path.throughput += vec4f(cached_contribution, 0.0);
        }
        
        path.rng_sample_count_frame_stamp.y += 1.0;
    }
    
    pixel_path_state[gid.x] = path;
}

