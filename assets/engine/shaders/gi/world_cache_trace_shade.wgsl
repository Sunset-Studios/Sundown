// =============================================================================
// GI-1.0 World Cache Ray Tracing - Shade Pass
// - Shades hits from world cache rays
// - Accumulates radiance into world cache cells
// - Uses simplified BRDF (Lambertian) for irradiance caching
// - Can query other world cache cells for multi-bounce indirect
// =============================================================================
#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "postprocess_common.wgsl"
#include "sky_common.wgsl"
#include "gi/gi_common.wgsl"
#include "gi/world_cache_common.wgsl"

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<uniform> scene_lighting_data: SceneLightingData;
@group(1) @binding(2) var<storage, read_write> world_cache: array<WorldCacheCell>;
@group(1) @binding(3) var<storage, read> compacted_indices: array<u32>;
@group(1) @binding(4) var<storage, read_write> world_cache_path_state: array<WorldCachePathState>;
@group(1) @binding(5) var<storage, read> material_params: array<StandardMaterialParams>;
@group(1) @binding(6) var<storage, read> material_table_offset: array<u32>;
@group(1) @binding(7) var<storage, read> material_palette: array<u32>;
@group(1) @binding(8) var<storage, read> gi_counters: GICountersReadOnly;
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

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Early exit if beyond active cell count
    if (gid.x >= gi_counters.active_cache_cell_count) {
        return;
    }
    
    // Get actual world cache cell index from compacted array
    let cell_index = compacted_indices[gid.x];
    let path = world_cache_path_state[gid.x];
    
    let light_view_index = u32(scene_lighting_data.view_index);
    let camera_position = view_buffer[u32(frame_info.view_index)].view_position.xyz;
    let sun_dir = normalize(-view_buffer[light_view_index].view_direction.xyz);

    var rng = u32(world_cache_path_state[gid.x].rng_rank_frame_stamp.x);
    if (rng == 0u) { rng = hash(cell_index ^ u32(gi_params.frame_index)); }
    else { rng = random_seed(rng); }

    // Accumulator for this cell's radiance contribution
    var radiance_contribution = vec3<f32>(0.0);
    var sample_count = 0.0;

    // === Handle primary vertex visibility ray throughput (direct lighting) ===
    if (path.shadow_origin.w >= 0.0 && path.state_u32.z == 1u) {
        radiance_contribution += safe_clamp_vec3_max(path.shadow_radiance.rgb, MAX_NEE_LUMINANCE);
        world_cache_path_state[gid.x].state_u32.z = 0u;
        sample_count += 1.0;
    }
    
    // === Handle Ray Miss (Sky contribution) ===
    // Always count sky contributions, even for "dead" rays from failed ReSTIR
    // This prevents cold-start where cells never update
    if (path.state_u32.w == 0xffffffffu && path.state_u32.y != 0u) {
        let ray_dir = path.direction_tmax.xyz;
        // Evaluate environment radiance
        let sky_radiance = evaluate_environment(
            ray_dir, 
            sun_dir, 
            scene_lighting_data,
            skybox_texture
        );
        // Add sky contribution weighted by path throughput
        radiance_contribution += safe_clamp_vec3_max(sky_radiance, MAX_RADIANCE_LUMINANCE);
        sample_count += 1.0;
    }
    
    // === Handle Ray Hit ===
    if (path.state_u32.w != 0xffffffffu && path.state_u32.y != 0u) {
        let hit_pos = path.origin_tmin.xyz;
        let world_n = path.normal_section_index.xyz;
        
        var albedo: vec3<f32>;
        var roughness: f32;
        var metallic: f32;
        var emissive: f32;
        var reflectance: f32;
        var n: vec3<f32>;
        
        // Sample material properties from textures
        let prim_store = u32(path.direction_tmax.w);
        let entity_palette_base = material_table_offset[prim_store];
        let section_index = u32(path.normal_section_index.w);
        let mat_params_index = material_palette[entity_palette_base + section_index];
        let material = material_params[mat_params_index];

        let tiling = material.emission_roughness_metallic_tiling.w;
        let base_uv = vec2<f32>(path.hit_attr0.w, path.hit_attr1.w) * tiling;
        let lod = 0.0;

        albedo = sample_texture_or_vec4_param_handle(
            u32(material.albedo_handle), base_uv, material.albedo,
            u32(material.texture_flags1.x), texture_pool_albedo, lod
        ).xyz;
        roughness = sample_texture_or_float_param_handle(
            u32(material.roughness_handle), base_uv,
            material.emission_roughness_metallic_tiling.y,
            u32(material.texture_flags1.z), texture_pool_roughness, lod
        );
        metallic = sample_texture_or_float_param_handle(
            u32(material.metallic_handle), base_uv,
            material.emission_roughness_metallic_tiling.z,
            u32(material.texture_flags1.w), texture_pool_metallic, lod
        );
        emissive = sample_texture_or_float_param_handle(
            u32(material.emission_handle), base_uv,
            material.emission_roughness_metallic_tiling.x,
            u32(material.texture_flags2.w), texture_pool_emission, lod
        );
        reflectance = sample_texture_or_float_param_handle(
            u32(material.specular_handle), base_uv,
            material.ao_height_specular.z,
            u32(material.texture_flags2.z), texture_pool_specular, lod
        );

        // Normal mapping
        let world_t = path.hit_attr0.xyz;
        let world_b = path.hit_attr1.xyz;
        n = world_n;
        if ((u32(material.texture_flags1.y) & 1u) != 0u) {
            let tbn = mat3x3<f32>(world_t, world_b, world_n);
            let nm = sample_handle_rgba(
                u32(material.normal_handle), base_uv,
                texture_pool_normal, lod
            ).xyz * 2.0 - 1.0;
            n = normalize(tbn * nm);
        }

        if (emissive > 0.0) {
            radiance_contribution += stabilize_emissive_hit_radiance(emissive * albedo);
        }

        // === INDIRECT LIGHTING - Query world cache for multi-bounce ===
        // Query world cache at hit point to get cached irradiance from previous frames
        // This provides multi-bounce indirect illumination without tracing further
        // Pass ray hit distance for light-leak prevention (short ray separation)
        let cached_radiance = query_world_cache_cell_probabilistic(
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
            50.0, // allocation radius
            rand_float(rng),
            path.origin_tmin.w,
            u32(path.rng_rank_frame_stamp.y)
        );
        
        // Check if we got valid cached data
        let cached_luminance = luminance(cached_radiance);
        if (cached_luminance > 0.0001) {
            radiance_contribution += safe_clamp_vec3_max(cached_radiance * path.path_weight.xyz, MAX_RADIANCE_LUMINANCE);
        }
        
        sample_count += 1.0;
    }
    
    // =============================================================================
    // UPDATE WORLD CACHE CELL with accumulated radiance
    // =============================================================================
    if (sample_count > 0.0) {
        let current_radiance = world_cache[cell_index].radiance_m;
        
        let alpha = 1.0 / min((current_radiance.w + sample_count), WORLD_CACHE_RADIANCE_UPDATE_SAMPLE_CAP);
        let new_radiance = mix(current_radiance.rgb, radiance_contribution, alpha);
        
        world_cache[cell_index].radiance_m = vec4<f32>(new_radiance, current_radiance.w + sample_count);
    }
}

