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
    
    let ray_id = gid.x;
    let probe_index = ray_id / rays_per_probe;
    
    var path = probe_path_state[ray_id];
    
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
        path.throughput += vec4<f32>(sky_radiance * path.path_weight.xyz, 0.0);
        // Mark path as dead
        path.state_u32.y = 0u;
    }
    
    // === Handle Ray Hit ===
    if (tri_id != 0xffffffffu && path.state_u32.y != 0u) {
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
        let base_uv = vec2f(path.hit_attr0.w, path.hit_attr1.w) * tiling;
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
        let specular = sample_texture_or_float_param_handle(
            u32(material.specular_handle), base_uv,
            material.ao_height_specular.z,
            u32(material.texture_flags2.z), texture_pool_specular, lod
        );
        reflectance = specular * 0.0009765625;

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

        let clear_coat = 0.0;
        let clear_coat_roughness = 0.0;
        let v_dir = -normalize(path.direction_tmax.xyz);
        let n_dot_v = max(dot(v_dir, n), 0.0001);

        // =============================================================================
        // === ReSTIR GI: Generate candidates + reservoir sampling ===
        // =============================================================================
        
        // BRDF precomputation
        let clamped_roughness = clamp(roughness, 0.001, 1.0);
        let dielectric_f0 = 0.16 * reflectance * reflectance;
        let f0 = mix(vec3<f32>(dielectric_f0), albedo, metallic);

        let f = f_schlick_vec3(f0, 1.0, n_dot_v);
        let fresnel_luminance = (f.x + f.y + f.z) / 3.0;
        let use_ggx = (clamped_roughness < 0.3) || (metallic > 0.5);
        let specular_prob_if_ggx = clamp(fresnel_luminance, 0.001, 0.99);
        let mis_specular_prob = select(0.0, specular_prob_if_ggx, use_ggx);
        
        // Storage for candidate GI samples
        var candidate_samples: array<GISample, 8>;
        var num_candidates = 0u;
        let num_lights = gi_counters.light_count;
        
        // === EMISSIVE CONTRIBUTION ===
        if (emissive > 0.0) {
            let emissive_contribution = emissive * albedo * path.path_weight.xyz;
            path.throughput += vec4f(emissive_contribution, 0.0);
        }

        // Get RNG state
        var rng_state = u32(path.rng_sample_count_frame_stamp.x);
        if (rng_state == 0u) {
            rng_state = hash(ray_id ^ u32(gi_params.frame_index));
        } else {
            rng_state = random_seed(rng_state);
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
            let cached_contribution = cached_radiance * path.path_weight.xyz;
            path.throughput += vec4f(cached_contribution, 0.0);
        }
        
        // Generate BRDF sampling candidates (indirect lighting)
        for (var i = 0u; i < num_ris_samples; i = i + 1u) {
            rng_state = random_seed(rng_state);
            let r1 = rand_float(rng_state);
            rng_state = random_seed(rng_state);
            let r2 = rand_float(rng_state);
            rng_state = random_seed(rng_state);
            let r3 = rand_float(rng_state);
            
            var dir: vec3<f32>;
            if (use_ggx && r3 < specular_prob_if_ggx) {
                let h = importance_sample_ggx(vec2<f32>(r1, r2), n, clamped_roughness);
                dir = normalize(reflect(-v_dir, h));
            } else {
                let phi = 2.0 * PI * r1;
                let cos_theta = sqrt(1.0 - r2);
                let sin_theta = sqrt(r2);
                let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.999);
                let tangent = normalize(cross(up, n));
                let bitangent = normalize(cross(n, tangent));
                let dir_local = vec3f(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
                dir = normalize(tangent * dir_local.x + bitangent * dir_local.y + n * dir_local.z);
            }
            
            let brdf = calculate_brdf_rt(
                n, v_dir, dir, albedo, roughness, metallic,
                reflectance, clear_coat, clear_coat_roughness
            );
            
            let brdf_sample_pdf = brdf_pdf(n, v_dir, dir, clamped_roughness, mis_specular_prob);
            
            let brdf_lum = max(0.0, brdf.x * 0.2126 + brdf.y * 0.7152 + brdf.z * 0.0722);
            
            candidate_samples[num_candidates].radiance_and_target_pdf = vec4f(brdf, brdf_lum);
            candidate_samples[num_candidates].direction_and_source_pdf = vec4f(dir, brdf_sample_pdf);
            num_candidates += 1u;
        }
        
        // === STEP 2: Spatial Reuse - Sample neighboring probes' reservoirs ===
        let same_ray_offset = ray_id % rays_per_probe;
        // Sample a few neighboring probes (within same ray index for consistency)
        for (var s = 0u; s < 3u; s = s + 1u) {
            rng_state = random_seed(rng_state);
            let neighbor_probe_offset = i32(rand_float(rng_state) * 8.0) - 4; // -4 to +3 range
            let neighbor_probe_idx = i32(probe_index) + neighbor_probe_offset;
            
            if (neighbor_probe_idx >= 0 && neighbor_probe_idx < i32(probe_count)) {
                let neighbor_ray_id = u32(neighbor_probe_idx) * rays_per_probe + same_ray_offset;
                let neighbor_path = probe_path_state[neighbor_ray_id];
                
                // Geometric similarity test
                if (neighbor_path.state_u32.w != 0xffffffffu) {
                    let neighbor_hit_pos = neighbor_path.origin_tmin.xyz;
                    let neighbor_normal = neighbor_path.normal_section_index.xyz;
                    let position_distance = length(hit_pos - neighbor_hit_pos);
                    let normal_similarity = dot(n, neighbor_normal);
                    
                    // Only reuse from geometrically similar surfaces
                    if (position_distance < 0.5 && normal_similarity > 0.8) {
                        let neighbor_direction = neighbor_path.reservoir_direction_w.xyz;
                        let neighbor_m = u32(neighbor_path.reservoir_radiance_m.w);
                        
                        if (neighbor_m > 0u && length(neighbor_direction) > 0.01) {
                            // UNBIASED: Evaluate neighbor's direction at CURRENT surface
                            let neighbor_brdf = calculate_brdf_rt(
                                n, v_dir, neighbor_direction, albedo, roughness, metallic,
                                reflectance, clear_coat, clear_coat_roughness
                            );
                            
                            // Compute PDF for this direction
                            let neighbor_pdf = brdf_pdf(n, v_dir, neighbor_direction, clamped_roughness, mis_specular_prob);
                            
                            let brdf_lum = max(0.0, neighbor_brdf.x * 0.2126 + neighbor_brdf.y * 0.7152 + neighbor_brdf.z * 0.0722);
                            
                            if (num_candidates < 8u && brdf_lum > 0.0) {
                                candidate_samples[num_candidates].radiance_and_target_pdf = vec4f(neighbor_brdf, brdf_lum);
                                candidate_samples[num_candidates].direction_and_source_pdf = vec4f(neighbor_direction, neighbor_pdf);
                                num_candidates += 1u;
                            }
                        }
                    }
                }
            }
        }
        
        // === STEP 3: Temporal Reuse - UNBIASED approach ===
        let prev_direction = path.reservoir_direction_w.xyz;
        let prev_m = u32(path.reservoir_radiance_m.w);
        let prev_importance_hint = path.reservoir_radiance_m.xyz;
        
        if (prev_m > 0u && length(prev_direction) > 0.01) {
            let prev_brdf = calculate_brdf_rt(
                n, v_dir, prev_direction, albedo, roughness, metallic,
                reflectance, clear_coat, clear_coat_roughness
            );
            
            let prev_pdf = brdf_pdf(n, v_dir, prev_direction, clamped_roughness, mis_specular_prob);
            let prev_hint_lum = max(0.0, prev_importance_hint.x * 0.2126 + prev_importance_hint.y * 0.7152 + prev_importance_hint.z * 0.0722);
            let temporal_boost = min(2.0, 1.0 + prev_hint_lum * 0.3);
            let brdf_lum = max(0.0, prev_brdf.x * 0.2126 + prev_brdf.y * 0.7152 + prev_brdf.z * 0.0722) * temporal_boost;
            
            if (num_candidates < 8u && brdf_lum > 0.0) {
                candidate_samples[num_candidates].radiance_and_target_pdf = vec4f(prev_brdf * temporal_boost, brdf_lum);
                candidate_samples[num_candidates].direction_and_source_pdf = vec4f(prev_direction, prev_pdf);
                num_candidates += 1u;
            }
        }
        
        // === STEP 4: Perform RIS on all candidates (new + temporal + spatial) ===
        var gi_reservoir = gi_reservoir_init();
        for (var i = 0u; i < num_candidates; i = i + 1u) {
            let sample = candidate_samples[i];
            let brdf_for_target = calculate_brdf_rt(
                n, v_dir, sample.direction_and_source_pdf.xyz, albedo, roughness, metallic,
                reflectance, clear_coat, clear_coat_roughness
            );
            let target_pdf = compute_gi_target_pdf(sample.radiance_and_target_pdf.xyz, brdf_for_target);
            let ris_weight = target_pdf / max(sample.direction_and_source_pdf.w, 0.0001);
            
            if (ris_weight > 0.0 && !isinf(ris_weight)) {
                gi_reservoir_update(&gi_reservoir, i, ris_weight, &rng_state);
            }
        }
        
        // Finalize reservoir
        if (gi_reservoir.m > 0u) {
            let selected_sample = candidate_samples[gi_reservoir.selected_index];
            let selected_dir = selected_sample.direction_and_source_pdf.xyz;
            let selected_brdf = calculate_brdf_rt(
                n, v_dir, selected_dir, albedo, roughness, metallic,
                reflectance, clear_coat, clear_coat_roughness
            );
            let selected_target = compute_gi_target_pdf(selected_sample.radiance_and_target_pdf.xyz, selected_brdf);
            gi_reservoir_finalize(&gi_reservoir, selected_target);
            
            // Update path weight
            let brdf_weight = selected_brdf * gi_reservoir.w;
            let new_path_weight = path.path_weight.xyz * brdf_weight;
            
            // Russian Roulette: Kill paths with very low throughput to prevent underflow
            // This prevents "zombie" paths that contribute nothing but still propagate bad state
            let weight_luminance = new_path_weight.x * 0.2126 + new_path_weight.y * 0.7152 + new_path_weight.z * 0.0722;
            let min_weight_threshold = 0.00001;
            
            if (weight_luminance < min_weight_threshold) {
                // Path weight too low - kill path and clear reservoir
                path.state_u32.y = 0u;
                path.reservoir_radiance_m = vec4f(0.0);
                path.reservoir_direction_w = vec4f(0.0);
            } else {
                // Store only the DIRECTION for temporal reuse - UNBIASED!
                // We don't store radiance, just the direction and sample count
                // The BRDF estimate is stored as a hint for importance sampling
                var selected_brdf_estimate = selected_sample.radiance_and_target_pdf.xyz;
                
                // Boost importance hint if we hit an emissive with this direction
                // Closer emissives get stronger boost (more relevant for local lighting)
                if (emissive > 0.0) {
                    let hit_distance = max(path.origin_tmin.w, 0.1);
                    // Proximity boost: nearby emissives are more important to cache
                    // 1.0 at distance=0, 0.5 at distance=1, 0.33 at distance=2, etc.
                    let proximity_boost = 1.0 / (1.0 + hit_distance);
                    let emissive_importance = emissive * albedo.x * 0.2126 + emissive * albedo.y * 0.7152 + emissive * albedo.z * 0.0722;
                    let boost_factor = clamp(emissive_importance * 10.0 * proximity_boost, 0.0, 50.0);
                    selected_brdf_estimate = selected_brdf_estimate * (1.0 + boost_factor);
                }
                
                path.reservoir_radiance_m = vec4f(selected_brdf_estimate, f32(gi_reservoir.m));
                path.reservoir_direction_w = vec4f(selected_dir, gi_reservoir.w);
                
                // Store the source PDF of the spawned ray in path_weight.w for future MIS
                let selected_source_pdf = selected_sample.direction_and_source_pdf.w;
                path.path_weight = vec4f(new_path_weight, selected_source_pdf);
                
                // Continue path
                path.origin_tmin = vec4f(hit_pos + n * 0.001, 0.0001);
                path.direction_tmax = vec4f(selected_dir, 1e30);
                path.state_u32.y = 1u;
                path.state_u32.w = 0xffffffffu;
            }
        } else {
            // Reservoir selection failed - kill path and clear reservoir
            path.state_u32.y = 0u;
            path.reservoir_radiance_m = vec4f(0.0);
            path.reservoir_direction_w = vec4f(0.0);
        }

        // Save updated RNG state
        path.rng_sample_count_frame_stamp.x = f32(rng_state);
        path.rng_sample_count_frame_stamp.y += 1.0;
    }
    
    probe_path_state[ray_id] = path;
}

