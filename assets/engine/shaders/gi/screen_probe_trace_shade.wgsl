// =============================================================================
// GI-1.0 Screen Probe Ray Tracing - Shade Pass (ReSTIR GI)
// - Implements ReSTIR GI for probe path reuse
// - Combines direct and indirect lighting with reservoir sampling
// - Updates world cache with secondary bounce radiance
// - Based on path_trace_shade.wgsl for consistency
// =============================================================================
#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "lighting_common.wgsl"
#include "sky_common.wgsl"
#include "gi/world_cache_common.wgsl"

const num_ris_samples = 2u;
const num_env_samples = 2u;
const max_bounces = 2u;

struct GIParams {
    screen_probe_spawn_rate: u32,
    screen_probe_size: u32,
    screen_ray_count: u32,
    world_cache_size: u32,
    max_screen_probes: u32,
    frame_index: u32,
    reset_caches: u32,
    indirect_boost: u32,
    upscale_x: u32,
    upscale_y: u32,
    cell_size_heuristic: u32,
    padding: u32,
};

struct ScreenProbe {
    position_radius: vec4<f32>,
    normal_frame: vec4<f32>,
    radiance_m: vec4<f32>,
    albedo_roughness: vec4<f32>,
    state: vec4<u32>,
};

struct ProbePathState {
    origin_tmin: vec4<f32>,
    direction_tmax: vec4<f32>,
    normal_section_index: vec4<f32>,
    state_u32: vec4<u32>,      // x=bounce, y=alive, z=unused, w=tri_id
    hit_attr0: vec4<f32>,       // xyz = world_tangent, w = uv.x
    hit_attr1: vec4<f32>,       // xyz = world_bitangent, w = uv.y
    shadow_origin: vec4<f32>,
    shadow_direction: vec4<f32>,
    shadow_radiance: vec4<f32>,
};

// ReSTIR GI Reservoir - stores DIRECTION for temporal/spatial reuse
struct ProbePathShade {
    path_weight: vec4<f32>,                // xyz=throughput weight, w=source_pdf
    rng_sample_count_frame_stamp: vec4<f32>,
    throughput: vec4<f32>,
    reservoir_radiance_m: vec4<f32>,       // xyz=BRDF estimate (importance hint), w=m
    reservoir_direction_w: vec4<f32>,      // xyz=next bounce direction, w=final weight
};

// ReSTIR GI Reservoir for path resampling
struct GIReservoir {
    selected_index: u32,
    weight_sum: f32,
    m: u32,
    w: f32,
};

// GI Sample - represents a complete path contribution
struct GISample {
    radiance_and_target_pdf: vec4<f32>,
    direction_and_source_pdf: vec4<f32>,
};

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<uniform> scene_lighting_data: SceneLightingData;
@group(1) @binding(2) var<storage, read_write> screen_probes: array<ScreenProbe>;
@group(1) @binding(3) var<storage, read> screen_probe_counter: array<u32>;
@group(1) @binding(4) var<storage, read_write> probe_path_state: array<ProbePathState>;
@group(1) @binding(5) var<storage, read_write> probe_path_shade: array<ProbePathShade>;
@group(1) @binding(6) var<storage, read_write> world_cache: array<WorldCacheCell>;
@group(1) @binding(7) var<storage, read> material_params: array<StandardMaterialParams>;
@group(1) @binding(8) var<storage, read> material_table_offset: array<u32>;
@group(1) @binding(9) var<storage, read> material_palette: array<u32>;
@group(1) @binding(10) var<storage, read> dense_lights_buffer: array<Light>;
@group(1) @binding(11) var<storage, read> light_count_buffer: array<u32>;
@group(1) @binding(12) var texture_pool_albedo: texture_2d_array<f32>;
@group(1) @binding(13) var texture_pool_normal: texture_2d_array<f32>;
@group(1) @binding(14) var texture_pool_roughness: texture_2d_array<f32>;
@group(1) @binding(15) var texture_pool_metallic: texture_2d_array<f32>;
@group(1) @binding(16) var texture_pool_ao: texture_2d_array<f32>;
@group(1) @binding(17) var texture_pool_height: texture_2d_array<f32>;
@group(1) @binding(18) var texture_pool_specular: texture_2d_array<f32>;
@group(1) @binding(19) var texture_pool_emission: texture_2d_array<f32>;
@group(1) @binding(20) var skybox_texture: texture_cube<f32>;

fn sample_texture_or_vec4_param_handle(
    tex_handle: u32,
    uv_coords: vec2<f32>,
    param_val: vec4<f32>,
    flag: u32,
    pool: texture_2d_array<f32>,
    lod: f32
) -> vec4<f32> {
    if ((flag & 1u) != 0u) {
        return sample_handle_rgba(tex_handle, uv_coords, pool, lod);
    }
    return param_val;
}

fn sample_texture_or_float_param_handle(
    tex_handle: u32,
    uv_coords: vec2<f32>,
    param_val: f32,
    flag: u32,
    pool: texture_2d_array<f32>,
    lod: f32
) -> f32 {
    if ((flag & 1u) != 0u) {
        let sampled_val = sample_handle_rgba(tex_handle, uv_coords, pool, lod);
        let channel_index = (flag >> 1u) & 3u;
        return select(select(select(sampled_val.r, sampled_val.g, channel_index == 1u), sampled_val.b, channel_index == 2u), sampled_val.a, channel_index == 3u);
    }
    return param_val;
}

// =============================================================================
// ReSTIR GI Helper Functions (from path_trace_shade.wgsl)
// =============================================================================

fn gi_reservoir_init() -> GIReservoir {
    var reservoir: GIReservoir;
    reservoir.selected_index = 0u;
    reservoir.weight_sum = 0.0;
    reservoir.m = 0u;
    reservoir.w = 0.0;
    return reservoir;
}

fn gi_reservoir_update(
    reservoir: ptr<function, GIReservoir>,
    candidate_index: u32,
    weight: f32,
    rng_state: ptr<function, u32>
) {
    (*reservoir).weight_sum += weight;
    (*reservoir).m += 1u;
    
    *rng_state = random_seed(*rng_state);
    let xi = rand_float(*rng_state);
    if (xi * (*reservoir).weight_sum < weight) {
        (*reservoir).selected_index = candidate_index;
    }
}

fn gi_reservoir_finalize(
    reservoir: ptr<function, GIReservoir>,
    selected_target_pdf: f32
) {
    let contributes = (*reservoir).m > 0u && selected_target_pdf > 0.0;
    let unclamped_weight = (*reservoir).weight_sum / (f32((*reservoir).m) * max(selected_target_pdf, 0.0001));
    
    let max_weight = 200.0;
    (*reservoir).w = select(
        0.0,
        min(max_weight, unclamped_weight),
        contributes
    );
}

fn compute_gi_target_pdf(
    sample_radiance: vec3<f32>,
    brdf_value: vec3<f32>
) -> f32 {
    let contribution = sample_radiance * brdf_value;
    let luminance = contribution.x * 0.2126 + contribution.y * 0.7152 + contribution.z * 0.0722;
    return max(luminance, 0.0);
}

fn sample_cone_uniform(u1: f32, u2: f32, cos_theta_max: f32, tangent: vec3<f32>, bitangent: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let cos_theta = (1.0 - u1) + u1 * cos_theta_max;
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let phi = u2 * 2.0 * PI;
    
    let local_dir = vec3<f32>(
        cos(phi) * sin_theta,
        sin(phi) * sin_theta,
        cos_theta
    );
    
    return normalize(tangent * local_dir.x + bitangent * local_dir.y + normal * local_dir.z);
}

fn cone_pdf(cos_theta_max: f32) -> f32 {
    return 1.0 / (2.0 * PI * (1.0 - cos_theta_max));
}

fn mis_weight(pdf_a: f32, pdf_b: f32) -> f32 {
    let a = pdf_a * pdf_a;
    let b = pdf_b * pdf_b;
    return a / max(a + b, 0.0001);
}

@compute @workgroup_size(64, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let probe_count = screen_probe_counter[0];
    let rays_per_probe = gi_params.screen_ray_count;
    let total_rays = probe_count * rays_per_probe;
    
    if (gid.x >= total_rays) {
        return;
    }
    
    let ray_id = gid.x;
    let probe_index = ray_id / rays_per_probe;
    
    var path = probe_path_state[ray_id];
    var shade = probe_path_shade[ray_id];
    
    // Check if path is alive
    if (path.state_u32.y == 0u) {
        return;
    }
    
    let bounce = path.state_u32.x;
    let tri_id = path.state_u32.w;
    
    // Get RNG state
    var rng_state = u32(shade.rng_sample_count_frame_stamp.x);
    if (rng_state == 0u) {
        rng_state = hash(ray_id ^ gi_params.frame_index);
    } else {
        rng_state = random_seed(rng_state);
    }
    
    let light_view_index = u32(scene_lighting_data.view_index);
    let light_view = view_buffer[light_view_index];
    let sun_dir = normalize(-light_view.view_direction.xyz);
    
    // === Handle Ray Miss (Sky) ===
    if (tri_id == 0xffffffffu) {
        let ray_dir = path.direction_tmax.xyz;
        
        // Evaluate environment radiance
        let sky_radiance = evaluate_environment(
            ray_dir, 
            sun_dir, 
            scene_lighting_data,
            skybox_texture
        );
        
        // Apply MIS if ray hit near sun disk (BRDF-generated ray hitting environment)
        var sky_contribution = sky_radiance;
        
        if (bounce > 0u) {
            let sun_angular_radius = scene_lighting_data.sunlight_angular_radius;
            let cos_theta_max = cos(sun_angular_radius);
            let angle_to_sun = dot(ray_dir, sun_dir);
            
            if (angle_to_sun > cos_theta_max) {
                let ray_source_pdf = shade.path_weight.w;
                let sun_sample_pdf = cone_pdf(cos_theta_max);
                let mis_w = mis_weight(ray_source_pdf, sun_sample_pdf);
                sky_contribution *= mis_w;
            }
        }
        
        // Add sky contribution weighted by path throughput
        shade.throughput += vec4<f32>(sky_contribution * shade.path_weight.xyz, 0.0);
        
        // Mark path as dead
        path.state_u32.y = 0u;
        
        probe_path_state[ray_id] = path;
        probe_path_shade[ray_id] = shade;
        return;
    }
    
    // === Handle Ray Hit ===
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
    let num_lights = light_count_buffer[0];
    
    // === EMISSIVE CONTRIBUTION ===
    if (emissive > 0.0) {
        let emissive_radiance = emissive * albedo;
        
        if (bounce > 0u) {
            // Indirect hit: Apply distance-aware attenuation
            let hit_distance = max(path.origin_tmin.w, 0.01);
            let ray_source_pdf = shade.path_weight.w;
            let raw_contribution = emissive_radiance * shade.path_weight.xyz;
            let contribution_luminance = raw_contribution.x * 0.2126 + raw_contribution.y * 0.7152 + raw_contribution.z * 0.0722;
            
            let distance_factor = 1.0 / hit_distance;
            let max_contribution = emissive * PI * distance_factor;
            let scale = min(1.0, (max_contribution * ray_source_pdf) / max(contribution_luminance, 0.001));
            
            let emissive_contribution = raw_contribution * scale;
            shade.throughput += vec4f(emissive_contribution, 0.0);
        } else {
            // First bounce: full contribution
            let emissive_contribution = emissive_radiance * shade.path_weight.xyz;
            shade.throughput += vec4f(emissive_contribution, 0.0);
        }
    }
    
    // === Direct Lighting (NEE) ===
    if (num_lights > 0u) {
        rng_state = random_seed(rng_state);
        let light_idx = u32(rand_float(rng_state) * f32(num_lights)) % num_lights;
        let light = dense_lights_buffer[light_idx];
        
        let light_dir = get_light_dir(light, hit_pos);
        let attenuation = get_light_attenuation(light, hit_pos);
        
        let brdf = calculate_brdf_rt(
            n, v_dir, light_dir, albedo, roughness, metallic,
            reflectance, clear_coat, clear_coat_roughness
        );
        
        let indirect_boost = bitcast<f32>(gi_params.indirect_boost);
        let bounce_multiplier = select(1.0, indirect_boost, bounce > 0u);
        let light_contrib = brdf * light.color.rgb * light.intensity * attenuation 
            * bounce_multiplier * shade.path_weight.xyz * f32(num_lights);
        
        // Direct contribution (simplified - no shadow tracing for probes to save bandwidth)
        if (dot(n, light_dir) > 0.0) {
            shade.throughput += vec4f(light_contrib, 0.0);
        }
    }
    
    // === Query World Cache for Indirect Lighting ===
    let world_cache_size = gi_params.world_cache_size;
    let cell_size = 1.0;
    
    let cached_radiance = query_world_cache(
        hit_pos,
        n,
        &world_cache,
        world_cache_size,
        cell_size
    );
    
    // If we have cached radiance, use it; otherwise generate new path
    if (length(cached_radiance) > 0.001 && bounce > 0u) {
        // Use cached radiance
        let indirect_boost = bitcast<f32>(gi_params.indirect_boost);
        shade.throughput += vec4<f32>(cached_radiance * albedo * shade.path_weight.xyz * indirect_boost, 0.0);
        
        // Terminate path after using cache
        path.state_u32.y = 0u;
    } else {
        // Generate BRDF sampling candidates
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
            
            // Check for sun disk MIS
            let sun_angular_radius = scene_lighting_data.sunlight_angular_radius;
            let cos_theta_max = cos(sun_angular_radius);
            let angle_to_sun = dot(dir, sun_dir);
            
            var mis_w = 1.0;
            if (angle_to_sun > cos_theta_max) {
                let sun_sample_pdf = cone_pdf(cos_theta_max);
                mis_w = mis_weight(brdf_sample_pdf, sun_sample_pdf);
            }
            
            let brdf_estimate = brdf * mis_w;
            let brdf_lum = max(0.0, brdf_estimate.x * 0.2126 + brdf_estimate.y * 0.7152 + brdf_estimate.z * 0.0722);
            
            candidate_samples[num_candidates].radiance_and_target_pdf = vec4f(brdf_estimate, brdf_lum);
            candidate_samples[num_candidates].direction_and_source_pdf = vec4f(dir, brdf_sample_pdf);
            num_candidates += 1u;
        }
        
        // === Sun Disk Importance Sampling ===
        for (var i = 0u; i < num_env_samples; i = i + 1u) {
            rng_state = random_seed(rng_state);
            let r1 = rand_float(rng_state);
            rng_state = random_seed(rng_state);
            let r2 = rand_float(rng_state);
            
            let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(sun_dir.y) > 0.999);
            let tangent = normalize(cross(up, sun_dir));
            let bitangent = normalize(cross(sun_dir, tangent));
            
            let sun_angular_radius = scene_lighting_data.sunlight_angular_radius;
            let cos_theta_max = cos(sun_angular_radius);
            let sun_sample_dir = sample_cone_uniform(r1, r2, cos_theta_max, tangent, bitangent, sun_dir);
            
            let sun_cos_theta = dot(sun_sample_dir, n);
            if (sun_cos_theta > 0.0) {
                let sun_radiance = evaluate_environment(sun_sample_dir, sun_dir, scene_lighting_data, skybox_texture);
                
                let sun_brdf = calculate_brdf_rt(
                    n, v_dir, sun_sample_dir, albedo, roughness, metallic,
                    reflectance, clear_coat, clear_coat_roughness
                );
                
                let sun_pdf = cone_pdf(cos_theta_max);
                let brdf_pdf_for_sun = brdf_pdf(n, v_dir, sun_sample_dir, clamped_roughness, mis_specular_prob);
                let mis_w = mis_weight(sun_pdf, brdf_pdf_for_sun);
                
                let weighted_radiance = sun_radiance * mis_w;
                let target_pdf = compute_gi_target_pdf(weighted_radiance, sun_brdf);
                
                if (num_candidates < 8u && target_pdf > 0.0) {
                    candidate_samples[num_candidates].radiance_and_target_pdf = vec4f(weighted_radiance, target_pdf);
                    candidate_samples[num_candidates].direction_and_source_pdf = vec4f(sun_sample_dir, sun_pdf);
                    num_candidates += 1u;
                }
            }
        }
        
        // === Perform RIS on all candidates ===
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
        
        // Finalize reservoir and spawn next bounce
        if (gi_reservoir.m > 0u && bounce < max_bounces) {
            let selected_sample = candidate_samples[gi_reservoir.selected_index];
            let selected_dir = selected_sample.direction_and_source_pdf.xyz;
            let selected_brdf = calculate_brdf_rt(
                n, v_dir, selected_dir, albedo, roughness, metallic,
                reflectance, clear_coat, clear_coat_roughness
            );
            let selected_target = compute_gi_target_pdf(selected_sample.radiance_and_target_pdf.xyz, selected_brdf);
            gi_reservoir_finalize(&gi_reservoir, selected_target);
            
            // Store direction for temporal reuse
            var selected_brdf_estimate = selected_sample.radiance_and_target_pdf.xyz;
            
            // Boost importance hint if we hit an emissive
            if (emissive > 0.0) {
                let hit_distance = max(path.origin_tmin.w, 0.1);
                let proximity_boost = 1.0 / (1.0 + hit_distance);
                let emissive_importance = emissive * albedo.x * 0.2126 + emissive * albedo.y * 0.7152 + emissive * albedo.z * 0.0722;
                let boost_factor = emissive_importance * 10.0 * proximity_boost;
                selected_brdf_estimate = selected_brdf_estimate * (1.0 + boost_factor);
            }
            
            shade.reservoir_radiance_m = vec4f(selected_brdf_estimate, f32(gi_reservoir.m));
            shade.reservoir_direction_w = vec4f(selected_dir, gi_reservoir.w);
            
            // Update path weight and spawn next ray
            let brdf_weight = selected_brdf * gi_reservoir.w;
            let selected_source_pdf = selected_sample.direction_and_source_pdf.w;
            shade.path_weight = vec4f(shade.path_weight.xyz * brdf_weight, selected_source_pdf);
            
            // Continue path
            path.origin_tmin = vec4f(hit_pos + n * 0.001, 0.0001);
            path.direction_tmax = vec4f(selected_dir, 1e30);
            path.state_u32.x = bounce + 1u;
            path.state_u32.y = 1u;
            path.state_u32.w = 0xffffffffu;
        } else {
            // Max bounces reached or no valid sample
            path.state_u32.y = 0u;
        }
    }
    
    // Save updated path state
    shade.rng_sample_count_frame_stamp.x = f32(rng_state);
    shade.rng_sample_count_frame_stamp.y += 1.0;
    
    probe_path_state[ray_id] = path;
    probe_path_shade[ray_id] = shade;
}

