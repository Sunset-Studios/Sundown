// =============================================================================
// Path Tracer - Shade Pass (ReSTIR GI)
// - Implements ReSTIR GI for generalized path reuse
// - Combines direct and indirect lighting in unified reservoir sampling
// - Performs temporal + spatial reuse for noise reduction
// =============================================================================
#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "lighting_common.wgsl"
#include "postprocess_common.wgsl"
#include "sky_common.wgsl"

const num_spatial_samples = 3u;
const num_temporal_samples = 1u;
const num_ris_samples = 2u;
const num_env_samples = 2u;
const num_max_samples = 8u;
const spatial_radius = 20.0;

struct PathTracerParams {
    max_bounces: u32,
    spp_per_frame: u32,
    reset_accum_flag: u32,
    use_gbuffer: u32,
    trace_rate: u32,      // 1=full res, 2=half res, 4=quarter res, etc.
    frame_phase: u32,     // cycles 0 to trace_rate-1
    ris_brdf_candidates: u32,     // Number of BRDF candidates for RIS (M)
    indirect_boost: u32,          // Multiplier for indirect bounces
};

struct PathState {
    origin_tmin: vec4<f32>,
    direction_tmax: vec4<f32>,
    normal_section_index: vec4<f32>,
    state_u32: vec4<u32>, // x=bounce, y=alive(0/1), z=shadow_flag(0/1), w=tri_id
    hit_attr0: vec4<f32>, // xyz = world_tangent, w = uv.x
    hit_attr1: vec4<f32>, // xyz = world_bitangent, w = uv.y
    shadow_origin: vec4<f32>,      // xyz = origin, w = tmin
    shadow_direction: vec4<f32>,    // xyz = direction, w = tmax
    shadow_radiance: vec4<f32>,     // rgb = light contribution, a = needs_trace flag
};

// ReSTIR GI Reservoir - stores DIRECTION ONLY (unbiased)
struct PathShade {
    path_weight: vec4<f32>,
    rng_sample_count_frame_stamp: vec4<f32>,
    throughput: vec4<f32>,
    reservoir_radiance_m: vec4<f32>,       // xyz=BRDF estimate (importance hint), w=m (sample count)
    reservoir_direction_w: vec4<f32>,      // xyz=next bounce direction, w=final weight
}

// ReSTIR GI Reservoir for path resampling
struct GIReservoir {
    selected_index: u32,
    weight_sum: f32,
    m: u32,                      // Number of samples seen
    w: f32,                      // Final weight for selected sample
};

// GI Sample - represents a complete path contribution
struct GISample {
    radiance_and_target_pdf: vec4<f32>,         // Total radiance contribution (direct + indirect) + Target PDF for this sample
    direction_and_source_pdf: vec4<f32>,        // Next bounce direction + Source PDF used to generate this sample
};

@group(1) @binding(0) var<uniform> pt_params: PathTracerParams;
@group(1) @binding(1) var<uniform> scene_lighting_data: SceneLightingData;
@group(1) @binding(2) var<storage, read_write> path_state: array<PathState>;
@group(1) @binding(3) var<storage, read_write> path_shade: array<PathShade>;
@group(1) @binding(4) var<storage, read> material_params: array<StandardMaterialParams>;
@group(1) @binding(5) var<storage, read> material_table_offset: array<u32>;
@group(1) @binding(6) var<storage, read> material_palette: array<u32>;
@group(1) @binding(7) var<storage, read> dense_lights_buffer: array<Light>;
@group(1) @binding(8) var<storage, read> light_count_buffer: array<u32>;
@group(1) @binding(9) var texture_pool_albedo: texture_2d_array<f32>;
@group(1) @binding(10) var texture_pool_normal: texture_2d_array<f32>;
@group(1) @binding(11) var texture_pool_roughness: texture_2d_array<f32>;
@group(1) @binding(12) var texture_pool_metallic: texture_2d_array<f32>;
@group(1) @binding(13) var texture_pool_ao: texture_2d_array<f32>;
@group(1) @binding(14) var texture_pool_height: texture_2d_array<f32>;
@group(1) @binding(15) var texture_pool_specular: texture_2d_array<f32>;
@group(1) @binding(16) var texture_pool_emission: texture_2d_array<f32>;
@group(1) @binding(17) var skybox_texture: texture_cube<f32>;
@group(1) @binding(18) var output_tex: texture_storage_2d<rgba16float, write>;

fn sample_texture_or_vec4_param_handle(
    tex_handle: u32,
    uv_coords: vec2<precision_float>,
    param_val: vec4<precision_float>,
    flag: u32,
    pool: texture_2d_array<f32>,
    lod: f32
) -> vec4<precision_float> {
    if ((flag & 1u) != 0u) {
        return sample_handle_rgba(tex_handle, uv_coords, pool, lod);
    }
    return param_val;
}

fn sample_texture_or_float_param_handle(
    tex_handle: u32,
    uv_coords: vec2<precision_float>,
    param_val: precision_float,
    flag: u32,
    pool: texture_2d_array<f32>,
    lod: f32
) -> precision_float {
    if ((flag & 1u) != 0u) {
        let sampled_val = sample_handle_rgba(tex_handle, uv_coords, pool, lod);
        let channel_index = (flag >> 1u) & 3u;
        return select(select(select(sampled_val.r, sampled_val.g, channel_index == 1u), sampled_val.b, channel_index == 2u), sampled_val.a, channel_index == 3u);
    }
    return param_val;
}

// =============================================================================
// ReSTIR GI Helper Functions
// =============================================================================

fn gi_reservoir_init() -> GIReservoir {
    var reservoir: GIReservoir;
    reservoir.selected_index = 0u;
    reservoir.weight_sum = 0.0;
    reservoir.m = 0u;
    reservoir.w = 0.0;
    return reservoir;
}

// Update reservoir with a new GI sample candidate
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

// Finalize GI reservoir and compute final weight
fn gi_reservoir_finalize(
    reservoir: ptr<function, GIReservoir>,
    selected_target_pdf: f32
) {
    let contributes = (*reservoir).m > 0u && selected_target_pdf > 0.0;
    let unclamped_weight = (*reservoir).weight_sum / (f32((*reservoir).m) * max(selected_target_pdf, 0.0001));
    
    // Reasonable clamping to handle extreme variance (unbiased approach doesn't accumulate fireflies)
    let max_weight = 100.0;
    (*reservoir).w = select(
        0.0,
        min(max_weight, unclamped_weight),
        contributes
    );
}

// Compute target PDF (p-hat) for a GI sample
// p-hat = luminance(BRDF * radiance)
fn compute_gi_target_pdf(
    sample_radiance: vec3<f32>,
    brdf_value: vec3<f32>
) -> f32 {
    let contribution = sample_radiance * brdf_value;
    let luminance = contribution.x * 0.2126 + contribution.y * 0.7152 + contribution.z * 0.0722;
    return max(luminance, 0.0);
}

// Helper function to compute pixel coordinates from linear index
fn compute_pixel_coords(linear_index: u32, res: vec2<u32>, trace_rate: u32, frame_phase: u32) -> vec2<u32> {
    if (trace_rate <= 1u) {
        return vec2<u32>(linear_index % res.x, linear_index / res.x);
    }
    
    let avg_pixels_per_row = res.x / trace_rate;
    let estimated_row = linear_index / max(avg_pixels_per_row, 1u);
    
    let search_start = select(0u, estimated_row - 1u, estimated_row >= 1u);
    let search_end = min(estimated_row + 4u, res.y);
    
    var cumulative_pixels = search_start * avg_pixels_per_row;
    
    for (var y = search_start; y < search_end; y = y + 1u) {
        let first_x = (frame_phase + trace_rate - (y * 2u) % trace_rate) % trace_rate;
        let pixels_in_row = (res.x + trace_rate - 1u - first_x) / trace_rate;
        
        if (linear_index < cumulative_pixels + pixels_in_row) {
            let offset_in_row = linear_index - cumulative_pixels;
            let x = first_x + offset_in_row * trace_rate;
            return vec2<u32>(x, y);
        }
        
        cumulative_pixels += pixels_in_row;
    }
    
    return vec2<u32>(0xFFFFFFFFu, 0xFFFFFFFFu);
}

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(output_tex);
    let pixel_coords = compute_pixel_coords(gid.x, res, pt_params.trace_rate, pt_params.frame_phase);
    
    if (pixel_coords.x >= res.x || pixel_coords.y >= res.y) { return; }

    let pixel_index = pixel_coords.y * res.x + pixel_coords.x;
    var info = path_state[pixel_index];
    var shade = path_shade[pixel_index];
    let current_bounce = info.state_u32.x;

    // === SHADOW CONTRIBUTION ===
    // Add visible light contributions directly to throughput (not reservoir)
    if (info.state_u32.z == 1u) {
        shade.throughput += vec4f(info.shadow_radiance.rgb, 0.0);
        info.shadow_radiance = vec4f(0.0);
        info.state_u32.z = 0u;
    }

    let light_view_index = u32(scene_lighting_data.view_index);
    let light_view = view_buffer[light_view_index];
    let sun_dir = normalize(-light_view.view_direction.xyz);
    
    // === SKY MISS HANDLING ===
    // If ray didn't hit anything (miss), evaluate the environment (skybox or skydome)
    if (info.state_u32.w == 0xffffffffu && info.state_u32.y != 0u) {
        let ray_dir = normalize(info.direction_tmax.xyz);
        
        // Evaluate environment radiance (chooses skybox or skydome based on sky_type)
        let sky_radiance = evaluate_environment(
            ray_dir, 
            sun_dir, 
            scene_lighting_data,
            skybox_texture,
        );
        
        // Add sky contribution weighted by path throughput
        shade.throughput += vec4f(sky_radiance * shade.path_weight.xyz, 0.0);
        
        // Mark path as dead
        info.state_u32.y = 0u;
    }
    
    // Only shade if we have a valid hit
    if (info.state_u32.w != 0xffffffffu) {
        var albedo: vec3<f32>;
        var roughness: f32;
        var metallic: f32;
        var emissive: f32;
        var reflectance: f32;
        var n: vec3<f32>;
        
        let hit_pos = info.origin_tmin.xyz;
        let world_n = info.normal_section_index.xyz;
        
        // Check if this is a G-buffer hit
        let is_gbuffer_hit = (pt_params.use_gbuffer != 0u) && (info.state_u32.w == 0x0u);
        
        if (is_gbuffer_hit) {
            // G-buffer mode: Read pre-computed material properties
            albedo = info.hit_attr0.rgb;
            roughness = info.hit_attr0.w;
            metallic = info.hit_attr1.x;
            reflectance = info.hit_attr1.y;
            emissive = info.hit_attr1.z;
            n = world_n;
        } else {
            // Regular ray tracing mode: Sample textures
            let prim_store = u32(info.direction_tmax.w);
            let entity_palette_base = material_table_offset[prim_store];
            let section_index = u32(info.normal_section_index.w);
            let mat_params_index = material_palette[entity_palette_base + section_index];
            let material = material_params[mat_params_index];

            let tiling = material.emission_roughness_metallic_tiling.w;
            let base_uv = vec2f(info.hit_attr0.w, info.hit_attr1.w) * tiling;
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
            let world_t = info.hit_attr0.xyz;
            let world_b = info.hit_attr1.xyz;
            n = world_n;
            if ((u32(material.texture_flags1.y) & 1u) != 0u) {
                let tbn = mat3x3<precision_float>(world_t, world_b, world_n);
                let nm = sample_handle_rgba(
                    u32(material.normal_handle), base_uv,
                    texture_pool_normal, lod
                ).xyz * 2.0 - 1.0;
                n = normalize(tbn * nm);
            }
        }

        let clear_coat = 0.0;
        let clear_coat_roughness = 0.0;
        let v_dir = -normalize(info.direction_tmax.xyz);
        let n_dot_v = max(dot(v_dir, n), 0.0001);

        // === EMISSIVE CONTRIBUTION ===
        if (emissive > 0.0) {
            let emissive_contribution = emissive * albedo * shade.path_weight.xyz;
            shade.throughput += vec4f(emissive_contribution, 0.0);
        }

        // =============================================================================
        // === ReSTIR GI: Generate candidates + temporal/spatial reuse ===
        // =============================================================================
        
        var rng = u32(shade.rng_sample_count_frame_stamp.x);
        if (rng == 0u) { rng = hash(pixel_index ^ u32(frame_info.frame_index)); }
        else { rng = random_seed(rng); }

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
        var candidate_samples: array<GISample, num_max_samples>;
        var num_candidates = 0u;
        let num_lights = light_count_buffer[0];
        
        // === Direct Lighting with Shadow Rays (NEE) ===
        if (num_lights > 0u) {
            // Sample one light for direct lighting with shadow ray
            rng = random_seed(rng);
            let light_idx = u32(rand_float(rng) * f32(num_lights)) % num_lights;
            let light = dense_lights_buffer[light_idx];
            
            let light_dir = get_light_dir(light, hit_pos);
            let attenuation = get_light_attenuation(light, hit_pos);
            
            let brdf = calculate_brdf_rt(
                n, v_dir, light_dir, albedo, roughness, metallic,
                reflectance, clear_coat, clear_coat_roughness
            );
            
            // Compute light contribution (will be added if shadow ray doesn't hit)
            let bounce_multiplier = select(1.0, f32(pt_params.indirect_boost), current_bounce > 0u);
            let light_contrib = brdf * light.color.rgb * light.intensity * attenuation 
                * bounce_multiplier * shade.path_weight.xyz * f32(num_lights);
            
            // Setup shadow ray for visibility test
            let selected_distance = select(1e30, length(light.position.xyz - hit_pos), light.light_type != 0.0);
            info.shadow_origin = vec4f(hit_pos + n * 0.001, 0.0001);
            info.shadow_direction = vec4f(light_dir, selected_distance * 0.999);
            info.shadow_radiance = vec4f(light_contrib, 1.0);
        }
        
        // Generate BRDF sampling candidates (indirect lighting)
        for (var i = 0u; i < num_ris_samples; i = i + 1u) {
            rng = random_seed(rng);
            let r1 = rand_float(rng);
            rng = random_seed(rng);
            let r2 = rand_float(rng);
            rng = random_seed(rng);
            let r3 = rand_float(rng);
            
            var dir: vec3<f32>;
            if (use_ggx && r3 < specular_prob_if_ggx) {
                // GGX sampling w/ specular probability
                let h = importance_sample_ggx(vec2<f32>(r1, r2), n, clamped_roughness);
                dir = normalize(reflect(-v_dir, h));
            } else {
                // Cosine-weighted hemisphere sampling
                let phi = 2.0 * PI * r1;
                let cos_theta = sqrt(1.0 - r2);
                let sin_theta = sqrt(r2);
                let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.999);
                let tangent = normalize(cross(up, n));
                let bitangent = normalize(cross(n, tangent));
                let dir_local = vec3f(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
                dir = normalize(tangent * dir_local.x + bitangent * dir_local.y + n * dir_local.z);
            }
            
            // Compute BRDF
            let brdf = calculate_brdf_rt(
                n, v_dir, dir, albedo, roughness, metallic,
                reflectance, clear_coat, clear_coat_roughness
            );
            
            // Compute PDF
            let pdf = brdf_pdf(n, v_dir, dir, clamped_roughness, mis_specular_prob);
            
            // For BRDF samples, we don't know the radiance yet (would need to trace)
            // So we use the BRDF value as an estimate
            let brdf_lum = max(0.0, brdf.x * 0.2126 + brdf.y * 0.7152 + brdf.z * 0.0722);
            
            candidate_samples[num_candidates].radiance_and_target_pdf = vec4f(brdf, brdf_lum); // Will be scaled by path weight
            candidate_samples[num_candidates].direction_and_source_pdf = vec4f(dir, pdf);
            num_candidates += 1u;
        }
        
        // === Environment Sampling Candidates (for better sky convergence) ===
        // Sample the environment directly and add as candidates
        // Add 2-3 environment samples (cosine-weighted hemisphere)
        for (var i = 0u; i < num_env_samples; i = i + 1u) {
            rng = random_seed(rng);
            let r1 = rand_float(rng);
            rng = random_seed(rng);
            let r2 = rand_float(rng);
            
            // Cosine-weighted hemisphere sampling
            let phi = 2.0 * PI * r1;
            let cos_theta = sqrt(1.0 - r2);
            let sin_theta = sqrt(r2);
            
            // Build TBN frame
            let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.z) > 0.999);
            let tangent = normalize(cross(up, n));
            let bitangent = normalize(cross(n, tangent));
            let dir_local = vec3f(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
            let env_dir = normalize(tangent * dir_local.x + bitangent * dir_local.y + n * dir_local.z);
            
            // Evaluate environment radiance
            let env_radiance = evaluate_environment(env_dir, sun_dir, scene_lighting_data, skybox_texture);
            
            // Compute BRDF for this direction
            let env_brdf = calculate_brdf_rt(
                n, v_dir, env_dir, albedo, roughness, metallic,
                reflectance, clear_coat, clear_coat_roughness
            );
            
            // Compute target PDF (importance of this sample)
            let target_pdf = compute_gi_target_pdf(env_radiance, env_brdf);
            
            // Source PDF for cosine-weighted hemisphere sampling
            let source_pdf = cos_theta / PI;
            
            if (num_candidates < num_max_samples && target_pdf > 0.0) {
                // Store actual environment radiance (not just BRDF estimate!)
                candidate_samples[num_candidates].radiance_and_target_pdf = vec4f(env_radiance, target_pdf);
                candidate_samples[num_candidates].direction_and_source_pdf = vec4f(env_dir, source_pdf);
                num_candidates += 1u;
            }
        }
        
        // === STEP 2: Temporal Reuse - UNBIASED approach ===
        // Reuse previous frame's DIRECTION (not radiance) as a candidate
        // Evaluate contribution fresh each frame - this is unbiased!
        let prev_direction = shade.reservoir_direction_w.xyz;
        let prev_m = u32(shade.reservoir_radiance_m.w);
        
        if (prev_m > 0u && length(prev_direction) > 0.01) {
            // Evaluate THIS FRAME's BRDF for the previous direction
            let prev_brdf = calculate_brdf_rt(
                n, v_dir, prev_direction, albedo, roughness, metallic,
                reflectance, clear_coat, clear_coat_roughness
            );
            
            // Compute PDF for this direction
            let prev_pdf = brdf_pdf(n, v_dir, prev_direction, clamped_roughness, mis_specular_prob);
            
            // Use BRDF as radiance estimate (indirect lighting is unknown until we trace)
            let brdf_lum = max(0.0, prev_brdf.x * 0.2126 + prev_brdf.y * 0.7152 + prev_brdf.z * 0.0722);
            
            if (num_candidates < num_max_samples && brdf_lum > 0.0) {
                candidate_samples[num_candidates].radiance_and_target_pdf = vec4f(prev_brdf, brdf_lum);
                candidate_samples[num_candidates].direction_and_source_pdf = vec4f(prev_direction, prev_pdf);
                num_candidates += 1u;
            }
        }
        
        // === STEP 3: Spatial Reuse - Sample neighboring pixels' reservoirs ===
        for (var s = 0u; s < num_spatial_samples; s = s + 1u) {
            rng = random_seed(rng);
            let r1 = rand_float(rng);
            rng = random_seed(rng);
            let r2 = rand_float(rng);
            
            let theta = 2.0 * PI * r1;
            let radius = sqrt(r2) * spatial_radius;
            let offset_x = i32(cos(theta) * radius);
            let offset_y = i32(sin(theta) * radius);
            
            let neighbor_x = i32(pixel_coords.x) + offset_x;
            let neighbor_y = i32(pixel_coords.y) + offset_y;

            if (neighbor_x >= 0 && neighbor_x < i32(res.x) && neighbor_y >= 0 && neighbor_y < i32(res.y)) {
                let neighbor_index = u32(neighbor_y) * res.x + u32(neighbor_x);
                let neighbor_ps = path_state[neighbor_index];
                let neighbor_shade = path_shade[neighbor_index];
                
                // Geometric similarity test
                if (neighbor_ps.state_u32.w != 0xffffffffu) {
                    let neighbor_hit_pos = neighbor_ps.origin_tmin.xyz;
                    let neighbor_normal = neighbor_ps.normal_section_index.xyz;
                    let position_distance = length(hit_pos - neighbor_hit_pos);
                    let normal_similarity = dot(n, neighbor_normal);
                    
                    // Only reuse from geometrically similar surfaces
                    if (position_distance < 0.1 && normal_similarity > 0.9) {
                        let neighbor_direction = neighbor_shade.reservoir_direction_w.xyz;
                        let neighbor_m = u32(neighbor_shade.reservoir_radiance_m.w);
                        
                        if (neighbor_m > 0u && length(neighbor_direction) > 0.01) {
                            // UNBIASED: Evaluate neighbor's direction at CURRENT surface
                            let neighbor_brdf = calculate_brdf_rt(
                                n, v_dir, neighbor_direction, albedo, roughness, metallic,
                                reflectance, clear_coat, clear_coat_roughness
                            );
                            
                            // Compute PDF for this direction
                            let neighbor_pdf = brdf_pdf(n, v_dir, neighbor_direction, clamped_roughness, mis_specular_prob);
                            
                            let brdf_lum = max(0.0, neighbor_brdf.x * 0.2126 + neighbor_brdf.y * 0.7152 + neighbor_brdf.z * 0.0722);
                            
                            if (num_candidates < num_max_samples && brdf_lum > 0.0) {
                                candidate_samples[num_candidates].radiance_and_target_pdf = vec4f(neighbor_brdf, brdf_lum);
                                candidate_samples[num_candidates].direction_and_source_pdf = vec4f(neighbor_direction, neighbor_pdf);
                                num_candidates += 1u;
                            }
                        }
                    }
                }
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
                gi_reservoir_update(&gi_reservoir, i, ris_weight, &rng);
            }
        }
        
        // Finalize reservoir and spawn next bounce
        if (gi_reservoir.m > 0u) {
            let selected_sample = candidate_samples[gi_reservoir.selected_index];
            let selected_dir = selected_sample.direction_and_source_pdf.xyz;
            let selected_brdf = calculate_brdf_rt(
                n, v_dir, selected_dir, albedo, roughness, metallic,
                reflectance, clear_coat, clear_coat_roughness
            );
            let selected_target = compute_gi_target_pdf(selected_sample.radiance_and_target_pdf.xyz, selected_brdf);
            gi_reservoir_finalize(&gi_reservoir, selected_target);
            
            // Store only the DIRECTION for temporal reuse - UNBIASED!
            // We don't store radiance, just the direction and sample count
            // The BRDF estimate is stored just as a hint for importance sampling
            let selected_brdf_estimate = selected_sample.radiance_and_target_pdf.xyz;
            
            shade.reservoir_radiance_m = vec4f(selected_brdf_estimate, f32(gi_reservoir.m));
            shade.reservoir_direction_w = vec4f(selected_dir, gi_reservoir.w);
            
            // Update path weight and spawn next ray
            // DO NOT add to throughput here - that happens when the ray hits something
            let brdf_weight = selected_brdf * gi_reservoir.w;
            shade.path_weight = vec4f(shade.path_weight.xyz * brdf_weight, 0.0);
        
            let alive_next = select(0u, 1u, (info.state_u32.x + 1u) < pt_params.max_bounces);
            info.origin_tmin = vec4f(hit_pos + n * 0.001, 0.0001);
            info.direction_tmax = vec4f(selected_dir, 1e30);
            info.state_u32.x = info.state_u32.x + 1u;
            info.state_u32.y = alive_next;
            info.state_u32.w = 0xffffffffu;

            shade.rng_sample_count_frame_stamp.x = f32(rng);
            shade.rng_sample_count_frame_stamp.y += 1.0;
        }
    }

    // Write results
    path_state[pixel_index] = info;
    path_shade[pixel_index] = shade;
    
    let denom = max(shade.rng_sample_count_frame_stamp.y, 1.0);
    let avg = vec4f(shade.throughput.xyz / denom, 1.0);
    textureStore(output_tex, vec2<i32>(pixel_coords), avg);
}
