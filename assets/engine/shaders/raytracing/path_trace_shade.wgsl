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
    indirect_boost: u32,          // Multiplier for indirect bounces
    padding: u32,
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
    path_weight: vec4<f32>,                // xyz=throughput weight, w=source_pdf of current ray
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
    
    // Reasonable clamping to handle extreme variance
    // With proper MIS, variance should be much lower, but still cap to prevent numerical issues
    let max_weight = 200.0;
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

// Sample a uniform direction within a cone (for sun disk sampling)
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

// Compute PDF for uniform cone sampling
fn cone_pdf(cos_theta_max: f32) -> f32 {
    return 1.0 / (2.0 * PI * (1.0 - cos_theta_max));
}

// Power heuristic for MIS (balance heuristic with power=2)
fn mis_weight(pdf_a: f32, pdf_b: f32) -> f32 {
    let a = pdf_a * pdf_a;
    let b = pdf_b * pdf_b;
    return a / max(a + b, 0.0001);
}

// Check if a pixel was traced recently (current or previous frame phase)
fn is_pixel_traced_recently(coord: vec2<u32>, trace_rate: u32, frame_phase: u32) -> bool {
    if (trace_rate <= 1u) { return true; }
    
    // Check current frame phase
    let first_x_now = (frame_phase + trace_rate - (coord.y * 2u) % trace_rate) % trace_rate;
    let is_traced_now = (coord.x >= first_x_now) && ((coord.x - first_x_now) % trace_rate == 0u);
    
    if (is_traced_now) { return true; }
    
    // Check previous frame phase for recency tolerance
    let prev_phase = (frame_phase + trace_rate - 1u) % max(trace_rate, 1u);
    let first_x_prev = (prev_phase + trace_rate - (coord.y * 2u) % trace_rate) % trace_rate;
    let is_traced_prev = (coord.x >= first_x_prev) && ((coord.x - first_x_prev) % trace_rate == 0u);
    
    return is_traced_prev;
}

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(output_tex);
    let pixel_coords = compute_phased_pixel_coords(gid.x, res, pt_params.trace_rate, pt_params.frame_phase);
    
    if (pixel_coords.x >= res.x || pixel_coords.y >= res.y) { return; }

    let pixel_index = pixel_coords.y * res.x + pixel_coords.x;
    var info = path_state[pixel_index];
    var shade = path_shade[pixel_index];
    let current_bounce = info.state_u32.x;

    // === SHADOW CONTRIBUTION ===
    // Add visible light contributions directly to throughput (not reservoir)
    if (info.state_u32.z == 1u) {
        let safe_shadow_contrib = safe_clamp_vec3(info.shadow_radiance.rgb);
        shade.throughput += vec4f(safe_shadow_contrib, 0.0);
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
        let safe_sky_contrib = safe_clamp_vec3(sky_radiance * shade.path_weight.xyz);
        shade.throughput += vec4f(safe_sky_contrib, 0.0);
        
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
        
        let hit_pos = info.origin_tmin.xyz;
        let world_n = info.normal_section_index.xyz;
        var n = world_n;
        
        // Sample textures for material properties
        let prim_store = u32(info.direction_tmax.w);
        let entity_palette_base = material_table_offset[prim_store];
        let section_index = u32(info.normal_section_index.w);
        let mat_params_index = material_palette[entity_palette_base + section_index];
        let material = material_params[mat_params_index];

        let tiling = material.emission_roughness_metallic_tiling.w;
        // Reconstruct UVs if barycentrics + vertex indices were stored by hit pass
        let has_deferred_uv = (info.hit_attr1.x == 0.0 && info.hit_attr1.y == 0.0 && info.hit_attr1.z == 0.0);
        var base_uv: vec2<f32>;
        if (has_deferred_uv) {
            let v0i = u32(info.hit_attr0.x);
            let v1i = u32(info.hit_attr0.y);
            let v2i = u32(info.hit_attr0.z);
            let u_bc = info.hit_attr0.w;
            let v_bc = info.hit_attr1.w;
            let w_bc = 1.0 - u_bc - v_bc;
            let uv0 = vertex_buffer[v0i].uv.xy;
            let uv1 = vertex_buffer[v1i].uv.xy;
            let uv2 = vertex_buffer[v2i].uv.xy;
            base_uv = uv0 * w_bc + uv1 * u_bc + uv2 * v_bc;
        } else {
            base_uv = vec2f(info.hit_attr0.w, info.hit_attr1.w);
        }
        base_uv = base_uv * tiling;
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

        // Normal mapping (only if TBN provided by hit pass)
        let world_t = info.hit_attr0.xyz;
        let world_b = info.hit_attr1.xyz;
        let has_tbn = length(world_t) > 0.0001 && length(world_b) > 0.0001;
        if ((u32(material.texture_flags1.y) & 1u) != 0u && has_tbn) {
            let tbn = mat3x3<precision_float>(world_t, world_b, world_n);
            let nm = sample_handle_rgba(
                u32(material.normal_handle), base_uv,
                texture_pool_normal, lod
            ).xyz * 2.0 - 1.0;
            n = normalize(tbn * nm);
        }

        let clear_coat = 0.0;
        let clear_coat_roughness = 0.0;
        let v_dir = -normalize(info.direction_tmax.xyz);
        let n_dot_v = max(dot(v_dir, n), 0.0001);

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
        
        // === EMISSIVE CONTRIBUTION (if we hit an emissive surface) ===
        // Add emissive with variance reduction - but keep temporal stability
        if (emissive > 0.0) {
            let emissive_radiance = emissive * albedo;
            
            if (current_bounce > 0u) {
                // Indirect hit: Apply distance-aware attenuation with temporal stability
                // The formula is deterministic: same distance + PDF -> same result every frame
                let hit_distance = max(info.origin_tmin.w, 0.01); // Clamp to avoid division by zero
                let ray_source_pdf = shade.path_weight.w;
                let raw_contribution = emissive_radiance * shade.path_weight.xyz;
                let contribution_luminance = raw_contribution.x * 0.2126 + raw_contribution.y * 0.7152 + raw_contribution.z * 0.0722;
                
                // Distance-based maximum: closer emissives can contribute more
                // This naturally reduces fireflies from distant small emissives
                let distance_factor = 1.0 / hit_distance;
                let max_contribution = emissive * PI * distance_factor;
                
                // Compute stable scale factor (deterministic for same inputs)
                let scale = min(1.0, (max_contribution * ray_source_pdf) / max(contribution_luminance, 0.001));
                
                let emissive_contribution = safe_clamp_vec3(raw_contribution * scale);
                shade.throughput += vec4f(emissive_contribution, 0.0);
            } else {
                // Direct camera hit: full contribution always
                let emissive_contribution = safe_clamp_vec3(emissive_radiance * shade.path_weight.xyz);
                shade.throughput += vec4f(emissive_contribution, 0.0);
            }
        }
        
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
            
            // Compute PDF for BRDF sampling
            let brdf_sample_pdf = brdf_pdf(n, v_dir, dir, clamped_roughness, mis_specular_prob);
            
            // For BRDF samples, we don't know the radiance yet (would need to trace)
            // So we use the BRDF value as an estimate, weighted by MIS
            let brdf_lum = max(0.0, brdf.x * 0.2126 + brdf.y * 0.7152 + brdf.z * 0.0722);
            
            candidate_samples[num_candidates].radiance_and_target_pdf = vec4f(brdf, brdf_lum);
            candidate_samples[num_candidates].direction_and_source_pdf = vec4f(dir, brdf_sample_pdf);
            num_candidates += 1u;
        }
        
        // === STEP 2: Spatial Reuse - Sample neighboring pixels' reservoirs ===
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
                let neighbor_coord = vec2<u32>(u32(neighbor_x), u32(neighbor_y));
                
                // CRITICAL: Only reuse from pixels traced recently to avoid ghosting
                if (!is_pixel_traced_recently(neighbor_coord, pt_params.trace_rate, pt_params.frame_phase)) {
                    continue;
                }
                
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

        // === STEP 3: Temporal Reuse - UNBIASED approach ===
        let prev_direction = shade.reservoir_direction_w.xyz;
        let prev_m = u32(shade.reservoir_radiance_m.w);
        let prev_importance_hint = shade.reservoir_radiance_m.xyz;
        
        if (prev_m > 0u && length(prev_direction) > 0.01) {
            let prev_brdf = calculate_brdf_rt(
                n, v_dir, prev_direction, albedo, roughness, metallic,
                reflectance, clear_coat, clear_coat_roughness
            );
            
            let prev_pdf = brdf_pdf(n, v_dir, prev_direction, clamped_roughness, mis_specular_prob);
            let prev_hint_lum = max(0.0, prev_importance_hint.x * 0.2126 + prev_importance_hint.y * 0.7152 + prev_importance_hint.z * 0.0722);
            let temporal_boost = min(2.0, 1.0 + prev_hint_lum * 0.3);
            let brdf_lum = max(0.0, prev_brdf.x * 0.2126 + prev_brdf.y * 0.7152 + prev_brdf.z * 0.0722) * temporal_boost;
            
            if (num_candidates < num_max_samples && brdf_lum > 0.0) {
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
            // The BRDF estimate is stored as a hint for importance sampling
            // If we hit an emissive, boost the hint so temporal reuse favors this direction
            var selected_brdf_estimate = selected_sample.radiance_and_target_pdf.xyz;
            
            // Boost importance hint if we hit an emissive with this direction
            // Closer emissives get stronger boost (more relevant for local lighting)
            if (emissive > 0.0) {
                let hit_distance = max(info.origin_tmin.w, 0.1);
                // Proximity boost: nearby emissives are more important to cache
                // 1.0 at distance=0, 0.5 at distance=1, 0.33 at distance=2, etc.
                let proximity_boost = 1.0 / (1.0 + hit_distance);
                let emissive_importance = emissive * albedo.x * 0.2126 + emissive * albedo.y * 0.7152 + emissive * albedo.z * 0.0722;
                let boost_factor = emissive_importance * 10.0 * proximity_boost;
                selected_brdf_estimate = selected_brdf_estimate * (1.0 + boost_factor);
            }
            
            shade.reservoir_radiance_m = vec4f(selected_brdf_estimate, f32(gi_reservoir.m));
            shade.reservoir_direction_w = vec4f(selected_dir, gi_reservoir.w);
            
            // Update path weight and spawn next ray
            // DO NOT add to throughput here - that happens when the ray hits something
            let brdf_weight = selected_brdf * gi_reservoir.w;
            
            // Store the source PDF of the spawned ray in path_weight.w for future MIS
            let selected_source_pdf = selected_sample.direction_and_source_pdf.w;
            shade.path_weight = vec4f(shade.path_weight.xyz * brdf_weight, selected_source_pdf);
        
            // Kill path if we hit an emissive on first bounce (camera direct hit)
            // Emissive surfaces are light sources - they don't need indirect lighting accumulation
            // This prevents temporal instability from varying indirect contributions
            let is_first_bounce_emissive = (current_bounce == 0u) && (emissive > 0.1);
            let should_continue = (info.state_u32.x + 1u) <= pt_params.max_bounces && !is_first_bounce_emissive;
            let alive_next = select(0u, 1u, should_continue);
            
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
}
