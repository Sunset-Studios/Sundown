// =============================================================================
// Path Tracer - Shade Pass
// - Consumes TLAS hits and performs BLAS traversal
// - Shades the hit (debug color) and accumulates
// - Spawns the next ray in `path_state` for next bounce
// =============================================================================
#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "lighting_common.wgsl"

struct PathTracerParams {
    max_bounces: u32,
    spp_per_frame: u32,
    reset_accum_flag: u32,
    use_gbuffer: u32,
    trace_rate: u32,      // 1=full res, 2=half res, 4=quarter res, etc.
    frame_phase: u32,     // cycles 0 to trace_rate-1
    ris_light_candidates: u32,    // Number of light candidates for RIS (M)
    ris_brdf_candidates: u32,     // Number of BRDF candidates for RIS (M)
    indirect_boost: u32,          // Multiplier for indirect bounces
};

struct PathState {
    origin_tmin: vec4<f32>,
    direction_tmax: vec4<f32>,
    normal_section_index: vec4<f32>,
    state_u32: vec4<u32>, // x=bounce, y=alive(0/1), z=shadow_flag(0/1)
    hit_attr0: vec4<f32>, // xyz = world_tangent, w = uv.x
    hit_attr1: vec4<f32>, // xyz = world_bitangent, w = uv.y
    shadow_origin: vec4<f32>,      // xyz = origin, w = tmin
    shadow_direction: vec4<f32>,    // xyz = direction, w = tmax
    shadow_radiance: vec4<f32>,     // rgb = light contribution, a = needs_trace flag
};

struct PathShade {
    path_weight: vec4<f32>,
    rng_sample_count_frame_stamp: vec4<f32>,
    throughput: vec4<f32>,
    reservoir_light_index_m_w: vec4<f32>,  // x=light_idx, y=m, z=w, w=last_brdf_pdf
    reservoir_data: vec4<f32>,              // xyz=light_dir, w=attenuation
}

struct RISReservoir {
    selected_index: u32,
    weight_sum: f32,
    m: u32,              // Number of samples seen
    w: f32,              // Final weight for selected sample
};

@group(1) @binding(0) var<uniform> pt_params: PathTracerParams;
@group(1) @binding(1) var<storage, read_write> path_state: array<PathState>;
@group(1) @binding(2) var<storage, read_write> path_shade: array<PathShade>;
@group(1) @binding(3) var<storage, read> material_params: array<StandardMaterialParams>;
@group(1) @binding(4) var<storage, read> material_table_offset: array<u32>;
@group(1) @binding(5) var<storage, read> material_palette: array<u32>;
@group(1) @binding(6) var<storage, read> dense_lights_buffer: array<Light>;
@group(1) @binding(7) var<storage, read> light_count_buffer: array<u32>;
@group(1) @binding(8) var texture_pool_albedo: texture_2d_array<f32>;
@group(1) @binding(9) var texture_pool_normal: texture_2d_array<f32>;
@group(1) @binding(10) var texture_pool_roughness: texture_2d_array<f32>;
@group(1) @binding(11) var texture_pool_metallic: texture_2d_array<f32>;
@group(1) @binding(12) var texture_pool_ao: texture_2d_array<f32>;
@group(1) @binding(13) var texture_pool_height: texture_2d_array<f32>;
@group(1) @binding(14) var texture_pool_specular: texture_2d_array<f32>;
@group(1) @binding(15) var texture_pool_emission: texture_2d_array<f32>;
@group(1) @binding(16) var output_tex: texture_storage_2d<rgba16float, write>;

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
// Multiple Importance Sampling (MIS) Helper Functions
// =============================================================================

// Power heuristic (β=2) for MIS weight calculation
fn mis_power_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    let a2 = pdf_a * pdf_a;
    let b2 = pdf_b * pdf_b;
    return a2 / max(a2 + b2, 0.0001);
}

// Compute PDF of sampling a point light from a given position
// NOTE: This is a simplified approximation. Point/spot lights are infinitesimal,
// so BRDF sampling has near-zero probability of hitting them. This gives high
// PDF to light sampling, which is correct for MIS (light sampling should dominate).
fn compute_light_sample_pdf(
    light: Light,
    hit_pos: vec3<f32>,
    light_dir: vec3<f32>,
    num_lights: u32
) -> f32 {
    // Uniform light selection probability
    let light_selection_prob = 1.0 / max(f32(num_lights), 1.0);
    
    if (light.light_type == 0.0) {
        // Directional light: delta distribution (infinite PDF)
        return 0.0; // Special case - will be handled with MIS weight = 1.0
    } else {
        // Point/Spot light: Approximate as small area light
        // PDF increases with distance (farther = less likely to sample via BRDF)
        let light_to_point = hit_pos - light.position.xyz;
        let dist_sq = max(dot(light_to_point, light_to_point), 0.01);
        
        // Conservative: Just use light selection probability
        // This gives reasonable MIS weights without complex solid angle math
        return light_selection_prob;
    }
}

// =============================================================================
// RIS (Resampled Importance Sampling) Helper Functions
// =============================================================================
fn ris_reservoir_init() -> RISReservoir {
    var reservoir: RISReservoir;
    reservoir.selected_index = 0u;
    reservoir.weight_sum = 0.0;
    reservoir.m = 0u;
    reservoir.w = 0.0;
    return reservoir;
}

// Update reservoir with a new candidate sample
// weight = target_pdf / source_pdf (unnormalized)
// rng_state: pointer to RNG state that will be updated
fn ris_reservoir_update(
    reservoir: ptr<function, RISReservoir>,
    candidate_index: u32,
    weight: f32,
    rng_state: ptr<function, u32>
) {
    (*reservoir).weight_sum += weight;
    (*reservoir).m += 1u;
    
    // Weighted reservoir sampling: accept with probability weight / weight_sum
    *rng_state = random_seed(*rng_state);
    let xi = rand_float(*rng_state);
    if (xi * (*reservoir).weight_sum < weight) {
        (*reservoir).selected_index = candidate_index;
    }
}

// Finalize reservoir and compute final weight
fn ris_reservoir_finalize(
    reservoir: ptr<function, RISReservoir>,
    selected_target_pdf: f32
) {
    // W = (1/M) * (weight_sum / selected_target_pdf)
    // This ensures unbiased estimation
    let contributes = (*reservoir).m > 0u && selected_target_pdf > 0.0;
    let unclamped_weight = (*reservoir).weight_sum / (f32((*reservoir).m) * max(selected_target_pdf, 0.0001));
    
    // Clamp RIS weight to prevent fireflies from extreme variance
    // This introduces a small bias but dramatically reduces fireflies
    let max_ris_weight = 100.0;
    (*reservoir).w = select(
        0.0,
        min(max_ris_weight, unclamped_weight),
        contributes
    );
}

// Helper function to compute the Nth pixel that matches the frame_phase pattern
// Optimized: ~3-5 iterations max, independent of resolution
fn compute_pixel_coords(linear_index: u32, res: vec2<u32>, trace_rate: u32, frame_phase: u32) -> vec2<u32> {
    if (trace_rate <= 1u) {
        return vec2<u32>(linear_index % res.x, linear_index / res.x);
    }
    
    // Estimate which row the pixel is in
    // Most rows have approx res.x / trace_rate pixels
    let avg_pixels_per_row = res.x / trace_rate;
    let estimated_row = linear_index / max(avg_pixels_per_row, 1u);
    
    // Search a small window around the estimate (max ~5 iterations)
    let search_start = select(0u, estimated_row - 1u, estimated_row >= 1u);
    let search_end = min(estimated_row + 4u, res.y);
    
    // Estimate cumulative pixels before search_start
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
    
    // Compute actual pixel coordinates based on linear thread index and trace pattern
    let pixel_coords = compute_pixel_coords(gid.x, res, pt_params.trace_rate, pt_params.frame_phase);
    
    // Early exit if we're out of bounds
    if (pixel_coords.x >= res.x || pixel_coords.y >= res.y) { return; }

    let pixel_index = pixel_coords.y * res.x + pixel_coords.x;

    var info = path_state[pixel_index];
    var shade = path_shade[pixel_index];
    var sample_rgb = vec3f(0.0);

    let current_bounce = info.state_u32.x;

    // === SHADOW CONTRIBUTION ===
    // Shadow contribution is added directly to the throughput if there is no shadow hit
    if (info.state_u32.z == 1u) {
        shade.throughput += vec4f(info.shadow_radiance.rgb, 0.0);
        info.shadow_radiance = vec4f(0.0);
        info.state_u32.z = 0u;
    }
    
    if (info.state_u32.w != 0xffffffffu) {
        var albedo: vec3<f32>;
        var roughness: f32;
        var metallic: f32;
        var emissive: f32;
        var reflectance: f32;
        var n: vec3<f32>;
        
        let hit_pos = info.origin_tmin.xyz;
        let world_n = info.normal_section_index.xyz;
        
        // Check if this is a G-buffer hit (tri_id=0x0 means G-buffer mode on first bounce)
        let is_gbuffer_hit = (pt_params.use_gbuffer != 0u) && (info.state_u32.w == 0x0u);
        
        if (is_gbuffer_hit) {
            // G-buffer mode: Read pre-computed material properties from hit_attr fields
            // hit_attr0: rgb = albedo, w = roughness
            albedo = info.hit_attr0.rgb;
            roughness = info.hit_attr0.w;
            // hit_attr1: x = metallic, y = specular/reflectance, z = emissive, w = ao
            metallic = info.hit_attr1.x;
            reflectance = info.hit_attr1.y;
            emissive = info.hit_attr1.z;
            
            // Normal is already computed with normal mapping in G-buffer
            n = world_n;
        } else {
            // Regular ray tracing mode: Sample textures and compute material properties
            // Lookup entity row (stored in direction_tmax.w as f32) -> resolve to entity index
            let prim_store = u32(info.direction_tmax.w);
            let entity_palette_base = material_table_offset[prim_store];

            // Derive section_index by reading any triangle vertex's section from vertex_buffer
            let section_index = u32(info.normal_section_index.w);

            // Resolve material
            let mat_params_index = material_palette[entity_palette_base + section_index];
            let material = material_params[mat_params_index];

            // Use interpolated UVs produced by hit stage
            let tiling = material.emission_roughness_metallic_tiling.w;
            let base_uv = vec2f(info.hit_attr0.w, info.hit_attr1.w) * tiling;

            let lod = 0.0;

            albedo = sample_texture_or_vec4_param_handle(
                u32(material.albedo_handle),
                base_uv,
                material.albedo,
                u32(material.texture_flags1.x),
                texture_pool_albedo,
                lod
            ).xyz;
            roughness = sample_texture_or_float_param_handle(
                u32(material.roughness_handle),
                base_uv,
                material.emission_roughness_metallic_tiling.y,
                u32(material.texture_flags1.z),
                texture_pool_roughness,
                lod
            );
            metallic = sample_texture_or_float_param_handle(
                u32(material.metallic_handle),
                base_uv,
                material.emission_roughness_metallic_tiling.z,
                u32(material.texture_flags1.w),
                texture_pool_metallic,
                lod
            );
            emissive = sample_texture_or_float_param_handle(
                u32(material.emission_handle),
                base_uv,
                material.emission_roughness_metallic_tiling.x,
                u32(material.texture_flags2.w),
                texture_pool_emission,
                lod
            );
            let specular = sample_texture_or_float_param_handle(
                u32(material.specular_handle),
                base_uv,
                material.ao_height_specular.z,
                u32(material.texture_flags2.z),
                texture_pool_specular,
                lod
            );
            reflectance = specular * 0.0009765625 /* 1.0f / 1024 */;

            // Build world-space TBN and derive normal from normal map if enabled
            let world_t = info.hit_attr0.xyz;
            let world_b = info.hit_attr1.xyz;
            n = world_n;
            if ((u32(material.texture_flags1.y) & 1u) != 0u) {
                let tbn = mat3x3<precision_float>(world_t, world_b, world_n);
                let nm = sample_handle_rgba(
                    u32(material.normal_handle),
                    base_uv,
                    texture_pool_normal,
                    lod
                ).xyz * 2.0 - 1.0;
                n = normalize(tbn * nm);
            }
        }

        let clear_coat = 0.0;
        let clear_coat_roughness = 0.0;

        // PBR BRDF evaluation for irradiance accumulation
        // The view direction is stored in direction_tmax.xyz (from init pass for G-buffer, hit pass for ray tracing)
        let v_dir = -normalize(info.direction_tmax.xyz);

        // === EMISSIVE CONTRIBUTION ===
        // Emissive surfaces contribute light directly when hit
        // NOTE: For emissive geometry not in the light list, MIS isn't applicable
        // since NEE can't sample them. They only contribute via BRDF sampling.
        // TODO: Implement proper emissive mesh tracking for full MIS support
        if (emissive > 0.0) {
            let emissive_contribution = emissive * albedo * shade.path_weight.xyz;
            shade.throughput += vec4f(emissive_contribution, 0.0);
        }

        // === NEXT EVENT ESTIMATION: RIS + Inline Spatial Reuse ===
        let num_lights = light_count_buffer[0];
        if (num_lights > 0u) {
            var rng_light = u32(shade.rng_sample_count_frame_stamp.x);
            if (rng_light == 0u) { rng_light = hash(pixel_index ^ u32(frame_info.frame_index)); }
            else { rng_light = random_seed(rng_light); }
            
            // STEP 1: Generate temporal reservoir with RIS (always do this)
            let m_light_candidates = max(1u, pt_params.ris_light_candidates);
            var light_reservoir = ris_reservoir_init();
            
            var candidate_light_indices: array<u32, 8>;
            var candidate_dirs: array<vec3<f32>, 8>;
            var candidate_attenuations: array<f32, 8>;
            
            for (var i = 0u; i < min(m_light_candidates, 8u); i = i + 1u) {
                rng_light = random_seed(rng_light);
                let light_idx = u32(rand_float(rng_light) * f32(num_lights)) % num_lights;
                let candidate_light = dense_lights_buffer[light_idx];
                
                candidate_light_indices[i] = light_idx;
                
                let candidate_light_dir = get_light_dir(candidate_light, hit_pos);
                
                var candidate_attenuation = 1.0;
                if (candidate_light.light_type == 1.0) {
                    let light_vec = candidate_light.position.xyz - hit_pos;
                    let distance_sq = dot(light_vec, light_vec);
                    candidate_attenuation = compute_distance_attenuation(distance_sq, candidate_light.radius);
                } else if (candidate_light.light_type == 2.0) {
                    let light_vec = candidate_light.position.xyz - hit_pos;
                    let distance_sq = dot(light_vec, light_vec);
                    let dist_att = compute_distance_attenuation(distance_sq, candidate_light.radius);
                    
                    let cos_theta = dot(-candidate_light_dir, normalize(candidate_light.direction.xyz));
                    let cos_inner = cos(candidate_light.direction.w);
                    let cos_outer = cos(candidate_light.outer_angle);
                    let angle_att = compute_spot_angle_attenuation(cos_theta, cos_inner, cos_outer);
                    
                    candidate_attenuation = dist_att * angle_att;
                }
                candidate_attenuation = clamp(candidate_attenuation, 0.0, 1.0);
                
                let candidate_brdf = calculate_brdf_rt(
                    n, v_dir, candidate_light_dir,
                    albedo, roughness, metallic,
                    reflectance, clear_coat, clear_coat_roughness
                );
                
                let brdf_luminance = max(0.0, candidate_brdf.x * 0.2126 + candidate_brdf.y * 0.7152 + candidate_brdf.z * 0.0722);
                let target_weight = brdf_luminance * candidate_light.intensity * candidate_attenuation;
                let ris_weight = max(0.0, target_weight * f32(num_lights));
                
                candidate_dirs[i] = candidate_light_dir;
                candidate_attenuations[i] = candidate_attenuation;
                
                if (ris_weight > 0.0 && !isinf(ris_weight)) {
                    ris_reservoir_update(&light_reservoir, i, ris_weight, &rng_light);
                }
            }
            
            // Store temporal reservoir in PathShade (no extra bindings needed!)
            if (light_reservoir.m > 0u) {
                let selected_idx = light_reservoir.selected_index;
                let selected_light_idx = candidate_light_indices[selected_idx];
                let selected_light_dir = candidate_dirs[selected_idx];
                let selected_attenuation = candidate_attenuations[selected_idx];
                
                let selected_light = dense_lights_buffer[selected_light_idx];
                let selected_brdf = calculate_brdf_rt(
                    n, v_dir, selected_light_dir,
                    albedo, roughness, metallic,
                    reflectance, clear_coat, clear_coat_roughness
                );
                
                let selected_brdf_luminance = max(0.0, selected_brdf.x * 0.2126 + selected_brdf.y * 0.7152 + selected_brdf.z * 0.0722);
                let selected_target_pdf = selected_brdf_luminance * selected_light.intensity * selected_attenuation;
                
                ris_reservoir_finalize(&light_reservoir, selected_target_pdf);
                
                // Store in PathShade
                shade.reservoir_light_index_m_w = vec4f(f32(selected_light_idx), f32(light_reservoir.m), light_reservoir.w, 0.0);
                shade.reservoir_data = vec4f(selected_light_dir, selected_attenuation);
            } else {
                // No valid temporal reservoir
                shade.reservoir_light_index_m_w = vec4f(0.0, 0.0, 0.0, 0.0);
            }
            
            // STEP 2: Spatial reuse on first bounce only (combine with neighbors)
            var final_light_idx = u32(shade.reservoir_light_index_m_w.x);
            var final_light_dir = shade.reservoir_data.xyz;
            var final_attenuation = shade.reservoir_data.w;
            var final_weight = shade.reservoir_light_index_m_w.z;
            
            // Only do spatial reuse on FIRST bounce to avoid over-accumulation
            if (shade.reservoir_light_index_m_w.y > 0.0 && final_light_idx < num_lights) {
                // Read neighbors and combine using ReSTIR
                let res = textureDimensions(output_tex);
                let pixel_x = pixel_coords.x;
                let pixel_y = pixel_coords.y;
                
                // Initialize spatial reservoir with current temporal result
                // Need to compute target PDF for current sample
                let current_light = dense_lights_buffer[final_light_idx];
                let current_brdf = calculate_brdf_rt(
                    n, v_dir, final_light_dir,
                    albedo, roughness, metallic,
                    reflectance, clear_coat, clear_coat_roughness
                );
                let current_brdf_luminance = max(0.0, current_brdf.x * 0.2126 + current_brdf.y * 0.7152 + current_brdf.z * 0.0722);
                let current_target = current_brdf_luminance * current_light.intensity * final_attenuation;
                
                var spatial_reservoir_weight_sum = final_weight * shade.reservoir_light_index_m_w.y * current_target;
                var spatial_reservoir_m = u32(shade.reservoir_light_index_m_w.y);
                
                // Sample 3 neighbors for spatial reuse
                let num_spatial_samples = 3u;
                let spatial_radius = 20.0;
                
                for (var i = 0u; i < num_spatial_samples; i = i + 1u) {
                    rng_light = random_seed(rng_light);
                    let r1 = rand_float(rng_light);
                    rng_light = random_seed(rng_light);
                    let r2 = rand_float(rng_light);
                    
                    let theta = 2.0 * PI * r1;
                    let radius = sqrt(r2) * spatial_radius;
                    let offset_x = i32(cos(theta) * radius);
                    let offset_y = i32(sin(theta) * radius);
                    
                    let neighbor_x = i32(pixel_x) + offset_x;
                    let neighbor_y = i32(pixel_y) + offset_y;
                    
                    if (neighbor_x < 0 || neighbor_x >= i32(res.x) || neighbor_y < 0 || neighbor_y >= i32(res.y)) {
                        continue;
                    }
                    
                    let neighbor_index = u32(neighbor_y) * res.x + u32(neighbor_x);
                    let neighbor_ps = path_state[neighbor_index];
                    let neighbor_shade = path_shade[neighbor_index];
                    
                    // Skip if neighbor has no valid hit or reservoir
                    if (neighbor_ps.state_u32.w == 0xffffffffu || neighbor_shade.reservoir_light_index_m_w.y <= 0.0) {
                        continue;
                    }
                    
                    // Geometric similarity test
                    let neighbor_hit_pos = neighbor_ps.origin_tmin.xyz;
                    let neighbor_normal = neighbor_ps.normal_section_index.xyz;
                    let position_distance = length(hit_pos - neighbor_hit_pos);
                    let normal_similarity = dot(n, neighbor_normal);
                    
                    if (position_distance > 0.1 || normal_similarity < 0.9) {
                        continue;
                    }
                    
                    // Retrieve neighbor's sample
                    let neighbor_light_idx = u32(neighbor_shade.reservoir_light_index_m_w.x);
                    if (neighbor_light_idx >= num_lights) { continue; }
                    
                    let neighbor_light = dense_lights_buffer[neighbor_light_idx];
                    let neighbor_light_dir = neighbor_shade.reservoir_data.xyz;
                    
                    // Recompute target PDF at current pixel
                    var recomputed_attenuation = 1.0;
                    if (neighbor_light.light_type == 1.0) {
                        let light_vec = neighbor_light.position.xyz - hit_pos;
                        let distance_sq = dot(light_vec, light_vec);
                        recomputed_attenuation = compute_distance_attenuation(distance_sq, neighbor_light.radius);
                    } else if (neighbor_light.light_type == 2.0) {
                        let light_vec = neighbor_light.position.xyz - hit_pos;
                        let distance_sq = dot(light_vec, light_vec);
                        let dist_att = compute_distance_attenuation(distance_sq, neighbor_light.radius);
                        let cos_theta = dot(-neighbor_light_dir, normalize(neighbor_light.direction.xyz));
                        let cos_inner = cos(neighbor_light.direction.w);
                        let cos_outer = cos(neighbor_light.outer_angle);
                        let angle_att = compute_spot_angle_attenuation(cos_theta, cos_inner, cos_outer);
                        recomputed_attenuation = dist_att * angle_att;
                    }
                    recomputed_attenuation = clamp(recomputed_attenuation, 0.0, 1.0);
                    
                    let neighbor_brdf = calculate_brdf_rt(
                        n, v_dir, neighbor_light_dir,
                        albedo, roughness, metallic,
                        reflectance, clear_coat, clear_coat_roughness
                    );
                    
                    let neighbor_brdf_luminance = max(0.0, neighbor_brdf.x * 0.2126 + neighbor_brdf.y * 0.7152 + neighbor_brdf.z * 0.0722);
                    let neighbor_target_at_current = neighbor_brdf_luminance * neighbor_light.intensity * recomputed_attenuation;
                    
                    // Combine with ReSTIR
                    let neighbor_m = u32(neighbor_shade.reservoir_light_index_m_w.y);
                    let neighbor_w = neighbor_shade.reservoir_light_index_m_w.z;
                    let neighbor_weight = neighbor_w * f32(neighbor_m) * neighbor_target_at_current;
                    
                    spatial_reservoir_weight_sum += neighbor_weight;
                    spatial_reservoir_m += neighbor_m;
                    
                    // Weighted reservoir sampling
                    rng_light = random_seed(rng_light);
                    let xi = rand_float(rng_light);
                    if (xi * spatial_reservoir_weight_sum < neighbor_weight) {
                        final_light_idx = neighbor_light_idx;
                        final_light_dir = neighbor_light_dir;
                        final_attenuation = recomputed_attenuation;
                    }
                }
                
                // Finalize spatial reservoir
                if (spatial_reservoir_m > 0u) {
                    let final_light = dense_lights_buffer[final_light_idx];
                    let final_brdf = calculate_brdf_rt(
                        n, v_dir, final_light_dir,
                        albedo, roughness, metallic,
                        reflectance, clear_coat, clear_coat_roughness
                    );
                    let final_brdf_luminance = max(0.0, final_brdf.x * 0.2126 + final_brdf.y * 0.7152 + final_brdf.z * 0.0722);
                    let final_target = final_brdf_luminance * final_light.intensity * final_attenuation;
                    
                    let unclamped_weight = spatial_reservoir_weight_sum / (f32(spatial_reservoir_m) * max(final_target, 0.0001));
                    final_weight = min(200.0, unclamped_weight); // Clamp spatial weight
                }
            }
            
            // STEP 3: Apply lighting contribution with MIS
            if (final_weight > 0.0) {
                let final_light = dense_lights_buffer[final_light_idx];
                let final_brdf = calculate_brdf_rt(
                    n, v_dir, final_light_dir,
                    albedo, roughness, metallic,
                    reflectance, clear_coat, clear_coat_roughness
                );
                
                // Compute MIS weight to balance light sampling vs BRDF sampling
                // Compute BRDF sampling PDF for this light direction
                let clamped_roughness = clamp(roughness, 0.001, 1.0);
                let h = normalize(v_dir + final_light_dir);
                let n_dot_l = max(dot(n, final_light_dir), 0.0001);
                let n_dot_h = max(dot(n, h), 0.0001);
                let v_dot_h = max(dot(v_dir, h), 0.0001);
                let d = d_ggx(n_dot_h, clamped_roughness);
                let cosine_pdf = n_dot_l / PI;
                let ggx_pdf = d * n_dot_h / max(4.0 * v_dot_h, 0.0001);
                
                // Compute specular probability (same as used in BRDF sampling)
                let dielectric_f0 = 0.16 * reflectance * reflectance;
                let f0 = mix(vec3<f32>(dielectric_f0), albedo, metallic);
                let n_dot_v = max(dot(n, v_dir), 0.0001);
                let fresnel = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - n_dot_v, 5.0);
                let fresnel_luminance = (fresnel.x + fresnel.y + fresnel.z) / 3.0;
                let use_ggx = (clamped_roughness < 0.3) || (metallic > 0.5);
                let specular_prob_if_ggx = clamp(fresnel_luminance, 0.001, 0.99);
                let mis_specular_prob = select(0.0, specular_prob_if_ggx, use_ggx);
                
                let brdf_pdf = mis_specular_prob * ggx_pdf + (1.0 - mis_specular_prob) * cosine_pdf;
                
                // Compute light sampling PDF
                let light_pdf = compute_light_sample_pdf(final_light, hit_pos, final_light_dir, num_lights);
                
                // Apply MIS weight (power heuristic)
                // For directional lights (light_pdf = 0), use full weight since they can't be sampled via BRDF
                let mis_weight = select(
                    mis_power_heuristic(light_pdf, brdf_pdf),
                    1.0,
                    final_light.light_type == 0.0
                );
                
                let bounce_multiplier = select(1.0, f32(pt_params.indirect_boost), current_bounce > 0u);
                let light_contrib = final_brdf
                    * final_light.color.rgb
                    * final_light.intensity
                    * final_attenuation
                    * final_weight
                    * mis_weight
                    * bounce_multiplier
                    * shade.path_weight.xyz;
                
                let selected_distance = select(1e30, length(final_light.position.xyz - hit_pos), final_light.light_type != 0.0);
                info.shadow_origin = vec4f(hit_pos + n * 0.001, 0.0001);
                info.shadow_direction = vec4f(final_light_dir, selected_distance * 0.999);
                info.shadow_radiance = vec4f(light_contrib, 1.0);
            }
        }

        // === RUSSIAN ROULETTE ===
        var rng = u32(shade.rng_sample_count_frame_stamp.x);
        if (rng == 0u) { rng = hash(pixel_index ^ u32(frame_info.frame_index)); }
        else { rng = random_seed(rng); }

        // Probabilistically terminate paths based on throughput
        var survival_prob = max(shade.path_weight.x, max(shade.path_weight.y, shade.path_weight.z));
        let bounce_boost = select(1.0, 1.5, current_bounce > 0u);
        survival_prob = survival_prob * bounce_boost;

        var rng_rr = random_seed(rng);
        let rr_sample = rand_float(rng_rr);
        shade.rng_sample_count_frame_stamp.x = f32(rng_rr);

        if (rr_sample > survival_prob) {
            // Terminate path
            info.state_u32.y = 0u;
            path_state[pixel_index] = info;
            return;
        }

        // Boost surviving paths to maintain unbiased estimate
        shade.path_weight = vec4f(shade.path_weight.xyz / survival_prob, 0.0);

        // === RIS for BRDF Importance Sampling ===
        // Clamp roughness to match what calculate_brdf_rt uses
        let clamped_roughness = clamp(roughness, 0.001, 1.0);

        // Compute F0 for Fresnel term
        let dielectric_f0 = 0.16 * reflectance * reflectance;
        let f0 = mix(vec3<f32>(dielectric_f0), albedo, metallic);

        // Schlick fresnel approximation to determine specular probability
        let n_dot_v = max(dot(n, v_dir), 0.0001);
        let fresnel = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - n_dot_v, 5.0);
        let fresnel_luminance = (fresnel.x + fresnel.y + fresnel.z) / 3.0;
        
        let use_ggx = (clamped_roughness < 0.3) || (metallic > 0.5);
        let specular_prob_if_ggx = clamp(fresnel_luminance, 0.001, 0.99);
        let min_ggx_prob = 0.0;
        let mis_specular_prob = select(min_ggx_prob, specular_prob_if_ggx, use_ggx);

        // RIS: Generate M BRDF direction candidates and resample
        let m_brdf_candidates = max(1u, pt_params.ris_brdf_candidates);
        var brdf_reservoir = ris_reservoir_init();
        
        // Store candidate directions and their properties
        var candidate_dirs: array<vec3<f32>, 8>; // Support up to 8 candidates
        var candidate_pdfs: array<f32, 8>;
        
        for (var i = 0u; i < min(m_brdf_candidates, 8u); i = i + 1u) {
            let r1 = rand_float(rng);
            rng = random_seed(rng);
            let r2 = rand_float(rng);
            rng = random_seed(rng);
            let r3 = rand_float(rng);
            rng = random_seed(rng);
            
            var candidate_dir: vec3<f32>;
            
            // Sample direction using same strategy as before
            if (use_ggx && r3 < specular_prob_if_ggx) {
                let h = importance_sample_ggx(vec2<f32>(r1, r2), n, clamped_roughness);
                candidate_dir = normalize(reflect(-v_dir, h));
            } else {
                let phi = 2.0 * PI * r1;
                let cos_theta = sqrt(1.0 - r2);
                let sin_theta = sqrt(r2);
                
                let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.z) > 0.999);
                let tangent = normalize(cross(up, n));
                let bitangent = normalize(cross(n, tangent));
                let dir_local = vec3f(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
                candidate_dir = normalize(tangent * dir_local.x + bitangent * dir_local.y + n * dir_local.z);
            }
            
            // Compute PDF for this candidate
            let h = normalize(v_dir + candidate_dir);
            let n_dot_l = max(dot(n, candidate_dir), 0.0001);
            let n_dot_h = max(dot(n, h), 0.0001);
            let v_dot_h = max(dot(v_dir, h), 0.0001);
            let d = d_ggx(n_dot_h, clamped_roughness);
            let cosine_pdf = n_dot_l / PI;
            let ggx_pdf = d * n_dot_h / max(4.0 * v_dot_h, 0.0001);
            let candidate_pdf = mis_specular_prob * ggx_pdf + (1.0 - mis_specular_prob) * cosine_pdf;
            
            // Evaluate BRDF for this candidate (target function)
            let candidate_brdf = calculate_brdf_rt(
                n, v_dir, candidate_dir,
                albedo, roughness, metallic,
                reflectance, clear_coat, clear_coat_roughness
            );
            
            // Use luminance for better stability
            let target_value = max(0.0, candidate_brdf.x * 0.2126 + candidate_brdf.y * 0.7152 + candidate_brdf.z * 0.0722);
            
            // RIS weight = target / source_pdf
            let ris_weight = max(0.0, target_value / max(candidate_pdf, 0.0001));
            
            // Store candidate data
            candidate_dirs[i] = candidate_dir;
            candidate_pdfs[i] = candidate_pdf;
            
            // Update reservoir (only if weight is valid and positive)
            if (ris_weight > 0.0 && !isinf(ris_weight)) {
                ris_reservoir_update(&brdf_reservoir, i, ris_weight, &rng);
            }
        }
        
        shade.rng_sample_count_frame_stamp.x = f32(rng);
        
        // Retrieve selected direction from reservoir
        let selected_idx = brdf_reservoir.selected_index;
        let l = candidate_dirs[selected_idx];
        let pdf = candidate_pdfs[selected_idx];
        
        // Recompute target for selected sample (must match candidate computation)
        let brdf_value = calculate_brdf_rt(
            n, v_dir, l,
            albedo, roughness, metallic,
            reflectance, clear_coat, clear_coat_roughness
        );
        let selected_target = max(0.0, brdf_value.x * 0.2126 + brdf_value.y * 0.7152 + brdf_value.z * 0.0722);
        
        // Finalize reservoir
        ris_reservoir_finalize(&brdf_reservoir, selected_target);
        
        // Weight = (BRDF * cos(theta)) * RIS_weight
        let brdf_weight = brdf_value * brdf_reservoir.w;
        
        let alive_next = select(0u, 1u, (info.state_u32.x + 1u) < pt_params.max_bounces);
        // Spawn next ray from hit position along sampled direction
        info.origin_tmin = vec4f(hit_pos + n * 0.001, 0.0001);
        // Store direction for next bounce and reset tmax
        info.direction_tmax = vec4f(l, 1e30);
        // Store bounce count
        info.state_u32.x = info.state_u32.x + 1u;
        // Store alive flag
        info.state_u32.y = alive_next;
        // Clear triangle id for next stage
        info.state_u32.w = 0xffffffffu;
        // Increment sample count
        shade.rng_sample_count_frame_stamp.y += 1.0;
        // Update path weight for next bounce
        shade.path_weight = vec4f(shade.path_weight.xyz * brdf_weight, 0.0);
        // Store BRDF PDF for future MIS improvements (e.g., when emissive mesh tracking is added)
        shade.reservoir_light_index_m_w.w = pdf;
    }

    // Write updated path state (shadow ray will be processed by shadow pass)
    path_state[pixel_index] = info;
    path_shade[pixel_index] = shade;
    
    // Display current accumulated result
    let denom = max(shade.rng_sample_count_frame_stamp.y, 1.0);
    let avg = vec4f(shade.throughput.xyz / denom, 1.0);
    
    textureStore(output_tex, vec2<i32>(pixel_coords), avg);
}


