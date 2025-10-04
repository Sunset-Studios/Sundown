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
    max_spp: u32,
};

struct PathState {
    origin_tmin: vec4<f32>,
    direction_tmax: vec4<f32>,
    normal_section_index: vec4<f32>,
    throughput: vec4<f32>,
    state_u32: vec4<u32>, // x=bounce, y=alive(0/1)
    hit_attr0: vec4<f32>, // xyz = world_tangent, w = uv.x
    hit_attr1: vec4<f32>, // xyz = world_bitangent, w = uv.y
    rng: f32,
    sample_count: f32,
    prim_id: f32,
    frame_stamp: f32,
    // Shadow ray state for Next Event Estimation
    shadow_origin: vec4<f32>,      // xyz = origin, w = tmin
    shadow_direction: vec4<f32>,    // xyz = direction, w = tmax
    shadow_radiance: vec4<f32>,     // rgb = light contribution, a = needs_trace flag
    path_weight: vec4<f32>,         // rgb = cumulative BRDF weight along path, a = unused
};

@group(1) @binding(0) var<uniform> pt_params: PathTracerParams;
@group(1) @binding(1) var<storage, read_write> path_state: array<PathState>;
@group(1) @binding(2) var<storage, read> material_params: array<StandardMaterialParams>;
@group(1) @binding(3) var<storage, read> material_table_offset: array<u32>;
@group(1) @binding(4) var<storage, read> material_palette: array<u32>;
@group(1) @binding(5) var<storage, read> dense_lights_buffer: array<Light>;
@group(1) @binding(6) var<storage, read> light_count_buffer: array<u32>;
@group(1) @binding(7) var texture_pool_albedo: texture_2d_array<f32>;
@group(1) @binding(8) var texture_pool_normal: texture_2d_array<f32>;
@group(1) @binding(9) var texture_pool_roughness: texture_2d_array<f32>;
@group(1) @binding(10) var texture_pool_metallic: texture_2d_array<f32>;
@group(1) @binding(11) var texture_pool_ao: texture_2d_array<f32>;
@group(1) @binding(12) var texture_pool_height: texture_2d_array<f32>;
@group(1) @binding(13) var texture_pool_specular: texture_2d_array<f32>;
@group(1) @binding(14) var texture_pool_emission: texture_2d_array<f32>;
@group(1) @binding(15) var output_tex: texture_storage_2d<rgba16float, write>;

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

@compute @workgroup_size(8, 8)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(output_tex);
    if (gid.x >= res.x || gid.y >= res.y) { return; }

    let pixel_index = gid.y * res.x + gid.x;

    var info = path_state[pixel_index];
    if (pt_params.max_spp != 0u && info.sample_count >= f32(pt_params.max_spp)) {
        let denom = max(info.sample_count, 1.0);
        let avg = vec4f(info.throughput.xyz / denom, 1.0);
        textureStore(output_tex, vec2<i32>(gid.xy), avg);
        return;
    }

    var sample_rgb = vec3f(0.0);
    
    // Initialize shadow ray as inactive
    info.shadow_radiance = vec4f(0.0, 0.0, 0.0, 0.0);

    if (info.state_u32.w != 0xffffffffu) {
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

        let albedo = sample_texture_or_vec4_param_handle(
            u32(material.albedo_handle),
            base_uv,
            material.albedo,
            u32(material.texture_flags1.x),
            texture_pool_albedo,
            0.0
        ).xyz;
        let roughness = sample_texture_or_float_param_handle(
            u32(material.roughness_handle),
            base_uv,
            material.emission_roughness_metallic_tiling.y,
            u32(material.texture_flags1.z),
            texture_pool_roughness,
            0.0
        );
        let metallic = sample_texture_or_float_param_handle(
            u32(material.metallic_handle),
            base_uv,
            material.emission_roughness_metallic_tiling.z,
            u32(material.texture_flags1.w),
            texture_pool_metallic,
            0.0
        );
        let emissive = sample_texture_or_float_param_handle(
            u32(material.emission_handle),
            base_uv,
            material.emission_roughness_metallic_tiling.x,
            u32(material.texture_flags2.w),
            texture_pool_emission,
            0.0
        );
        let specular = sample_texture_or_float_param_handle(
            u32(material.specular_handle),
            base_uv,
            material.ao_height_specular.z,
            u32(material.texture_flags2.z),
            texture_pool_specular,
            0.0
        );
        let reflectance = specular * 0.0009765625 /* 1.0f / 1024 */;
        let clear_coat = 0.0;
        let clear_coat_roughness = 0.0;

        // Build world-space TBN and derive normal from normal map if enabled
        let world_t = normalize(info.hit_attr0.xyz);
        let world_b = normalize(info.hit_attr1.xyz);
        let world_n = normalize(info.normal_section_index.xyz);
        var n = world_n;
        if ((u32(material.texture_flags1.y) & 1u) != 0u) {
            let tbn = mat3x3<precision_float>(world_t, world_b, world_n);
            let nm = sample_handle_rgba(
                u32(material.normal_handle),
                base_uv,
                texture_pool_normal,
                0.0
            ).xyz * 2.0 - 1.0;
            n = normalize(tbn * nm);
        }

        // PBR BRDF evaluation for irradiance accumulation
        let hit_pos = info.origin_tmin.xyz;
        // The incoming ray direction was stored in direction_tmax.xyz by the hit pass
        let v_dir = -normalize(info.direction_tmax.xyz);

        // === EMISSIVE CONTRIBUTION ===
        // Emissive surfaces contribute light directly when hit
        // Weighted by the path throughput accumulated so far
        if (emissive > 0.0) {
            let emissive_contribution = emissive * albedo * info.path_weight.xyz;
            info.throughput += vec4f(emissive_contribution, 0.0);
        }

        // === NEXT EVENT ESTIMATION: Sample Light for Direct Lighting ===
        let num_lights = light_count_buffer[0];
        if (num_lights > 0u) {
            // Sample random light
            var rng_light = u32(info.rng);
            if (rng_light == 0u) { rng_light = hash(pixel_index ^ u32(frame_info.frame_index)); }
            else { rng_light = random_seed(rng_light); }
            let light_idx = u32(rand_float(rng_light) * f32(num_lights)) % num_lights;
            let light = dense_lights_buffer[light_idx];
            
            let light_dir = get_light_dir(light, hit_pos);

            // Compute attenuation for point/spot lights
            var attenuation = 1.0;
            if (light.light_type == 1.0) { // Point
                let light_vec = light.position.xyz - hit_pos;
                let distance_sq = dot(light_vec, light_vec);
                attenuation = compute_distance_attenuation(distance_sq, light.radius);
            } else if (light.light_type == 2.0) { // Spot
                let light_vec = light.position.xyz - hit_pos;
                let distance_sq = dot(light_vec, light_vec);
                let dist_att = compute_distance_attenuation(distance_sq, light.radius);
                
                let cos_theta = dot(-light_dir, normalize(light.direction.xyz));
                let cos_inner = cos(light.direction.w);
                let cos_outer = cos(light.outer_angle);
                let angle_att = compute_spot_angle_attenuation(cos_theta, cos_inner, cos_outer);
                
                attenuation = dist_att * angle_att;
            }
            attenuation = clamp(attenuation, 0.0, 1.0);
            
            // Compute direct lighting contribution (BRDF * light)
            let direct_brdf = calculate_brdf_rt(
                n, v_dir, light_dir,
                albedo, roughness, metallic,
                reflectance, clear_coat, clear_coat_roughness
            );
            
            // Account for light selection probability (1/num_lights) and light properties
            // Multiply by path_weight to properly accumulate weighted contributions
            let light_contrib = direct_brdf
                * light.color.rgb
                * light.intensity
                * attenuation
                * f32(num_lights)
                * info.path_weight.xyz;
            
            // Write shadow ray for visibility test
            info.shadow_origin = vec4f(hit_pos + light_dir * 0.001, 0.0001);
            
            // For directional light, use large tmax; for point/spot, compute distance to light
            let light_dist = select(1e30, length(light.position.xyz - hit_pos), light.light_type != 0.0);
            info.shadow_direction = vec4f(light_dir, light_dist * 0.999);
            info.shadow_radiance = vec4f(light_contrib, 1.0); // a=1.0 means active shadow ray
        }

        // === RUSSIAN ROULETTE ===
        var rng = u32(info.rng);
        if (rng == 0u) { rng = hash(pixel_index ^ u32(frame_info.frame_index)); }
        else { rng = random_seed(rng); }

        // Probabilistically terminate paths based on throughput
        let path_throughput = (info.path_weight.x + info.path_weight.y + info.path_weight.z) / 3.0;
        let survival_prob = clamp(path_throughput, 0.05, 0.95);

        var rng_rr = random_seed(rng);
        let rr_sample = rand_float(rng_rr);
        info.rng = f32(rng_rr);

        if (rr_sample > survival_prob) {
            // Terminate path
            info.state_u32.y = 0u;
            path_state[pixel_index] = info;
            return;
        }

        // Boost surviving paths to maintain unbiased estimate
        info.path_weight = vec4f(info.path_weight.xyz / survival_prob, 0.0);

        // === BRDF Importance Sampling Strategy ===
        // Choose between diffuse and specular sampling based on material properties
        let r1 = rand_float(rng);
        rng = random_seed(rng);
        let r2 = rand_float(rng);
        rng = random_seed(rng);
        let r3 = rand_float(rng);
        info.rng = f32(rng);

        // Clamp roughness to match what calculate_brdf_rt uses
        let clamped_roughness = clamp(roughness, 0.001, 1.0);

        // Compute F0 for Fresnel term
        let dielectric_f0 = 0.16 * reflectance * reflectance;
        let f0 = mix(vec3<f32>(dielectric_f0), albedo, metallic);

        // Schlick fresnel approximation to determine specular probability
        let n_dot_v = max(dot(n, v_dir), 0.0001);
        let fresnel = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - n_dot_v, 5.0);
        let fresnel_luminance = (fresnel.x + fresnel.y + fresnel.z) / 3.0;
        
        // Simple, robust sampling strategy to avoid variance explosion:
        // - Only use GGX sampling for smooth materials or metals
        // - For rough dielectrics, always use cosine sampling
        // This prevents rare high-weight specular samples from dominating on rough surfaces
        
        let use_ggx = (clamped_roughness < 0.3) || (metallic > 0.5);
        let specular_prob_if_ggx = clamp(fresnel_luminance, 0.01, 0.99);

        var pdf: f32;
        var brdf_weight: vec3<f32>;
        var l: vec3<f32>;

        // === COSINE-WEIGHTED HEMISPHERE SAMPLING (Diffuse) ===
        if (use_ggx && r3 < specular_prob_if_ggx) {
            let h = importance_sample_ggx(vec2<f32>(r1, r2), n, clamped_roughness);
            l = normalize(reflect(-v_dir, h));
        } else {
            let phi = 2.0 * PI * r1;
            let cos_theta = sqrt(1.0 - r2);
            let sin_theta = sqrt(r2);
            
            let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.z) > 0.999);
            let tangent = normalize(cross(up, n));
            let bitangent = normalize(cross(n, tangent));
            let dir_local = vec3f(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
            l = normalize(tangent * dir_local.x + bitangent * dir_local.y + n * dir_local.z);
        }
        
        // ALWAYS compute GGX PDF for MIS, even for rough materials
        // This prevents huge weights when cosine accidentally samples near specular peak
        let h = normalize(v_dir + l);
        let n_dot_l = max(dot(n, l), 0.0001);
        let n_dot_h = max(dot(n, h), 0.0001);
        let v_dot_h = max(dot(v_dir, h), 0.0001);
        let d = d_ggx(n_dot_h, clamped_roughness);
        let cosine_pdf = n_dot_l / PI;
        let ggx_pdf = d * n_dot_h / max(4.0 * v_dot_h, 0.0001);
        
        // Balance heuristic MIS: Always account for both PDFs
        // Even for rough materials where we never sample GGX, we still compute its PDF
        // When cosine accidentally samples near specular peak, GGX PDF will be high, increasing total PDF
        // This prevents the weight from exploding at grazing angles
        // Use a small minimum probability to ensure GGX PDF always contributes
        let min_ggx_prob = 0.05; // Always consider at least 5% chance of GGX sampling
        let mis_specular_prob = select(min_ggx_prob, specular_prob_if_ggx, use_ggx);
        pdf = mis_specular_prob * ggx_pdf + (1.0 - mis_specular_prob) * cosine_pdf;
        
        // Evaluate full BRDF
        let brdf_value = calculate_brdf_rt(
            n, v_dir, l,
            albedo, roughness, metallic,
            reflectance, clear_coat, clear_coat_roughness
        );
        
        // Weight = BRDF * cos(theta) / PDF
        brdf_weight = brdf_value / max(pdf, 0.0001);
        
        // Update path weight for next bounce (multiply accumulated weight by this bounce's BRDF contribution)
        info.path_weight = vec4f(info.path_weight.xyz * brdf_weight, 0.0);

        let alive_next = select(0u, 1u, (info.state_u32.x + 1u) < pt_params.max_bounces);
        // Spawn next ray from hit position along sampled direction
        // Offset along surface normal to avoid self-intersection
        info.origin_tmin = vec4f(hit_pos + l * 0.001, 0.0001);
        // Store direction for next bounce and reset tmax
        info.direction_tmax = vec4f(l, 1e30);
        // Store bounce count
        info.state_u32.x = info.state_u32.x + 1u;
        // Store alive flag
        info.state_u32.y = alive_next;
        // Clear mesh id for next stage
        info.state_u32.z = 0u;
        // Clear triangle id for next stage
        info.state_u32.w = 0xffffffffu;

        info.sample_count += 1.0;
    }

    // Write updated path state (shadow ray will be processed by shadow pass)
    path_state[pixel_index] = info;
    
    // Display current accumulated result
    let denom = max(info.sample_count, 1.0);
    let avg = vec4f(info.throughput.xyz / denom, 1.0);
    textureStore(output_tex, vec2<i32>(gid.xy), avg);
}


