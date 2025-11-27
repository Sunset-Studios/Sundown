// =============================================================================
// Path Tracer - Shade Pass (Simple Monte Carlo)
// =============================================================================
// Unbiased Monte Carlo path tracing with BRDF importance sampling.
// - Evaluates direct lighting via Next Event Estimation (NEE)
// - Samples BRDF to generate next bounce direction
// - Accumulates path contribution over multiple frames
// =============================================================================

#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "lighting_common.wgsl"
#include "postprocess_common.wgsl"
#include "sky_common.wgsl"

// ─────────────────────────────────────────────────────────────────────────────
// Path Tracer Parameters
// ─────────────────────────────────────────────────────────────────────────────
struct PathTracerParams {
    max_bounces: u32,
    reset_accum_flag: u32,
    use_gbuffer: u32,
    trace_rate: u32,
    frame_phase: u32,
    samples_per_pixel: u32,
    sample_index: u32,
    padding: u32,
};

// ─────────────────────────────────────────────────────────────────────────────
// Path State & Shade Structures
// ─────────────────────────────────────────────────────────────────────────────
struct PathState {
    origin_tmin: vec4<f32>,
    direction_tmax: vec4<f32>,
    normal_section_index: vec4<f32>,
    state_u32: vec4<u32>,
    hit_attr0: vec4<f32>,
    hit_attr1: vec4<f32>,
    shadow_origin: vec4<f32>,
    shadow_direction: vec4<f32>,
    shadow_radiance: vec4<f32>,
    path_weight: vec4<f32>,
    rng_sample_count: vec4<f32>,
    accumulated_radiance: vec4<f32>,
};

// ─────────────────────────────────────────────────────────────────────────────
// Bindings
// ─────────────────────────────────────────────────────────────────────────────
@group(1) @binding(0) var<uniform> pt_params: PathTracerParams;
@group(1) @binding(1) var<uniform> scene_lighting_data: SceneLightingData;
@group(1) @binding(2) var<storage, read_write> path_state: array<PathState>;
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
@group(1) @binding(16) var skybox_texture: texture_cube<f32>;
@group(1) @binding(17) var output_tex: texture_storage_2d<rgba16float, write>;

// =============================================================================
// BRDF Sampling Functions
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// Cosine-weighted hemisphere sampling
// Returns: sampled direction in world space
// ─────────────────────────────────────────────────────────────────────────────
fn sample_cosine_hemisphere(n: vec3<f32>, r1: f32, r2: f32) -> vec3<f32> {
    let phi = 2.0 * PI * r1;
    let cos_theta = sqrt(1.0 - r2);
    let sin_theta = sqrt(r2);
    
    // Build orthonormal basis around normal
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.999);
    let tangent = normalize(cross(up, n));
    let bitangent = normalize(cross(n, tangent));
    
    let dir_local = vec3f(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    return normalize(tangent * dir_local.x + bitangent * dir_local.y + n * dir_local.z);
}

// ─────────────────────────────────────────────────────────────────────────────
// GGX importance sampling for specular reflection
// Returns: half vector in world space
// ─────────────────────────────────────────────────────────────────────────────
fn sample_ggx(n: vec3<f32>, roughness: f32, r1: f32, r2: f32) -> vec3<f32> {
    let a = roughness * roughness;
    let a2 = a * a;
    
    let phi = 2.0 * PI * r1;
    let cos_theta = sqrt((1.0 - r2) / (1.0 + (a2 - 1.0) * r2));
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    
    // Build orthonormal basis
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.999);
    let tangent = normalize(cross(up, n));
    let bitangent = normalize(cross(n, tangent));
    
    let h_local = vec3f(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    return normalize(tangent * h_local.x + bitangent * h_local.y + n * h_local.z);
}

// ─────────────────────────────────────────────────────────────────────────────
// PDF for cosine-weighted hemisphere sampling
// ─────────────────────────────────────────────────────────────────────────────
fn pdf_cosine_hemisphere(n_dot_l: f32) -> f32 {
    return max(n_dot_l, 0.0) / PI;
}

// ─────────────────────────────────────────────────────────────────────────────
// PDF for GGX sampling (in terms of reflected direction)
// ─────────────────────────────────────────────────────────────────────────────
fn pdf_ggx_reflection(n: vec3<f32>, h: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let n_dot_h = max(dot(n, h), 0.0);
    let h_dot_v = max(dot(h, v), 0.0);
    
    // GGX distribution
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    let d = a2 / (PI * denom * denom);
    
    // Convert from half-vector to reflection direction
    return (d * n_dot_h) / max(4.0 * h_dot_v, 0.0001);
}

// =============================================================================
// Main Compute Shader
// =============================================================================
@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(output_tex);
    let pixel_coords = compute_phased_pixel_coords(gid.x, res, pt_params.trace_rate, pt_params.frame_phase);
    
    if (pixel_coords.x >= res.x || pixel_coords.y >= res.y) { return; }

    let pixel_index = pixel_coords.y * res.x + pixel_coords.x;
    var info = path_state[pixel_index];
    let current_bounce = info.state_u32.x;

    // ─────────────────────────────────────────────────────────────────────────
    // Add visible shadow ray contribution (from previous bounce's NEE)
    // ─────────────────────────────────────────────────────────────────────────
    if (info.state_u32.z == 1u) {
        let shadow_contrib = safe_clamp_vec3(info.shadow_radiance.rgb);
        info.accumulated_radiance += vec4f(shadow_contrib, 0.0);
        info.shadow_radiance = vec4f(0.0);
        info.state_u32.z = 0u;
    }

    // Get sun direction for sky evaluation
    let light_view_index = u32(scene_lighting_data.view_index);
    let light_view = view_buffer[light_view_index];
    let sun_dir = normalize(-light_view.view_direction.xyz);
    
    // ─────────────────────────────────────────────────────────────────────────
    // Handle sky miss - ray escaped the scene
    // ─────────────────────────────────────────────────────────────────────────
    if (info.state_u32.w == 0xffffffffu && info.state_u32.y != 0u) {
        let ray_dir = normalize(info.direction_tmax.xyz);
        
        // Evaluate environment radiance
        let sky_radiance = evaluate_environment(
            ray_dir, 
            sun_dir, 
            scene_lighting_data,
            skybox_texture,
        );
        
        // Add sky contribution weighted by path throughput
        let sky_contrib = safe_clamp_vec3(sky_radiance * info.path_weight.xyz);
        info.accumulated_radiance += vec4f(sky_contrib, 0.0);
        
        // Mark path as complete
        info.state_u32.y = 0u;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Shade valid surface hit
    // ─────────────────────────────────────────────────────────────────────────
    if (info.state_u32.w != 0xffffffffu && info.state_u32.y != 0u) {
        // =====================================================================
        // Material Property Extraction
        // =====================================================================
        var albedo: vec3<f32>;
        var roughness: f32;
        var metallic: f32;
        var emissive: f32;
        var reflectance: f32;
        
        let hit_pos = info.origin_tmin.xyz;
        let world_n = info.normal_section_index.xyz;
        var n = world_n;
        
        // Get material from texture sampling
        let prim_store = u32(info.direction_tmax.w);
        let entity_palette_base = material_table_offset[prim_store];
        let section_index = u32(info.normal_section_index.w);
        let mat_params_index = material_palette[entity_palette_base + section_index];
        let material = material_params[mat_params_index];

        let tiling = material.emission_roughness_metallic_tiling.w;
        
        // Reconstruct UVs from barycentrics if needed
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

        // Sample material textures
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
        reflectance = specular;

        // Normal mapping if TBN available
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

        // =====================================================================
        // Emissive Contribution
        // =====================================================================
        if (emissive > 0.0) {
            let emissive_radiance = emissive * albedo;
            let emissive_contrib = safe_clamp_vec3(emissive_radiance * info.path_weight.xyz);
            info.accumulated_radiance += vec4f(emissive_contrib, 0.0);
        }
        
        // =====================================================================
        // RNG Setup
        // =====================================================================
        var rng = u32(info.rng_sample_count.x);
        if (rng == 0u) { rng = hash(pixel_index ^ u32(frame_info.frame_index)); }
        else { rng = random_seed(rng); }

        // =====================================================================
        // BRDF Setup
        // =====================================================================
        let clamped_roughness = clamp(roughness, 0.04, 1.0);
        let dielectric_f0 = 0.16 * reflectance * reflectance;
        let f0 = mix(vec3<f32>(dielectric_f0), albedo, metallic);
        
        // Fresnel at normal incidence for sampling probability
        let f = f_schlick_vec3(f0, 1.0, n_dot_v);
        let fresnel_luminance = (f.x + f.y + f.z) / 3.0;
        
        // Probability of sampling specular vs diffuse
        let specular_prob = clamp(fresnel_luminance * (1.0 - clamped_roughness * 0.5), 0.1, 0.9);

        // =====================================================================
        // Direct Lighting via Next Event Estimation (NEE)
        // =====================================================================
        let num_lights = light_count_buffer[0];
        if (num_lights > 0u) {
            // Sample one random light
            rng = random_seed(rng);
            let light_idx = u32(rand_float(rng) * f32(num_lights)) % num_lights;
            let light = dense_lights_buffer[light_idx];
            
            let light_dir = get_light_dir(light, hit_pos);
            let attenuation = get_light_attenuation(light, hit_pos);
            
            // Evaluate BRDF for this light direction
            let brdf = calculate_brdf_rt(
                n, v_dir, light_dir, albedo, roughness, metallic,
                reflectance, clear_coat, clear_coat_roughness
            );
            
            // Light contribution (multiply by num_lights for unbiased estimator)
            let light_contrib = brdf * light.color.rgb * light.intensity * attenuation 
                * info.path_weight.xyz * f32(num_lights);
            
            // Setup shadow ray for visibility test
            let light_distance = select(1e30, length(light.position.xyz - hit_pos), light.light_type != 0.0);
            info.shadow_origin = vec4f(hit_pos + n * 0.001, 0.0001);
            info.shadow_direction = vec4f(light_dir, light_distance * 0.999);
            info.shadow_radiance = vec4f(light_contrib, 1.0);
        }

        // =====================================================================
        // Sample BRDF for Next Bounce Direction
        // =====================================================================
        rng = random_seed(rng);
        let r1 = rand_float(rng);
        rng = random_seed(rng);
        let r2 = rand_float(rng);
        rng = random_seed(rng);
        let r3 = rand_float(rng);
        
        var next_dir: vec3<f32>;
        var pdf: f32;
        var brdf_value: vec3<f32>;
        
        if (r3 < specular_prob) {
            // ─────────────────────────────────────────────────────────────────
            // GGX Specular Sampling
            // ─────────────────────────────────────────────────────────────────
            let h = sample_ggx(n, clamped_roughness, r1, r2);
            next_dir = normalize(reflect(-v_dir, h));
            
            // Ensure valid reflection
            if (dot(next_dir, n) <= 0.0) {
                next_dir = sample_cosine_hemisphere(n, r1, r2);
                pdf = pdf_cosine_hemisphere(max(dot(next_dir, n), 0.0));
            } else {
                let ggx_pdf = pdf_ggx_reflection(n, h, v_dir, next_dir, clamped_roughness);
                let cosine_pdf = pdf_cosine_hemisphere(max(dot(next_dir, n), 0.0));
                // MIS: combine specular and diffuse PDFs
                pdf = specular_prob * ggx_pdf + (1.0 - specular_prob) * cosine_pdf;
            }
        } else {
            // ─────────────────────────────────────────────────────────────────
            // Cosine-Weighted Diffuse Sampling
            // ─────────────────────────────────────────────────────────────────
            next_dir = sample_cosine_hemisphere(n, r1, r2);
            
            let h = normalize(v_dir + next_dir);
            let ggx_pdf = pdf_ggx_reflection(n, h, v_dir, next_dir, clamped_roughness);
            let cosine_pdf = pdf_cosine_hemisphere(max(dot(next_dir, n), 0.0));
            // MIS: combine specular and diffuse PDFs
            pdf = specular_prob * ggx_pdf + (1.0 - specular_prob) * cosine_pdf;
        }
        
        // Evaluate full BRDF for sampled direction
        brdf_value = calculate_brdf_rt(
            n, v_dir, next_dir, albedo, roughness, metallic,
            reflectance, clear_coat, clear_coat_roughness
        );
        
        // =====================================================================
        // Update Path Throughput and Spawn Next Ray
        // =====================================================================
        let safe_pdf = max(pdf, 0.0001);
        
        // Monte Carlo estimator: (BRDF * cos(theta)) / PDF
        // Note: calculate_brdf_rt already includes the cosine term
        let throughput_update = brdf_value / safe_pdf;
        let new_path_weight = info.path_weight.xyz * throughput_update;
        
        
        let reached_max_bounces = (info.state_u32.x + 1u) > pt_params.max_bounces;
        
        if (reached_max_bounces) {
            // Path terminates
            info.state_u32.y = 0u;
        } else {
            // Spawn next ray
            info.origin_tmin = vec4f(hit_pos + n * 0.001, 0.0001);
            info.direction_tmax = vec4f(next_dir, 1e30);
            info.state_u32.x = info.state_u32.x + 1u;
            info.state_u32.w = 0xffffffffu; // Mark as needing intersection test
        }

        info.path_weight = vec4f(new_path_weight, 0.0);
        info.rng_sample_count.x = f32(rng);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Write Results
    // ─────────────────────────────────────────────────────────────────────────
    path_state[pixel_index] = info;
}
