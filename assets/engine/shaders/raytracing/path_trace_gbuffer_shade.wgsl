// =============================================================================
// Path Tracer - G-Buffer Initial Shade Pass (Simple Monte Carlo)
// =============================================================================
// Dedicated pass for G-buffer first hit shading (bounce 0).
// - Runs once after init, before main bounce loop
// - Evaluates emissive and sets up direct lighting shadow rays
// - Spawns the first indirect bounce ray using BRDF sampling
// =============================================================================

#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "lighting_common.wgsl"
#include "postprocess_common.wgsl"
#include "sky_common.wgsl"

// ─────────────────────────────────────────────────────────────────────────────
// Structures
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
@group(1) @binding(1) var<storage, read_write> path_state: array<PathState>;
@group(1) @binding(2) var<storage, read> dense_lights_buffer: array<Light>;
@group(1) @binding(3) var<storage, read> light_count_buffer: array<u32>;
@group(1) @binding(4) var output_tex: texture_storage_2d<rgba16float, write>;

// =============================================================================
// BRDF Sampling Functions (duplicated for standalone compilation)
// =============================================================================

fn sample_cosine_hemisphere(n: vec3<f32>, r1: f32, r2: f32) -> vec3<f32> {
    let phi = 2.0 * PI * r1;
    let cos_theta = sqrt(1.0 - r2);
    let sin_theta = sqrt(r2);
    
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.999);
    let tangent = normalize(cross(up, n));
    let bitangent = normalize(cross(n, tangent));
    
    let dir_local = vec3f(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    return normalize(tangent * dir_local.x + bitangent * dir_local.y + n * dir_local.z);
}

fn sample_ggx(n: vec3<f32>, roughness: f32, r1: f32, r2: f32) -> vec3<f32> {
    let a = roughness * roughness;
    let a2 = a * a;
    
    let phi = 2.0 * PI * r1;
    let cos_theta = sqrt((1.0 - r2) / (1.0 + (a2 - 1.0) * r2));
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.999);
    let tangent = normalize(cross(up, n));
    let bitangent = normalize(cross(n, tangent));
    
    let h_local = vec3f(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    return normalize(tangent * h_local.x + bitangent * h_local.y + n * h_local.z);
}

fn pdf_cosine_hemisphere(n_dot_l: f32) -> f32 {
    return max(n_dot_l, 0.0) / PI;
}

fn pdf_ggx_reflection(n: vec3<f32>, h: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let n_dot_h = max(dot(n, h), 0.0);
    let h_dot_v = max(dot(h, v), 0.0);
    
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    let d = a2 / (PI * denom * denom);
    
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
    
    // Only process G-buffer hits (marked with tri_id == 0x0)
    // Skip misses (tri_id == 0xffffffff) - those are sky rays
    if (info.state_u32.w != 0x0u) { return; }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Read pre-computed material properties from G-buffer
    // ─────────────────────────────────────────────────────────────────────────
    let albedo = info.hit_attr0.rgb;
    let roughness = info.hit_attr0.w;
    let metallic = info.hit_attr1.x;
    let reflectance = info.hit_attr1.y;
    let emissive = info.hit_attr1.z;
    
    let hit_pos = info.origin_tmin.xyz;
    let n = info.normal_section_index.xyz;
    let v_dir = -normalize(info.direction_tmax.xyz);
    let n_dot_v = max(dot(v_dir, n), 0.0001);
    
    let clear_coat = 0.0;
    let clear_coat_roughness = 0.0;
    
    // ─────────────────────────────────────────────────────────────────────────
    // Emissive Contribution (camera direct hit)
    // ─────────────────────────────────────────────────────────────────────────
    if (emissive > 0.0) {
        let emissive_radiance = emissive * albedo;
        let emissive_contrib = safe_clamp_vec3(emissive_radiance * info.path_weight.xyz);
        info.accumulated_radiance += vec4f(emissive_contrib, 0.0);
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // RNG Setup
    // ─────────────────────────────────────────────────────────────────────────
    var rng = u32(info.rng_sample_count.x);
    if (rng == 0u) { rng = hash(pixel_index ^ u32(frame_info.frame_index)); }
    else { rng = random_seed(rng); }
    
    // ─────────────────────────────────────────────────────────────────────────
    // BRDF Setup
    // ─────────────────────────────────────────────────────────────────────────
    let dielectric_f0 = 0.16 * reflectance * reflectance;
    let f0 = mix(vec3<f32>(dielectric_f0), albedo, metallic);
    
    let f = f_schlick_vec3(f0, 1.0, n_dot_v);
    let fresnel_luminance = (f.x + f.y + f.z) / 3.0;

    // Probability of sampling specular vs diffuse
    let use_ggx = (roughness < 0.3) || (metallic > 0.5);
    let specular_prob_if_ggx = clamp(fresnel_luminance, 0.001, 0.99);
    let specular_prob = select(0.0, specular_prob_if_ggx, use_ggx);

    // ─────────────────────────────────────────────────────────────────────────
    // Direct Lighting via NEE (sample one light)
    // ─────────────────────────────────────────────────────────────────────────
    let num_lights = light_count_buffer[0];
    if (num_lights > 0u) {
        rng = random_seed(rng);
        let light_idx = u32(rand_float(rng) * f32(num_lights)) % num_lights;
        let light = dense_lights_buffer[light_idx];
        
        let light_dir = get_light_dir(light, hit_pos);
        let attenuation = get_light_attenuation(light, hit_pos);
        
        let brdf = calculate_brdf_rt(
            n, v_dir, light_dir, albedo, roughness, metallic,
            reflectance, clear_coat, clear_coat_roughness
        );
        
        let light_contrib = brdf * light.color.rgb * light.intensity * attenuation 
            * info.path_weight.xyz * f32(num_lights);
        
        let light_distance = select(1e30, length(light.position.xyz - hit_pos), light.light_type != 0.0);
        info.shadow_origin = vec4f(hit_pos + n * 0.001, 0.0001);
        info.shadow_direction = vec4f(light_dir, light_distance * 0.999);
        info.shadow_radiance = vec4f(light_contrib, 1.0);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Sample BRDF for First Bounce
    // ─────────────────────────────────────────────────────────────────────────
    rng = random_seed(rng);
    let r1 = rand_float(rng);
    rng = random_seed(rng);
    let r2 = rand_float(rng);
    rng = random_seed(rng);
    let r3 = rand_float(rng);
    
    var next_dir: vec3<f32>;
    var pdf: f32;
    
    if (r3 < specular_prob) {
        // GGX specular sampling
        let h = sample_ggx(n, roughness, r1, r2);
        next_dir = normalize(reflect(-v_dir, h));
    } else {
        // Cosine-weighted diffuse sampling
        next_dir = sample_uniform_hemisphere(n, r1, r2);
    }
    
    pdf = brdf_pdf(n, v_dir, next_dir, roughness, specular_prob);
    
    // Evaluate BRDF
    let brdf_value = calculate_brdf_rt(
        n, v_dir, next_dir, albedo, roughness, metallic,
        reflectance, clear_coat, clear_coat_roughness
    );
    
    // ─────────────────────────────────────────────────────────────────────────
    // Update Path Throughput and Spawn Next Ray
    // ─────────────────────────────────────────────────────────────────────────
    let safe_pdf = max(pdf, 0.0001);
    
    // Monte Carlo estimator: (BRDF * cos(theta)) / PDF
    // Note: calculate_brdf_rt already includes the cosine term
    let throughput_update = brdf_value / safe_pdf;
    let new_path_weight = info.path_weight.xyz * throughput_update;
    
    info.path_weight = vec4f(new_path_weight, 0.0);
    info.origin_tmin = vec4f(hit_pos + n * 0.001, 0.0001);
    info.direction_tmax = vec4f(next_dir, 1e30);
    info.state_u32.x = 1u; // Move to bounce 1
    info.state_u32.y = 1u; // Still alive
    info.state_u32.w = 0xffffffffu; // Mark as needing intersection test
    info.rng_sample_count.x = f32(rng);
    
    // ─────────────────────────────────────────────────────────────────────────
    // Write Results
    // ─────────────────────────────────────────────────────────────────────────
    path_state[pixel_index] = info;
}
