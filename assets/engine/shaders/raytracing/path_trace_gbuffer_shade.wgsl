// =============================================================================
// Path Tracer - G-Buffer Initial Shade Pass
// 
// Dedicated pass for G-buffer first hit shading (bounce 0)
// - Runs once after init, before main bounce loop
// - Evaluates direct lighting and sets up shadow rays
// - Spawns the first indirect bounce ray
// - Eliminates need for G-buffer checks in main path tracing kernels
// =============================================================================
#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "lighting_common.wgsl"
#include "postprocess_common.wgsl"
#include "sky_common.wgsl"

const num_ris_samples = 2u;

struct PathTracerParams {
    max_bounces: u32,
    spp_per_frame: u32,
    reset_accum_flag: u32,
    use_gbuffer: u32,
    trace_rate: u32,
    frame_phase: u32,
    indirect_boost: u32,
    padding: u32,
};

struct PathState {
    origin_tmin: vec4<f32>,
    direction_tmax: vec4<f32>,
    normal_section_index: vec4<f32>,
    state_u32: vec4<u32>, // x=bounce, y=alive(0/1), z=shadow_flag(0/1), w=tri_id
    hit_attr0: vec4<f32>, // For G-buffer: rgb=albedo, w=roughness
    hit_attr1: vec4<f32>, // For G-buffer: x=metallic, y=reflectance, z=emissive, w=unused
    shadow_origin: vec4<f32>,
    shadow_direction: vec4<f32>,
    shadow_radiance: vec4<f32>,
};

struct PathShade {
    path_weight: vec4<f32>,
    rng_sample_count_frame_stamp: vec4<f32>,
    throughput: vec4<f32>,
    reservoir_radiance_m: vec4<f32>,
    reservoir_direction_w: vec4<f32>,
}

struct GIReservoir {
    selected_index: u32,
    weight_sum: f32,
    m: u32,
    w: f32,
};

struct GISample {
    radiance_and_target_pdf: vec4<f32>,
    direction_and_source_pdf: vec4<f32>,
};

@group(1) @binding(0) var<uniform> pt_params: PathTracerParams;
@group(1) @binding(1) var<storage, read_write> path_state: array<PathState>;
@group(1) @binding(2) var<storage, read_write> path_shade: array<PathShade>;
@group(1) @binding(3) var<storage, read> dense_lights_buffer: array<Light>;
@group(1) @binding(4) var<storage, read> light_count_buffer: array<u32>;
@group(1) @binding(5) var output_tex: texture_storage_2d<rgba16float, write>;

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
    (*reservoir).w = select(0.0, min(max_weight, unclamped_weight), contributes);
}

fn compute_gi_target_pdf(sample_radiance: vec3<f32>, brdf_value: vec3<f32>) -> f32 {
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
    
    // Only process G-buffer hits (marked with tri_id == 0x0)
    // Skip misses (tri_id == 0xffffffff) - those are sky rays that will be handled in shade pass
    if (info.state_u32.w != 0x0u) { return; }
    
    // Read pre-computed material properties from G-buffer (stored in hit_attr by init pass)
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
    
    // =============================================================================
    // EMISSIVE CONTRIBUTION (camera direct hit)
    // =============================================================================
    if (emissive > 0.0) {
        let emissive_radiance = emissive * albedo;
        let emissive_contribution = safe_clamp_vec3(emissive_radiance * shade.path_weight.xyz);
        shade.throughput += vec4f(emissive_contribution, 0.0);
    }
    
    // =============================================================================
    // DIRECT LIGHTING with Shadow Rays (NEE)
    // =============================================================================
    let num_lights = light_count_buffer[0];
    
    var rng = u32(shade.rng_sample_count_frame_stamp.x);
    if (rng == 0u) { rng = hash(pixel_index ^ u32(frame_info.frame_index)); }
    else { rng = random_seed(rng); }
    
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
        
        // Direct lighting (no bounce multiplier for first bounce)
        let light_contrib = brdf * light.color.rgb * light.intensity * attenuation 
            * shade.path_weight.xyz * f32(num_lights);
        
        // Setup shadow ray for visibility test
        let selected_distance = select(1e30, length(light.position.xyz - hit_pos), light.light_type != 0.0);
        info.shadow_origin = vec4f(hit_pos + n * 0.001, 0.0001);
        info.shadow_direction = vec4f(light_dir, selected_distance * 0.999);
        info.shadow_radiance = vec4f(light_contrib, 1.0);
    }
    
    // =============================================================================
    // INDIRECT LIGHTING - Generate first bounce ray using ReSTIR
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
    
    // Generate BRDF sampling candidates
    var candidate_samples: array<GISample, num_ris_samples>;
    
    for (var i = 0u; i < num_ris_samples; i = i + 1u) {
        rng = random_seed(rng);
        let r1 = rand_float(rng);
        rng = random_seed(rng);
        let r2 = rand_float(rng);
        rng = random_seed(rng);
        let r3 = rand_float(rng);
        
        var dir: vec3<f32>;
        if (use_ggx && r3 < specular_prob_if_ggx) {
            // GGX sampling
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
        
        let brdf = calculate_brdf_rt(
            n, v_dir, dir, albedo, roughness, metallic,
            reflectance, clear_coat, clear_coat_roughness
        );
        
        let brdf_sample_pdf = brdf_pdf(n, v_dir, dir, clamped_roughness, mis_specular_prob);
        let brdf_lum = max(0.0, brdf.x * 0.2126 + brdf.y * 0.7152 + brdf.z * 0.0722);
        
        candidate_samples[i].radiance_and_target_pdf = vec4f(brdf, brdf_lum);
        candidate_samples[i].direction_and_source_pdf = vec4f(dir, brdf_sample_pdf);
    }
    
    // Perform RIS on candidates
    var gi_reservoir = gi_reservoir_init();
    for (var i = 0u; i < num_ris_samples; i = i + 1u) {
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
        
        // Store BRDF estimate for temporal reuse
        var selected_brdf_estimate = selected_sample.radiance_and_target_pdf.xyz;
        
        // Boost importance if we hit an emissive
        if (emissive > 0.0) {
            let hit_distance = max(info.origin_tmin.w, 0.1);
            let proximity_boost = 1.0 / (1.0 + hit_distance);
            let emissive_importance = emissive * albedo.x * 0.2126 + emissive * albedo.y * 0.7152 + emissive * albedo.z * 0.0722;
            let boost_factor = emissive_importance * 10.0 * proximity_boost;
            selected_brdf_estimate = selected_brdf_estimate * (1.0 + boost_factor);
        }
        
        shade.reservoir_radiance_m = vec4f(selected_brdf_estimate, f32(gi_reservoir.m));
        shade.reservoir_direction_w = vec4f(selected_dir, gi_reservoir.w);
        
        // Update path weight
        let brdf_weight = selected_brdf * gi_reservoir.w;
        let selected_source_pdf = selected_sample.direction_and_source_pdf.w;
        shade.path_weight = vec4f(shade.path_weight.xyz * brdf_weight, selected_source_pdf);
        
        // Kill path if we hit an emissive on first bounce (already contributed)
        let is_first_bounce_emissive = emissive > 0.1;
        let should_continue = (pt_params.max_bounces > 1u) && !is_first_bounce_emissive;
        let alive_next = select(0u, 1u, should_continue);
        
        // Spawn next ray (bounce 1)
        info.origin_tmin = vec4f(hit_pos + n * 0.001, 0.0001);
        info.direction_tmax = vec4f(selected_dir, 1e30);
        info.state_u32.x = 1u; // Move to bounce 1
        info.state_u32.y = alive_next;
        info.state_u32.w = 0xffffffffu; // Mark as needing intersection test
        
        shade.rng_sample_count_frame_stamp.x = f32(rng);
        shade.rng_sample_count_frame_stamp.y += 1.0;
    }
    
    // Write results
    path_state[pixel_index] = info;
    path_shade[pixel_index] = shade;
}

