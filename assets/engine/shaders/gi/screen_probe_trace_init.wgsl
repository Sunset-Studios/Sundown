// =============================================================================
// GI-1.0 Screen Probe Ray Tracing - Init Pass
// - Initializes rays for screen probes marked active this frame
// - Uses ReSTIR-based importance sampling guided by BRDF
// - Generates ray directions based on material properties (GGX/cosine)
// - Inactive probes (not updated this frame) get dead rays
// =============================================================================
#include "common.wgsl"
#include "gi/gi_common.wgsl"

const num_ris_samples = 2u;

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read_write> gi_counters: GICounters;
@group(1) @binding(2) var<storage, read> screen_probes: array<ScreenProbe>;
@group(1) @binding(3) var<storage, read_write> probe_path_state: array<ProbePathState>;
@group(1) @binding(4) var<storage, read_write> probe_path_shade: array<ProbePathShade>;
@group(1) @binding(5) var<storage, read> light_count_buffer: array<u32>;
@group(1) @binding(6) var<storage, read> dense_lights_buffer: array<Light>;

// =============================================================================
// ReSTIR GI Helper Functions
// =============================================================================

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

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Use total_screen_probes (derived from grid dimensions)
    // Some probes may be inactive (not updated this frame), which is fine
    let probe_count = u32(gi_params.total_screen_probes);
    let rays_per_probe = u32(gi_params.screen_ray_count);
    let total_rays = probe_count * rays_per_probe;
    
    if (gid.x >= total_rays) {
        return;
    }
    
    let probe_index = gid.x / rays_per_probe;
    let ray_index = gid.x % rays_per_probe;
    
    let probe = screen_probes[probe_index];
    let view_index = u32(frame_info.view_index);
    let view = view_buffer[view_index];
    
    // Check if probe is active (being updated this frame)
    // Inactive probes get dead rays and won't be traced
    // Also check if probe was actually spawned (has valid data from spawn pass)
    if (probe.state.x == 0.0 || probe.state.w == 0.0) {
        // Mark all rays and shade data for this inactive probe as dead
        probe_path_state[gid.x].state_u32 = vec4<u32>(0u, 0u, 0u, 0xffffffffu);
        return;
    }
    
    // =============================================================================
    // Extract probe surface properties
    // =============================================================================
    let position = probe.position_radius.xyz;
    let normal = probe.normal_frame.xyz;
    let albedo = probe.albedo_roughness.xyz;
    let roughness = probe.albedo_roughness.w;
    let metallic = probe.material_props.x;
    let reflectance = probe.material_props.y;
    let emissive = probe.material_props.z;
    let frame_id = u32(gi_params.frame_index);
    
    // Clear coat not stored in probes (assume 0)
    let clear_coat = 0.0;
    let clear_coat_roughness = 0.0;
    
    // View direction: probe is first bounce from camera, so use direction from camera to probe
    let v_dir = normalize(view.view_position.xyz - position);
    let n_dot_v = max(dot(v_dir, normal), 0.0001);
    
    // Initialize RNG for this probe ray
    var rng = u32(probe_path_state[gid.x].rng_sample_count_frame_stamp.x);
    if (rng == 0u) { rng = hash(gid.x ^ u32(gi_params.frame_index)); }
    else { rng = random_seed(rng); }
    
    // =============================================================================
    // BRDF-guided importance sampling with ReSTIR for indirect ray
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
            // GGX sampling for specular lobes
            let h = importance_sample_ggx(vec2<f32>(r1, r2), normal, clamped_roughness);
            dir = normalize(reflect(-v_dir, h));
        } else {
            // Cosine-weighted hemisphere sampling for diffuse
            let phi = 2.0 * PI * r1;
            let cos_theta = sqrt(1.0 - r2);
            let sin_theta = sqrt(r2);
            let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(normal.y) > 0.999);
            let tangent = normalize(cross(up, normal));
            let bitangent = normalize(cross(normal, tangent));
            let dir_local = vec3f(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
            dir = normalize(tangent * dir_local.x + bitangent * dir_local.y + normal * dir_local.z);
        }
        
        // Evaluate BRDF for this direction
        let brdf = calculate_brdf_rt(
            normal, v_dir, dir, albedo, roughness, metallic,
            reflectance, clear_coat, clear_coat_roughness
        );
        
        let brdf_sample_pdf = brdf_pdf(normal, v_dir, dir, clamped_roughness, mis_specular_prob);
        let brdf_lum = max(0.0, brdf.x * 0.2126 + brdf.y * 0.7152 + brdf.z * 0.0722);
        
        candidate_samples[i].radiance_and_target_pdf = vec4f(brdf, brdf_lum);
        candidate_samples[i].direction_and_source_pdf = vec4f(dir, brdf_sample_pdf);
    }
    
    // Perform RIS on candidates
    var gi_reservoir = gi_reservoir_init();
    for (var i = 0u; i < num_ris_samples; i = i + 1u) {
        let sample = candidate_samples[i];
        let brdf_for_target = calculate_brdf_rt(
            normal, v_dir, sample.direction_and_source_pdf.xyz, albedo, roughness, metallic,
            reflectance, clear_coat, clear_coat_roughness
        );
        let target_pdf = compute_gi_target_pdf(sample.radiance_and_target_pdf.xyz, brdf_for_target);
        let ris_weight = target_pdf / max(sample.direction_and_source_pdf.w, 0.0001);
        
        if (ris_weight > 0.0 && !isinf(ris_weight)) {
            gi_reservoir_update(&gi_reservoir, i, ris_weight, &rng);
        }
    }
    
    // Finalize reservoir and select best direction
    var ray_dir: vec3<f32>;
    var path_weight = vec3<f32>(1.0, 1.0, 1.0);
    var is_alive = 1u;
    
    if (gi_reservoir.m > 0u) {
        let selected_sample = candidate_samples[gi_reservoir.selected_index];
        let selected_dir = selected_sample.direction_and_source_pdf.xyz;
        let selected_brdf = calculate_brdf_rt(
            normal, v_dir, selected_dir, albedo, roughness, metallic,
            reflectance, clear_coat, clear_coat_roughness
        );
        let selected_target = compute_gi_target_pdf(selected_sample.radiance_and_target_pdf.xyz, selected_brdf);
        gi_reservoir_finalize(&gi_reservoir, selected_target);
        
        ray_dir = selected_dir;
        
        // Update path weight with BRDF and reservoir weight
        let brdf_weight = selected_brdf * gi_reservoir.w;
        path_weight = brdf_weight;
        
        // Russian Roulette: Kill paths with very low throughput to prevent underflow
        let weight_luminance = path_weight.x * 0.2126 + path_weight.y * 0.7152 + path_weight.z * 0.0722;
        let min_weight_threshold = 0.0001;
        
        if (weight_luminance < min_weight_threshold) {
            // Path weight too low - kill path and clear reservoir
            is_alive = 0u;
            probe_path_shade[gid.x].reservoir_radiance_m = vec4f(0.0);
            probe_path_shade[gid.x].reservoir_direction_w = vec4f(0.0);
        } else {
            // Store reservoir data for potential temporal reuse
            probe_path_shade[gid.x].reservoir_radiance_m = vec4f(selected_sample.radiance_and_target_pdf.xyz, f32(gi_reservoir.m));
            probe_path_shade[gid.x].reservoir_direction_w = vec4f(selected_dir, gi_reservoir.w);
        }
    } else {
        // Reservoir failed - kill path and clear reservoir to stop propagation
        is_alive = 0u;
        probe_path_shade[gid.x].reservoir_radiance_m = vec4f(0.0);
        probe_path_shade[gid.x].reservoir_direction_w = vec4f(0.0);
        
        // Still generate fallback direction for debugging (won't be traced since alive=0)
        rng = random_seed(rng);
        let u1 = rand_float(rng);
        rng = random_seed(rng);
        let u2 = rand_float(rng);
        ray_dir = sample_cosine_hemisphere(u1, u2, normal);
    }
    
    // =============================================================================
    // DIRECT LIGHTING with Shadow Rays (NEE)
    // Setup shadow rays for direct lighting from the probe surface (bounce 0)
    // =============================================================================
    let num_lights = light_count_buffer[0];
    
    if (num_lights > 0u) {
        rng = random_seed(rng);
        let light_idx = u32(rand_float(rng) * f32(num_lights)) % num_lights;
        let light = dense_lights_buffer[light_idx];
        
        let light_dir = get_light_dir(light, position);
        let attenuation = get_light_attenuation(light, position);
        
        let brdf = calculate_brdf_rt(
            normal, v_dir, light_dir, albedo, roughness, metallic,
            reflectance, clear_coat, clear_coat_roughness
        );
        
        // Direct lighting contribution (path_weight = 1 for first bounce from probe)
        // Note: We use (1,1,1) here because this is direct lighting from the probe surface itself
        let light_contrib = brdf * light.color.rgb * light.intensity * attenuation * f32(num_lights);
        
        // Setup shadow ray for visibility test
        let selected_distance = select(1e30, length(light.position.xyz - position), light.light_type != 0.0);
        probe_path_state[gid.x].shadow_origin = vec4f(position + normal * 0.001, 0.0001);
        probe_path_state[gid.x].shadow_direction = vec4f(light_dir, selected_distance * 0.999);
        probe_path_state[gid.x].shadow_radiance = vec4f(light_contrib, 1.0);
    } else {
        probe_path_state[gid.x].shadow_origin = vec4<f32>(0.0);
        probe_path_state[gid.x].shadow_direction = vec4<f32>(0.0);
        probe_path_state[gid.x].shadow_radiance = vec4<f32>(0.0);
    }
    
    // Initialize path state
    probe_path_state[gid.x].origin_tmin = vec4<f32>(position + normal * 0.001, 0.0001);
    probe_path_state[gid.x].direction_tmax = vec4<f32>(ray_dir, 1e30);
    probe_path_state[gid.x].normal_section_index = vec4<f32>(normal, 0.0);
    probe_path_state[gid.x].state_u32 = vec4<u32>(0u, is_alive, 0u, 0xffffffffu);
    probe_path_state[gid.x].hit_attr0 = vec4<f32>(0.0);
    probe_path_state[gid.x].hit_attr1 = vec4<f32>(0.0);
    probe_path_state[gid.x].rng_sample_count_frame_stamp = vec4<f32>(f32(rng), 0.0, f32(frame_id), 0.0);
    probe_path_state[gid.x].path_weight = vec4<f32>(path_weight, 1.0);
    probe_path_state[gid.x].throughput = vec4<f32>(0.0);
}

