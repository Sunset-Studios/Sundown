// =============================================================================
// GI-1.0 World Cache Ray Tracing - Init Pass
// - Initializes rays for active world cache cells
// - Each active cell traces 1 ray per frame to accumulate indirect radiance
// - Uses ReSTIR-based importance sampling for convergence
// - Rays spawn from cached cell position/normal from previous frame
// =============================================================================
#include "common.wgsl"
#include "gi/gi_common.wgsl"
#include "gi/world_cache_common.wgsl"
#include "raytracing/restir_common.wgsl"

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read_write> world_cache: array<WorldCacheCell>;
@group(1) @binding(2) var<storage, read> compacted_indices: array<u32>;
@group(1) @binding(3) var<storage, read> dispatch_params: array<u32>;
@group(1) @binding(4) var<storage, read_write> world_cache_path_state: array<WorldCachePathState>;
@group(1) @binding(5) var<storage, read> light_count_buffer: array<u32>;
@group(1) @binding(6) var<storage, read> dense_lights_buffer: array<Light>;
@group(1) @binding(7) var<storage, read_write> gi_counters: GICounters;

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Thread ID maps to index in compacted active cell array
    let active_index = gid.x;
    let active_cache_cell_count = atomicLoad(&gi_counters.active_cache_cell_count);
    
    // Early exit if beyond active cell count
    // Note: dispatch is already sized to active count, but check for safety
    if (active_index >= active_cache_cell_count) {
        return;
    }
    
    // Get actual world cache cell index from compacted array
    let cell_index = compacted_indices[active_index];
    // Check if cell is actually active (has valid data)
    if (atomicLoad(&world_cache[cell_index].fingerprint) == WORLD_CACHE_CELL_EMPTY) {
        // Mark ray as dead
        world_cache_path_state[active_index].state_u32.y = 0u;
        return;
    }
    
    // =============================================================================
    // Extract cell surface properties from cached data
    // =============================================================================
    let position = world_cache[cell_index].position_frame.xyz;
    let normal = world_cache[cell_index].normal_count.xyz;
    let albedo = world_cache[cell_index].albedo_roughness.xyz;
    let roughness = world_cache[cell_index].albedo_roughness.w;
    let metallic = world_cache[cell_index].material_props.x;
    let reflectance = world_cache[cell_index].material_props.y;
    let emissive = world_cache[cell_index].material_props.z;
    let clear_coat = 0.0;
    let clear_coat_roughness = 0.0;

    // Get view for camera position (for BRDF evaluation)
    let view_index = u32(frame_info.view_index);
    let view = view_buffer[view_index];
    let v_dir = normalize(view.view_position.xyz - position);
    let n_dot_v = max(dot(v_dir, normal), 0.0001);
    
    let frame_id = u32(gi_params.frame_index);
    
    // Initialize RNG for this cell
    var rng = u32(world_cache_path_state[active_index].rng_sample_count_frame_stamp.x);
    rng = select(random_seed(rng), hash(cell_index ^ u32(gi_params.frame_index)), rng == 0u); 
    
    // =============================================================================
    // Cosine-weighted hemisphere sampling with ReSTIR
    // Simple diffuse-like sampling for world cache (stable and efficient)
    // =============================================================================
    
    let clamped_roughness = max(roughness, 0.045); // Clamp to avoid numerical issues
    let mis_specular_prob = 0.0; // Pure cosine-weighted sampling (no specular)
    
    // Generate cosine-weighted hemisphere sampling candidates
    var candidate_samples: array<GISample, num_ris_samples>;
    
    for (var i = 0u; i < num_ris_samples; i = i + 1u) {
        rng = random_seed(rng);
        let r1 = rand_float(rng);
        rng = random_seed(rng);
        let r2 = rand_float(rng);
        
        // Cosine-weighted hemisphere sampling
        let phi = 2.0 * PI * r1;
        let cos_theta = sqrt(1.0 - r2);
        let sin_theta = sqrt(r2);
        let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(normal.y) > 0.999);
        let tangent = normalize(cross(up, normal));
        let bitangent = normalize(cross(normal, tangent));
        let dir_local = vec3f(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
        let dir = normalize(tangent * dir_local.x + bitangent * dir_local.y + normal * dir_local.z);
        
        // Evaluate BRDF for sampled direction
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
        path_weight = selected_brdf * gi_reservoir.w;
        
        // Russian Roulette: Kill paths with very low throughput
        let weight_luminance = path_weight.x * 0.2126 + path_weight.y * 0.7152 + path_weight.z * 0.0722;
        let min_weight_threshold = 0.0001;
        
        is_alive = select(0u, 1u, weight_luminance >= min_weight_threshold);
    } else {
        // Reservoir failed - kill path
        is_alive = 0u;
        
        // Generate fallback direction
        rng = random_seed(rng);
        let u1 = rand_float(rng);
        rng = random_seed(rng);
        let u2 = rand_float(rng);
        ray_dir = sample_cosine_hemisphere(u1, u2, normal);
    }

    // =============================================================================
    // DIRECT LIGHTING with Visibility Rays (NEE)
    // Setup shadow rays for direct lighting from the cache cell position
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
        
        let light_contrib = brdf * light.color.rgb * light.intensity * attenuation * f32(num_lights);
        
        // Setup shadow ray for visibility test
        let selected_distance = select(1e30, length(light.position.xyz - position), light.light_type != 0.0);
        world_cache_path_state[active_index].shadow_origin = vec4f(position + normal * 0.001, f32(light_idx));
        world_cache_path_state[active_index].shadow_direction = vec4f(light_dir, selected_distance * 0.999);
        world_cache_path_state[active_index].shadow_radiance = vec4f(light_contrib, 1.0);
    } else {
        world_cache_path_state[active_index].shadow_origin = vec4<f32>(0.0, 0.0, 0.0, -1.0);
        world_cache_path_state[active_index].shadow_direction = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        world_cache_path_state[active_index].shadow_radiance = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    // Initialize path state
    world_cache_path_state[active_index].origin_tmin = vec4<f32>(position + normal * 0.001, 0.0001);
    world_cache_path_state[active_index].direction_tmax = vec4<f32>(ray_dir, 1e30);
    world_cache_path_state[active_index].normal_section_index = vec4<f32>(normal, 0.0);
    world_cache_path_state[active_index].state_u32 = vec4<u32>(0u, is_alive, 0u, 0xffffffffu);
    world_cache_path_state[active_index].hit_attr0 = vec4<f32>(0.0);
    world_cache_path_state[active_index].hit_attr1 = vec4<f32>(0.0);
    world_cache_path_state[active_index].rng_sample_count_frame_stamp = vec4<f32>(f32(rng), 0.0, f32(frame_id), 0.0);
    world_cache_path_state[active_index].path_weight = vec4<f32>(path_weight, 1.0);
}

