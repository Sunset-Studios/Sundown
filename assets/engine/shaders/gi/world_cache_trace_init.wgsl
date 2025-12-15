// =============================================================================
// GI-1.0 World Cache Ray Tracing - Init Pass
// - Initializes rays for active world cache cells
// - Each active cell traces 1 ray per frame to accumulate indirect radiance
// - Uses RIS-based importance sampling for better sampling
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
    // Early exit if beyond active cell count
    if (gid.x >= atomicLoad(&gi_counters.active_cache_cell_count)) {
        return;
    }
    
    // Get actual world cache cell index from compacted array
    let cell_index = compacted_indices[gid.x];
    // Check if cell is actually active (has valid data)
    if (atomicLoad(&world_cache[cell_index].fingerprint) == WORLD_CACHE_CELL_EMPTY) {
        // Mark ray as dead
        world_cache_path_state[gid.x].state_u32 = vec4<u32>(0u, 0u, 0u, 0u);
        return;
    }

    // =============================================================================
    // Extract cell surface properties from cached data
    // =============================================================================
    let position = world_cache[cell_index].position_frame.xyz;
    let normal = world_cache[cell_index].normal_rank.xyz;
    let albedo = world_cache[cell_index].albedo_roughness.xyz;
    let roughness = world_cache[cell_index].albedo_roughness.w;
    let metallic = world_cache[cell_index].material_props.x;
    let reflectance = world_cache[cell_index].material_props.y;
    let emissive = world_cache[cell_index].material_props.z;
    let clear_coat = 0.0;
    let clear_coat_roughness = 0.0;

    // Get view for camera position (for BRDF evaluation)
    let view_index = u32(frame_info.view_index);
    let camera_position = view_buffer[view_index].view_position.xyz;
    let v_dir = normalize(view_buffer[view_index].view_position.xyz - position);
    
    let frame_id = u32(gi_params.frame_index);
    
    // Initialize RNG for this cell
    var rng = u32(world_cache_path_state[gid.x].rng_rank_frame_stamp.x);
    if (rng == 0u) { rng = hash(cell_index ^ u32(frame_id)); }
    else { rng = random_seed(rng); }
    
    // =============================================================================
    // Cosine-weighted hemisphere sampling with RIS
    // Simple diffuse-like sampling for world cache (stable and efficient)
    // =============================================================================
    
    // Generate cosine-weighted hemisphere sampling candidates
    var candidate_samples: array<GISampleCandidate, num_init_ris_samples>;
    var gi_reservoir = gi_reservoir_init();
    
    for (var i = 0u; i < num_init_ris_samples; i = i + 1u) {
        rng = random_seed(rng);
        let r1 = rand_float(rng);
        rng = random_seed(rng);
        let r2 = rand_float(rng);

        // Cosine-weighted diffuse sampling
        let dir = sample_uniform_hemisphere(normal, r1, r2);
        let source_pdf = brdf_pdf(normal, v_dir, dir, roughness, 0.0);
        
        // Evaluate BRDF for sampled direction
        let brdf = calculate_brdf_rt(
            normal, v_dir, dir, albedo, roughness, metallic,
            reflectance, clear_coat, clear_coat_roughness
        );
        let target_pdf = luminance(brdf);

        let ris_weight = target_pdf / max(source_pdf, 0.0001);
        if (ris_weight > 0.0 && !isinf(ris_weight)) {
            gi_reservoir_update(&gi_reservoir, i, ris_weight, &rng, max_spatial_samples);
        }
        
        candidate_samples[i].radiance_and_target_pdf = vec4f(brdf, target_pdf);
        candidate_samples[i].direction_and_source_pdf = vec4f(dir, source_pdf);
    }
    
    // Finalize reservoir and select best direction
    let selected_index = gi_reservoir.selected_index;
    let ray_dir = candidate_samples[selected_index].direction_and_source_pdf.xyz;
    let ray_source_pdf = candidate_samples[selected_index].direction_and_source_pdf.w;
    let selected_brdf = candidate_samples[selected_index].radiance_and_target_pdf.xyz;
    let selected_target_pdf = candidate_samples[selected_index].radiance_and_target_pdf.w;
    gi_reservoir_finalize(&gi_reservoir, selected_target_pdf);
    
    let path_weight = selected_brdf * gi_reservoir.w;

    // =============================================================================
    // DIRECT LIGHTING with Visibility Rays (NEE)
    // Setup shadow rays for direct lighting from the cache cell position
    // =============================================================================
    let num_lights = light_count_buffer[0];
    if (num_lights > 0u) {
        rng = random_seed(rng);
        let light_rand = rand_float(rng);
        let light_idx = u32(light_rand * f32(num_lights)) % num_lights;
        let light = dense_lights_buffer[light_idx];
        
        let light_dir = get_light_dir(light, position);
        let attenuation = get_light_attenuation(light, position);
        
        let brdf = calculate_brdf_rt(
            normal, v_dir, light_dir, albedo, roughness, metallic,
            reflectance, clear_coat, clear_coat_roughness
        );
        
        let light_contrib = brdf * light.color.rgb * light.intensity * attenuation * f32(num_lights);
        
        // Setup shadow ray for visibility test
        let selected_light_distance = select(1e30, length(light.position.xyz - position), light.light_type != 0.0);
        world_cache_path_state[gid.x].shadow_origin = vec4f(position + normal * 0.001, f32(light_idx));
        world_cache_path_state[gid.x].shadow_direction = vec4f(light_dir, selected_light_distance * 0.999);
        world_cache_path_state[gid.x].shadow_radiance = vec4f(light_contrib, 1.0);
    } else {
        world_cache_path_state[gid.x].shadow_origin = vec4<f32>(0.0, 0.0, 0.0, -1.0);
        world_cache_path_state[gid.x].shadow_direction = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        world_cache_path_state[gid.x].shadow_radiance = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let rank = read_world_cache_cell_rank(
        position,
        normal,
        camera_position,
        u32(gi_params.world_cache_size),
        gi_params.world_cache_cell_size,
        u32(gi_params.world_cache_lod_count),
        gi_params.world_cache_cell_size 
    );
    
    // Initialize path state
    world_cache_path_state[gid.x].origin_tmin = vec4<f32>(position + normal * 0.001, 0.0001);
    world_cache_path_state[gid.x].direction_tmax = vec4<f32>(ray_dir, 1e30);
    world_cache_path_state[gid.x].normal_section_index = vec4<f32>(normal, 0.0);
    world_cache_path_state[gid.x].state_u32 = vec4<u32>(0u, 1u, 0u, 0xffffffffu);
    world_cache_path_state[gid.x].hit_attr0 = vec4<f32>(0.0);
    world_cache_path_state[gid.x].hit_attr1 = vec4<f32>(0.0);
    world_cache_path_state[gid.x].rng_rank_frame_stamp = vec4<f32>(f32(rng), f32(rank), f32(frame_id), 0.0);
    world_cache_path_state[gid.x].path_weight = vec4<f32>(path_weight, ray_source_pdf);
}

