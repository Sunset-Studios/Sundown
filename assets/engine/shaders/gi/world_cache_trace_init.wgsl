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
@group(1) @binding(8) var blue_noise: texture_2d_array<f32>;

// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                      BLUE NOISE SAMPLING                                  ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Blue noise provides low-discrepancy sampling that converges faster       ║
// ║  than white noise by distributing samples more uniformly across the       ║
// ║  sampling domain. This is especially beneficial for path tracing where    ║
// ║  we need multiple uncorrelated random values per pixel per frame.         ║
// ║                                                                           ║
// ║  We use:                                                                  ║
// ║  • 64 layers of blue noise textures for temporal decorrelation           ║
// ║  • RGBA channels provide 3 values per texel                               ║
// ║  • Cranley-Patterson rotation adds per-pixel scrambling                   ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

const BLUE_NOISE_LAYER_COUNT: u32 = 64u;

// ─────────────────────────────────────────────────────────────────────────────
// Blue Noise Sampler State
// Tracks the current sampling dimension for a given pixel/frame combination
// ─────────────────────────────────────────────────────────────────────────────
struct BlueNoiseSampler {
    base_coord: vec2<u32>,     // Base sampling coordinates (tile or pixel)
    frame_index: u32,          // Current frame for temporal variation
    dimension: u32,            // Current dimension index (auto-incremented)
    scramble: u32,             // Per-pixel scrambling value (Cranley-Patterson)
};

// ─────────────────────────────────────────────────────────────────────────────
// Initialize blue noise sampler
// Uses tile index for initial scramble to ensure each tile gets unique samples
// ─────────────────────────────────────────────────────────────────────────────
fn blue_noise_init(base_coord: vec2<u32>, frame_index: u32, seed: u32) -> BlueNoiseSampler {
    // Generate Cranley-Patterson rotation value from seed for per-sample scrambling
    let scramble = hash(seed ^ (frame_index * 0x9E3779B9u));
    return BlueNoiseSampler(base_coord, frame_index, 0u, scramble);
}

// ─────────────────────────────────────────────────────────────────────────────
// Sample next blue noise value
// Returns a value in [0, 1) with low-discrepancy properties
// Automatically advances to next dimension for subsequent calls
// ─────────────────────────────────────────────────────────────────────────────
fn blue_noise_next(sampler: ptr<function, BlueNoiseSampler>) -> f32 {
    let dim = (*sampler).dimension;
    (*sampler).dimension = dim + 1u;
    
    // Get blue noise texture dimensions
    let noise_dims = textureDimensions(blue_noise);
    
    // Select layer based on frame + dimension for temporal decorrelation
    // This ensures different dimensions sample from different noise patterns
    let layer = ((*sampler).frame_index + dim / 4u) % BLUE_NOISE_LAYER_COUNT;
    
    // Select channel (0-2) based on dimension within the layer
    let channel = dim % 3u;
    
    // Compute sample coordinates with wrapping
    // Add dimension-based offset to decorrelate different dimensions spatially
    let offset = vec2<u32>((dim * 17u) % noise_dims.x, (dim * 31u) % noise_dims.y);
    let sample_coord = vec2<i32>(
        i32(((*sampler).base_coord.x + offset.x) % noise_dims.x),
        i32(((*sampler).base_coord.y + offset.y) % noise_dims.y)
    );
    
    // Sample blue noise texture
    let noise_texel = textureLoad(blue_noise, sample_coord, i32(layer), 0);
    
    // Extract value from appropriate channel
    var noise_value: f32;
    switch (channel) {
        case 0u: { noise_value = noise_texel.r; }
        case 1u: { noise_value = noise_texel.g; }
        case 2u: { noise_value = noise_texel.b; }
        default: { noise_value = 0.0; }
    }
    
    // Apply Cranley-Patterson rotation for additional scrambling
    // This adds a per-pixel/per-frame offset, wrapping around [0, 1)
    let rotation = f32(hash((*sampler).scramble + dim)) * one_over_float_max;
    noise_value = fract(noise_value + rotation);
    
    return noise_value;
}


// =============================================================================
// BRDF Sampling Functions (duplicated for standalone compilation)
// =============================================================================

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
    let camera_position = view.view_position.xyz;
    let v_dir = normalize(view.view_position.xyz - position);
    let n_dot_v = max(dot(v_dir, normal), 0.0001);
    
    let frame_id = u32(gi_params.frame_index);
    
    // Initialize RNG for this cell
    var bn_sampler = blue_noise_init(vec2<u32>(cell_index, cell_index), frame_id, gid.x);
    
    // =============================================================================
    // Cosine-weighted hemisphere sampling with ReSTIR
    // Simple diffuse-like sampling for world cache (stable and efficient)
    // =============================================================================
    
    let clamped_roughness = clamp(roughness, 0.04, 1.0);
    let dielectric_f0 = 0.16 * reflectance * reflectance;
    let f0 = mix(vec3<f32>(dielectric_f0), albedo, metallic);
    
    let f = f_schlick_vec3(f0, 1.0, n_dot_v);
    let fresnel_luminance = (f.x + f.y + f.z) / 3.0;

    // Generate cosine-weighted hemisphere sampling candidates
    var candidate_samples: array<GISample, num_ris_samples>;
    
    for (var i = 0u; i < num_ris_samples; i = i + 1u) {
        let r1 = blue_noise_next(&bn_sampler);
        let r2 = blue_noise_next(&bn_sampler);

        // Cosine-weighted diffuse sampling
        let dir = sample_cosine_hemisphere(r1, r2, normal);
        let pdf = brdf_pdf(normal, v_dir, dir, roughness, 0.0);
        
        // Evaluate BRDF for sampled direction
        let brdf = calculate_brdf_rt(
            normal, v_dir, dir, albedo, roughness, metallic,
            reflectance, clear_coat, clear_coat_roughness
        );
        
        let brdf_lum = max(0.0, brdf.x * 0.2126 + brdf.y * 0.7152 + brdf.z * 0.0722);
        
        candidate_samples[i].radiance_and_target_pdf = vec4f(brdf, brdf_lum);
        candidate_samples[i].direction_and_source_pdf = vec4f(dir, pdf);
    }
    
    // Perform RIS on candidates
    var gi_reservoir = gi_reservoir_init();
    for (var i = 0u; i < num_ris_samples; i = i + 1u) {
        let sample = candidate_samples[i];
        let target_pdf = sample.radiance_and_target_pdf.w;
        let ris_weight = target_pdf / max(sample.direction_and_source_pdf.w, 0.0001);
        
        if (ris_weight > 0.0 && !isinf(ris_weight)) {
            let reservoir_rand = blue_noise_next(&bn_sampler);
            gi_reservoir_update_with_rand(&gi_reservoir, i, ris_weight, reservoir_rand);
        }
    }
    
    // Finalize reservoir and select best direction
    var ray_dir: vec3<f32>;
    var path_weight = vec3<f32>(1.0, 1.0, 1.0);
    var ray_source_pdf = 0.0;
    var is_alive = 1u;
    
    if (gi_reservoir.m > 0u) {
        let selected_sample = candidate_samples[gi_reservoir.selected_index];
        let selected_dir = selected_sample.direction_and_source_pdf.xyz;
        let selected_brdf = selected_sample.radiance_and_target_pdf.xyz;
        let selected_target = selected_sample.radiance_and_target_pdf.w;
        gi_reservoir_finalize(&gi_reservoir, selected_target);
        
        ray_dir = selected_dir;
        ray_source_pdf = selected_sample.direction_and_source_pdf.w;
        path_weight = selected_brdf * gi_reservoir.w;
        
        // Russian Roulette: Kill paths with very low throughput
        let weight_luminance = path_weight.x * 0.2126 + path_weight.y * 0.7152 + path_weight.z * 0.0722;
        let min_weight_threshold = 0.0001;
        let is_alive = select(0u, 1u, weight_luminance >= min_weight_threshold);

        ray_source_pdf = select(0.0, ray_source_pdf, is_alive == 1u);
    } else {
        // Reservoir failed - kill path
        is_alive = 0u;
        ray_source_pdf = 0.0;
        
        // Generate fallback direction
        let u1 = blue_noise_next(&bn_sampler);
        let u2 = blue_noise_next(&bn_sampler);
        ray_dir = sample_cosine_hemisphere(u1, u2, normal);
    }

    // =============================================================================
    // DIRECT LIGHTING with Visibility Rays (NEE)
    // Setup shadow rays for direct lighting from the cache cell position
    // =============================================================================
    let num_lights = light_count_buffer[0];
    if (num_lights > 0u) {
        let light_rand = blue_noise_next(&bn_sampler);
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
        let selected_distance = select(1e30, length(light.position.xyz - position), light.light_type != 0.0);
        world_cache_path_state[active_index].shadow_origin = vec4f(position + normal * 0.001, f32(light_idx));
        world_cache_path_state[active_index].shadow_direction = vec4f(light_dir, selected_distance * 0.999);
        world_cache_path_state[active_index].shadow_radiance = vec4f(light_contrib, 1.0);
    } else {
        world_cache_path_state[active_index].shadow_origin = vec4<f32>(0.0, 0.0, 0.0, -1.0);
        world_cache_path_state[active_index].shadow_direction = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        world_cache_path_state[active_index].shadow_radiance = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let rank = read_world_cache_cell_rank(
        position,
        normal,
        camera_position,
        u32(gi_params.world_cache_size),
        gi_params.world_cache_cell_size,
        u32(gi_params.world_cache_lod_count),
        gi_params.world_cache_cell_size * 2.0 
    );
    
    // Initialize path state
    world_cache_path_state[active_index].origin_tmin = vec4<f32>(position + normal * 0.001, 0.0001);
    world_cache_path_state[active_index].direction_tmax = vec4<f32>(ray_dir, 1e30);
    world_cache_path_state[active_index].normal_section_index = vec4<f32>(normal, 0.0);
    world_cache_path_state[active_index].state_u32 = vec4<u32>(0u, is_alive, 0u, 0xffffffffu);
    world_cache_path_state[active_index].hit_attr0 = vec4<f32>(0.0);
    world_cache_path_state[active_index].hit_attr1 = vec4<f32>(0.0);
    world_cache_path_state[active_index].rng_rank_frame_stamp = vec4<f32>(f32(bn_sampler.scramble), f32(rank), f32(frame_id), f32(bn_sampler.dimension));
    world_cache_path_state[active_index].path_weight = vec4<f32>(path_weight, ray_source_pdf);
}

