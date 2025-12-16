// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║               PER-PIXEL PATH TRACING - RAY INITIALIZATION                 ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Initializes rays for per-pixel path tracing with:                        ║
// ║  • Per-pixel dispatch with blue noise tile selection                      ║
// ║  • BRDF-importance sampled ray directions                                 ║
// ║  • ReSTIR-based path sampling for high-quality convergence                ║
// ║  • Next Event Estimation (NEE) for direct lighting                        ║
// ║                                                                           ║
// ║  Each invocation handles one pixel. The pixel determines if it should     ║
// ║  be the one traced for its tile this frame using blue noise selection.    ║
// ║  This gives upscale_factor×upscale_factor pixels per tile.                ║
// ║                                                                           ║
// ║  Uses blue noise sampling for low-discrepancy quasi-random values,        ║
// ║  providing faster convergence than traditional white noise.               ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

#include "common.wgsl"
#include "gi/gi_common.wgsl"
#include "gi/world_cache_common.wgsl"
#include "raytracing/restir_common.wgsl"

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read_write> gi_counters: GICounters;
@group(1) @binding(2) var<storage, read_write> pixel_path_state: array<PixelPathState>;
@group(1) @binding(3) var<storage, read_write> ray_work_queue: array<u32>;
@group(1) @binding(4) var<storage, read> light_count_buffer: array<u32>;
@group(1) @binding(5) var<storage, read> dense_lights_buffer: array<Light>;
@group(1) @binding(6) var<storage, read_write> world_cache: array<WorldCacheCell>;
@group(1) @binding(7) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(8) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(9) var gbuffer_albedo: texture_2d<f32>;
@group(1) @binding(10) var gbuffer_smra: texture_2d<f32>;
@group(1) @binding(11) var gbuffer_motion: texture_2d<f32>;
@group(1) @binding(12) var blue_noise: texture_2d_array<f32>;

const MAX_INITIAL_EMISSIVE = 10.0;
const MAX_NEE_LUMINANCE = 10.0;

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
// MAIN COMPUTE SHADER
// =============================================================================

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let resolution = vec2<u32>(u32(gi_params.resolution_x), u32(gi_params.resolution_y));
    let upscale_factor = u32(gi_params.upscale_factor);
    let tile_grid_dims = vec2<u32>(resolution.x / upscale_factor, resolution.y / upscale_factor);
    let total_tiles = tile_grid_dims.x * tile_grid_dims.y;
    
    let frame_id = u32(gi_params.frame_index);
    let rays_per_tile = u32(gi_params.screen_ray_count);
    let total_rays = total_tiles * rays_per_tile;
    
    // ─────────────────────────────────────────────────────────────────────────
    // Per-ray-slot dispatch: each thread initializes exactly ONE ray slot.
    // This avoids races from multiple pixels in the same tile writing to the
    // same ray slot simultaneously.
    // ─────────────────────────────────────────────────────────────────────────
    if (gid.x >= total_rays) {
        return;
    }

    let ray_slot = gid.x;
    let tile_index = ray_slot / rays_per_tile;

    // Tile coords in the tile grid
    let tile_x = tile_index % tile_grid_dims.x;
    let tile_y = tile_index / tile_grid_dims.x;

    // Initialize blue noise sampler (tile space)
    var bn_sampler = blue_noise_init(vec2<u32>(tile_x, tile_y), frame_id, ray_slot);

    // Sample blue noise to pick a pixel within the tile
    let rand_tile_x = blue_noise_next(&bn_sampler);
    let rand_tile_y = blue_noise_next(&bn_sampler);

    let pixel_coord = tile_to_pixel_with_offset(
        tile_index,
        tile_grid_dims.x,
        upscale_factor,
        resolution,
        rand_tile_x,
        rand_tile_y
    );

    process_selected_pixel(ray_slot, pixel_coord, &bn_sampler, resolution);
}

// =============================================================================
// SELECTED PIXEL PROCESSING
// =============================================================================

fn process_selected_pixel(
    ray_slot: u32,
    pixel_coord: vec2<u32>,
    bn_sampler: ptr<function, BlueNoiseSampler>,
    resolution: vec2<u32>
) {
    let frame_id = u32(gi_params.frame_index);
    let pixel_index = pixel_coord.y * resolution.x + pixel_coord.x;
    
    // ─────────────────────────────────────────────────────────────────────────
    // Sample G-buffer at selected pixel location
    // ─────────────────────────────────────────────────────────────────────────
    let normal_data = textureLoad(gbuffer_normal, pixel_coord, 0u);
    let normal = safe_normalize(normal_data.xyz);
    let normal_length = length(normal_data.xyz);
    
    // Skip sky pixels (no geometry)
    if (normal_length <= 0.0) {
        pixel_path_state[ray_slot].state_u32 = vec4<u32>(0u, 0u, 0u, 0xffffffffu);
        pixel_path_state[ray_slot].throughput_direct = vec4<f32>(0.0);
        pixel_path_state[ray_slot].throughput_indirect_diffuse = vec4<f32>(0.0);
        pixel_path_state[ray_slot].throughput_indirect_specular = vec4<f32>(0.0);
        return;
    }
    
    let position = textureLoad(gbuffer_position, pixel_coord, 0u).xyz;
    let albedo = textureLoad(gbuffer_albedo, pixel_coord, 0u).rgb;
    let smra = textureLoad(gbuffer_smra, pixel_coord, 0u);
    let motion_emissive = textureLoad(gbuffer_motion, pixel_coord, 0u);

    let roughness = smra.g;
    let metallic = smra.b;
    let reflectance = smra.r;
    let emissive = motion_emissive.w;
    let clear_coat = 0.0;
    let clear_coat_roughness = 0.0;
    
    // ─────────────────────────────────────────────────────────────────────────
    // Get view direction
    // ─────────────────────────────────────────────────────────────────────────
    let view_index = u32(frame_info.view_index);
    let v_dir = normalize(view_buffer[view_index].view_position.xyz - position);
    let n_dot_v = max(dot(v_dir, normal), 0.0001);
    
    // ═════════════════════════════════════════════════════════════════════════
    // Initial candidate sample generation using RIS 
    // ═════════════════════════════════════════════════════════════════════════
    
    let clamped_roughness = clamp(roughness, 0.001, 1.0);
    let dielectric_f0 = 0.16 * reflectance * reflectance;
    let f0 = mix(vec3<f32>(dielectric_f0), albedo, metallic);
    
    let f = f_schlick_vec3(f0, 1.0, n_dot_v);
    let fresnel_luminance = luminance(f);

    // ─────────────────────────────────────────────────────────────────────────
    // RNG Setup
    // ─────────────────────────────────────────────────────────────────────────
    var rng = u32(pixel_path_state[ray_slot].rng_sample_count_frame_stamp.x);
    if (rng == 0u) { rng = hash(pixel_index ^ u32(frame_id)); }
    else { rng = random_seed(rng); }

    // ─────────────────────────────────────────────────────────────────────────
    // Compute optimal specular vs diffuse sampling probability
    // ─────────────────────────────────────────────────────────────────────────
    // Diffuse weight: only for non-metals, scaled by (1-Fresnel) and albedo
    // Metals have no diffuse term, so (1-metallic) zeros this out
    let diffuse_weight = (1.0 - metallic) * (1.0 - fresnel_luminance) * luminance(albedo);
    
    // Probability of sampling specular lobe
    let specular_prob = clamp(fresnel_luminance / max(fresnel_luminance + diffuse_weight, 0.001), 0.001, 0.999);

        // ─────────────────────────────────────────────────────────────────────────
    // Generate BRDF sampling candidates using blue noise
    // Blue noise provides better sample distribution than white noise,
    // reducing variance and improving convergence speed
    // ─────────────────────────────────────────────────────────────────────────
    var candidate_samples: array<GISampleCandidate, num_init_ris_samples>;
    var gi_reservoir = gi_reservoir_init();

    for (var i = 0u; i < num_init_ris_samples; i = i + 1u) {
        // Sample three blue noise values for this candidate
        rng = random_seed(rng);
        let r1 = rand_float(rng);
        rng = random_seed(rng);
        let r2 = rand_float(rng);
        rng = random_seed(rng);
        let r3 = rand_float(rng);

        var dir: vec3<f32>;
        
        let is_specular_lobe = r3 < specular_prob;
        if (is_specular_lobe) {
            // ─────────────────────────────────────────────────────────────────
            // GGX Specular Sampling
            // ─────────────────────────────────────────────────────────────────
            let h = sample_ggx(normal, clamped_roughness, r1, r2);
            dir = normalize(reflect(-v_dir, h));
        } else {
            // ─────────────────────────────────────────────────────────────────
            // Uniform Hemisphere Sampling
            // ─────────────────────────────────────────────────────────────────
            dir = sample_uniform_hemisphere(normal, r1, r2);
        }

        let source_pdf = brdf_pdf(normal, v_dir, dir, roughness, specular_prob);
        // Evaluate BRDF for this direction
        let brdf = calculate_brdf_lighting_rt(
            normal, v_dir, dir, roughness, metallic,
            reflectance, clear_coat, clear_coat_roughness
        );
        let target_pdf = luminance(brdf);

        let ris_weight = target_pdf / max(source_pdf, 0.0001);
        if (ris_weight > 0.0 && !isinf(ris_weight)) {
            gi_reservoir_update(&gi_reservoir, i, ris_weight, &rng, max_spatial_samples);
        }
        
        candidate_samples[i].radiance_and_target_pdf = vec4f(brdf, target_pdf);
        candidate_samples[i].direction_and_source_pdf = vec4f(dir, source_pdf);
        candidate_samples[i].lobe_type = select(0u, 1u, is_specular_lobe);
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Finalize reservoir and select best direction
    // ─────────────────────────────────────────────────────────────────────────
    let selected_index = gi_reservoir.selected_index;
    let selected_lobe_type = candidate_samples[selected_index].lobe_type;
    let ray_dir = candidate_samples[selected_index].direction_and_source_pdf.xyz;
    let ray_source_pdf = candidate_samples[selected_index].direction_and_source_pdf.w;
    let ray_brdf = candidate_samples[selected_index].radiance_and_target_pdf.xyz;
    let selected_target_pdf = candidate_samples[selected_index].radiance_and_target_pdf.w;
    gi_reservoir_finalize(&gi_reservoir, selected_target_pdf);
    
    // Update path weight with BRDF and reservoir weight
    let path_weight = ray_brdf * gi_reservoir.w * gi_params.indirect_boost;

    // ═════════════════════════════════════════════════════════════════════════
    // NEXT EVENT ESTIMATION (NEE) - DIRECT LIGHTING
    // Setup shadow rays for direct lighting contribution
    // ═════════════════════════════════════════════════════════════════════════
    let num_lights = light_count_buffer[0];
    if (num_lights > 0u) {
        rng = random_seed(rng);
        let light_rand = rand_float(rng);
        let light_idx = u32(light_rand * f32(num_lights)) % num_lights;
        let light = dense_lights_buffer[light_idx];
        
        let light_dir = get_light_dir(light, position);
        let attenuation = get_light_attenuation(light, position);
        
        let brdf = calculate_brdf_lighting_rt(
            normal, v_dir, light_dir, roughness, metallic,
            reflectance, clear_coat, clear_coat_roughness
        );
        
        // Compute light contribution with firefly clamping
        // Clamp before storing to prevent extreme values from propagating
        let raw_light_contrib = brdf * light.color.rgb * light.intensity * attenuation * f32(num_lights);
        let light_contrib = safe_clamp_vec3_max(raw_light_contrib, MAX_NEE_LUMINANCE);
        
        // Setup shadow ray for visibility test
        let light_distance = select(1e30, length(light.position.xyz - position), light.light_type != 0.0);
        pixel_path_state[ray_slot].shadow_origin = vec4f(position + normal * 0.001, f32(light_idx));
        pixel_path_state[ray_slot].shadow_direction = vec4f(light_dir, light_distance * 0.999);
        pixel_path_state[ray_slot].shadow_radiance = vec4f(light_contrib, 1.0);
    } else {
        pixel_path_state[ray_slot].shadow_origin = vec4<f32>(0.0, 0.0, 0.0, -1.0);
        pixel_path_state[ray_slot].shadow_direction = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        pixel_path_state[ray_slot].shadow_radiance = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Store path state for initial bounce
    // ─────────────────────────────────────────────────────────────────────────
    pixel_path_state[ray_slot].origin_tmin = vec4<f32>(position + normal * 0.001, 0.0001);
    pixel_path_state[ray_slot].direction_tmax = vec4<f32>(ray_dir, 1e30);
    pixel_path_state[ray_slot].normal_section_index = vec4<f32>(normal, 0.0);
    pixel_path_state[ray_slot].state_u32 = vec4<u32>(selected_lobe_type, 1u, 0u, 0xffffffffu);
    pixel_path_state[ray_slot].hit_attr0 = vec4<f32>(0.0);
    pixel_path_state[ray_slot].hit_attr1 = vec4<f32>(0.0);
    pixel_path_state[ray_slot].rng_sample_count_frame_stamp = vec4<f32>(f32(rng), 0.0, f32(frame_id), 0.0);
    pixel_path_state[ray_slot].path_weight = vec4<f32>(path_weight, ray_source_pdf);
    // Visible emissive at the shaded (camera-visible) surface is treated as "direct".
    pixel_path_state[ray_slot].throughput_direct = vec4<f32>(safe_clamp_vec3_max(emissive * albedo, MAX_INITIAL_EMISSIVE), 0.0);
    pixel_path_state[ray_slot].throughput_indirect_diffuse = vec4<f32>(0.0);
    pixel_path_state[ray_slot].throughput_indirect_specular = vec4<f32>(0.0);
    pixel_path_state[ray_slot].pixel_coords = vec4<f32>(f32(pixel_coord.x), f32(pixel_coord.y), 0.0, 0.0);

    // ─────────────────────────────────────────────────────────────────────────
    // Add to work queue
    // ─────────────────────────────────────────────────────────────────────────
    let queue_index = atomicAdd(&gi_counters.ray_queue_count, 1u);
    ray_work_queue[queue_index] = ray_slot;
}
