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
@group(1) @binding(4) var<storage, read> dense_lights_buffer: DenseLightsBuffer;
@group(1) @binding(5) var<storage, read> emissive_lights_buffer: EmissiveLightsBuffer;
@group(1) @binding(6) var<storage, read_write> world_cache: array<WorldCacheCell>;
@group(1) @binding(7) var depth_texture: texture_2d<f32>;
@group(1) @binding(8) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(9) var gbuffer_albedo: texture_2d<f32>;
@group(1) @binding(10) var gbuffer_smra: texture_2d<f32>;
@group(1) @binding(11) var gbuffer_motion: texture_2d<f32>;
@group(1) @binding(12) var blue_noise: texture_2d_array<f32>;
#if SPECULAR_MASK_ENABLED
@group(1) @binding(13) var specular_mask: texture_2d<u32>;
#endif

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
// ║  • 64 layers of blue noise textures for temporal decorrelation            ║
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

fn sample_weighted_emissive_light(
    rng: ptr<function, u32>,
    num_emissive_lights: u32,
    emissive_pdf: ptr<function, f32>
) -> u32 {
    let safe_emissive_count = max(num_emissive_lights, 1u);
    let uniform_pdf = 1.0 / f32(safe_emissive_count);
    (*emissive_pdf) = uniform_pdf;

    (*rng) = random_seed((*rng));
    let uniform_rand = rand_float((*rng));
    var selected_emissive_idx = u32(uniform_rand * f32(safe_emissive_count)) % safe_emissive_count;

    if (num_emissive_lights == 0u) {
        return selected_emissive_idx;
    }

    let total_sampling_weight =
        f32(emissive_lights_buffer.header._pad0) * EMISSIVE_WEIGHT_QUANTIZATION_INV;
    let max_sampling_weight =
        f32(emissive_lights_buffer.header._pad1) * EMISSIVE_WEIGHT_QUANTIZATION_INV;

    let can_use_weighted_sampling =
        total_sampling_weight > 0.0 && max_sampling_weight > 0.0;

    var accepted = false;
    if (can_use_weighted_sampling) {
        for (var attempt_idx = 0u; attempt_idx < EMISSIVE_WEIGHTED_SAMPLE_ATTEMPTS; attempt_idx = attempt_idx + 1u) {
            (*rng) = random_seed((*rng));
            let candidate_rand = rand_float((*rng));
            let candidate_idx = u32(candidate_rand * f32(num_emissive_lights)) % num_emissive_lights;
            let candidate_weight = max(emissive_lights_buffer.lights[candidate_idx].radiance_weight.w, 0.0);
            let accept_prob = min(candidate_weight / max_sampling_weight, 1.0);

            (*rng) = random_seed((*rng));
            let accept_rand = rand_float((*rng));
            if (accept_rand <= accept_prob) {
                selected_emissive_idx = candidate_idx;
                accepted = true;
                break;
            }
        }
    }

    if (accepted) {
        let selected_weight = max(emissive_lights_buffer.lights[selected_emissive_idx].radiance_weight.w, 0.0);
        (*emissive_pdf) = selected_weight / max(total_sampling_weight, 1e-6);
    }

    return selected_emissive_idx;
}

// =============================================================================
// MAIN COMPUTE SHADER
// =============================================================================

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let full_resolution = vec2<u32>(u32(gi_params.full_resolution_x), u32(gi_params.full_resolution_y));
    let gi_resolution = vec2<u32>(u32(gi_params.gi_resolution_x), u32(gi_params.gi_resolution_y));
    
    let frame_id = u32(gi_params.frame_index);
    let rays_per_pixel = u32(gi_params.screen_ray_count);
    let total_pixels = gi_resolution.x * gi_resolution.y;
    let total_rays = total_pixels * rays_per_pixel;
    
    // ─────────────────────────────────────────────────────────────────────────
    // Per-ray-slot dispatch: each thread initializes exactly ONE ray slot.
    // This avoids races from multiple pixels in the same tile writing to the
    // same ray slot simultaneously.
    // ─────────────────────────────────────────────────────────────────────────
    if (gid.x >= total_rays) {
        return;
    }

    let ray_slot = gid.x;
    let pixel_index = ray_slot / rays_per_pixel;

    // GI pixel coords
    let gi_x = pixel_index % gi_resolution.x;
    let gi_y = pixel_index / gi_resolution.x;
    let gi_pixel_coord = vec2<u32>(gi_x, gi_y);

    // Representative full-res pixel coord for GBuffer sampling
    let upscale_factor = u32(gi_params.upscale_factor);
    let full_pixel_coord = gi_pixel_to_full_res_pixel_coord(gi_pixel_coord, upscale_factor, full_resolution);

    // Initialize blue noise sampler (GI pixel space)
    var bn_sampler = blue_noise_init(gi_pixel_coord, frame_id, ray_slot);

    process_selected_pixel(ray_slot, gi_pixel_coord, full_pixel_coord, &bn_sampler, gi_resolution, full_resolution);
}

// =============================================================================
// SELECTED PIXEL PROCESSING
// =============================================================================

fn process_selected_pixel(
    ray_slot: u32,
    gi_pixel_coord: vec2<u32>,
    full_pixel_coord: vec2<u32>,
    bn_sampler: ptr<function, BlueNoiseSampler>,
    resolution: vec2<u32>,
    full_resolution: vec2<u32>
) {
    let frame_id = u32(gi_params.frame_index);
    let pixel_index = gi_pixel_coord.y * resolution.x + gi_pixel_coord.x;
    
    // ─────────────────────────────────────────────────────────────────────────
    // Sample G-buffer at selected pixel location
    // ─────────────────────────────────────────────────────────────────────────
    let normal_data = textureLoad(gbuffer_normal, full_pixel_coord, 0u);
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

#if SPECULAR_MASK_ENABLED
    // Skip pixels that are not deemed specular-relevant for this pipeline.
    if (textureLoad(specular_mask, vec2<i32>(gi_pixel_coord), 0).x == 0u) {
        pixel_path_state[ray_slot].state_u32 = vec4<u32>(0u, 0u, 0u, 0xffffffffu);
        pixel_path_state[ray_slot].throughput_direct = vec4<f32>(0.0);
        pixel_path_state[ray_slot].throughput_indirect_diffuse = vec4<f32>(0.0);
        pixel_path_state[ray_slot].throughput_indirect_specular = vec4<f32>(0.0);
        return;
    }
#endif
    
    let depth = textureLoad(depth_texture, full_pixel_coord, 0u).r;
    let position = reconstruct_world_position(
        coord_to_uv(vec2<i32>(full_pixel_coord), full_resolution),
        depth,
        u32(frame_info.view_index)
    );
    let albedo = textureLoad(gbuffer_albedo, full_pixel_coord, 0u).rgb;
    let smra = textureLoad(gbuffer_smra, full_pixel_coord, 0u);
    let motion_emissive = textureLoad(gbuffer_motion, full_pixel_coord, 0u);

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
    //let fresnel_luminance = (f.x + f.y + f.z) / 3.0;

    // Probability of sampling specular vs diffuse
    let use_ggx = (roughness < 0.3) || (metallic > 0.5);
    let specular_prob_if_ggx = clamp(fresnel_luminance, 0.001, 0.99);
    let specular_prob = select(0.0, specular_prob_if_ggx, use_ggx);

    // ─────────────────────────────────────────────────────────────────────────
    // RNG Setup
    // ─────────────────────────────────────────────────────────────────────────
    var rng = u32(pixel_path_state[ray_slot].rng_sample_count_frame_stamp.x);
    if (rng == 0u) { rng = hash(pixel_index ^ u32(frame_id)); }
    else { rng = random_seed(rng); }

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
            gi_reservoir_update(&gi_reservoir, i, ris_weight, &rng, num_init_ris_samples);
        }
        
        candidate_samples[i].radiance_and_target_pdf = vec4<f32>(brdf, target_pdf);
        candidate_samples[i].direction_and_source_pdf = vec4<f32>(dir, source_pdf);
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

    // ─────────────────────────────────────────────────────────────────────────
    // Store path state for initial bounce
    // ─────────────────────────────────────────────────────────────────────────
    pixel_path_state[ray_slot].origin_tmin = vec4<f32>(position + normal * 0.001, 0.0001);
    pixel_path_state[ray_slot].direction_tmax = vec4<f32>(ray_dir, gi_params.max_ray_length);
    pixel_path_state[ray_slot].normal_section_index = vec4<f32>(normal, 0.0);
    pixel_path_state[ray_slot].state_u32 = vec4<u32>(selected_lobe_type, 1u, 0u, 0xffffffffu);
    pixel_path_state[ray_slot].hit_attr0 = vec4<f32>(0.0);
    pixel_path_state[ray_slot].hit_attr1 = vec4<f32>(0.0);
    pixel_path_state[ray_slot].rng_sample_count_frame_stamp = vec4<f32>(f32(rng), 0.0, f32(frame_id), 0.0);
    pixel_path_state[ray_slot].path_weight = vec4<f32>(path_weight, ray_source_pdf);
    // Visible emissive at the shaded (camera-visible) surface is treated as "direct".
    pixel_path_state[ray_slot].throughput_direct = vec4<f32>(safe_clamp_vec3_max(emissive * albedo, MAX_NEE_LUMINANCE), 0.0);
    pixel_path_state[ray_slot].throughput_indirect_diffuse = vec4<f32>(0.0);
    pixel_path_state[ray_slot].throughput_indirect_specular = vec4<f32>(0.0);
    pixel_path_state[ray_slot].pixel_coords = vec4<f32>(f32(gi_pixel_coord.x), f32(gi_pixel_coord.y), 0.0, 0.0);

    // ─────────────────────────────────────────────────────────────────────────
    // Add to work queue
    // ─────────────────────────────────────────────────────────────────────────
    let queue_index = atomicAdd(&gi_counters.ray_queue_count, 1u);
    ray_work_queue[queue_index] = ray_slot;
}
