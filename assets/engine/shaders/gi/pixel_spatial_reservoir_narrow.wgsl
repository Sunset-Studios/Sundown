// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║           PER-PIXEL GI - SPATIAL RESERVOIR PASS (NARROW REUSE)            ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Second spatial resampling stage for ReSTIR GI. Uses a *narrow* screen-    ║
// ║  space disk with fewer samples to locally refine the wide-pass output.    ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

// =============================================================================
// DEFINES
// =============================================================================
// Uncomment to skip spatial reservoir resampling (pass through input reservoir)
// define SKIP_SPATIAL_RESAMPLING

#include "common.wgsl"
#include "gi/gi_common.wgsl"
#include "raytracing/restir_common.wgsl"

// =============================================================================
// TUNABLES (NARROW PASS)
// =============================================================================
const SPATIAL_RADIUS_PIXELS: f32 = 12.0;
const SPATIAL_SAMPLE_COUNT: u32 = 3u;
const MAX_SPATIAL_SAMPLES = 500u;
const SPATIAL_SKIP_ROUGHNESS_THRESHOLD: f32 = 0.15;

// ─────────────────────────────────────────────────────────────────────────────
// Spatial Reuse Similarity Thresholds
// ─────────────────────────────────────────────────────────────────────────────
const SPATIAL_NORMAL_THRESHOLD: f32 = 0.97;
const SPATIAL_DEPTH_THRESHOLD: f32 = 0.03;

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read> input_reservoir: array<GIReservoirData>;
@group(1) @binding(2) var<storage, read_write> spatial_reservoir_curr: array<GIReservoirData>;
@group(1) @binding(3) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(4) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(5) var gbuffer_smra: texture_2d<f32>;
#if SPECULAR_MASK_ENABLED
@group(1) @binding(6) var specular_mask: texture_2d<u32>;
#endif

// =============================================================================
// MAIN COMPUTE SHADER
// =============================================================================

@compute @workgroup_size(16, 16, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let full_res = vec2<u32>(u32(gi_params.full_resolution_x), u32(gi_params.full_resolution_y));
    let res = vec2<u32>(u32(gi_params.gi_resolution_x), u32(gi_params.gi_resolution_y));

    if (gid.x >= res.x || gid.y >= res.y) {
        return;
    }

    let pixel_index = gid.y * res.x + gid.x;
    let pixel_coord = vec2<u32>(gid.xy);

    let upscale_factor = u32(gi_params.upscale_factor);
    let full_pixel_coord = gi_pixel_to_full_res_pixel_coord(pixel_coord, upscale_factor, full_res);

    let normal_sample = textureLoad(gbuffer_normal, full_pixel_coord, 0u);
    let normal = safe_normalize(normal_sample.xyz);

    if (length(normal_sample.xyz) <= 0.0) {
        spatial_reservoir_curr[pixel_index] = create_empty();
        return;
    }

#if SPECULAR_MASK_ENABLED
    if (textureLoad(specular_mask, vec2<i32>(i32(gid.x), i32(gid.y)), 0).x == 0u) {
        spatial_reservoir_curr[pixel_index] = create_empty();
        return;
    }
#endif

    let roughness = clamp(textureLoad(gbuffer_smra, full_pixel_coord, 0u).g, 0.0, 1.0);
    if (roughness < SPATIAL_SKIP_ROUGHNESS_THRESHOLD) {
        spatial_reservoir_curr[pixel_index] = input_reservoir[pixel_index];
        return;
    }

    let center_position = textureLoad(gbuffer_position, full_pixel_coord, 0u).xyz;
    let camera_position = view_buffer[u32(frame_info.view_index)].view_position.xyz;

    var candidate_samples: array<GIReservoirSample, SPATIAL_SAMPLE_COUNT + 1u>;
    var candidate_count = 0u;

    var output: GIReservoirData;
    output.reservoir = gi_reservoir_init();

    var rng_state = hash(pixel_index ^ u32(frame_info.frame_index));

    // Local candidate from the input reservoir.
    if (input_reservoir[pixel_index].reservoir.m > 0u) {
        rng_state = random_seed(rng_state);
        
        // Environment samples do not have a finite sample point; avoid Jacobian-based reuse math.
        let is_environment_sample = input_reservoir[pixel_index].sample.sample_position.w > 0.5;
        let target_pdf = select(
            compute_reuse_target_pdf(input_reservoir[pixel_index].sample, center_position),
            input_reservoir[pixel_index].sample.sample_normal_target_pdf.w,
            is_environment_sample
        );
        gi_reservoir_merge(
            &output.reservoir,
            candidate_count,
            input_reservoir[pixel_index].reservoir,
            target_pdf,
            rand_float(rng_state),
            MAX_SPATIAL_SAMPLES
        );

        candidate_samples[candidate_count] = input_reservoir[pixel_index].sample;
        candidate_count = candidate_count + 1u;
    }

    #ifndef SKIP_SPATIAL_RESAMPLING
    // Narrow stochastic screen-space reuse (uniform disk in pixels)
    for (var i = 0u; i < SPATIAL_SAMPLE_COUNT; i = i + 1u) {
        rng_state = random_seed(rng_state);
        let rand_radius = rand_float(rng_state);
        rng_state = random_seed(rng_state);
        let rand_angle = rand_float(rng_state);

        // Keep radius roughly constant in full-res pixels by scaling in GI pixel units.
        let scaled_radius_pixels = SPATIAL_RADIUS_PIXELS / max(f32(upscale_factor), 1.0);
        let radius_pixels = rand_radius * scaled_radius_pixels;
        let angle = rand_angle * 2.0 * PI;
        let offset_pixels_f = vec2<f32>(cos(angle), sin(angle)) * radius_pixels;

        let neighbor_coord_i = vec2<i32>(gid.xy) + vec2<i32>(round(offset_pixels_f));
        let max_coord_i = vec2<i32>(i32(res.x) - 1, i32(res.y) - 1);
        let neighbor = vec2<u32>(clamp(neighbor_coord_i, vec2<i32>(0, 0), max_coord_i));

        let is_self = neighbor.x == pixel_coord.x && neighbor.y == pixel_coord.y;
        if (is_self) {
            continue;
        }

        let neighbor_full_pixel_coord = gi_pixel_to_full_res_pixel_coord(neighbor, upscale_factor, full_res);
        let neighbor_normal_sample = textureLoad(gbuffer_normal, neighbor_full_pixel_coord, 0u);
        let neighbor_normal = safe_normalize(neighbor_normal_sample.xyz);
        if (length(neighbor_normal_sample.xyz) <= 0.0) {
            continue;
        }

        let neighbor_index = neighbor.y * res.x + neighbor.x;
        if (input_reservoir[neighbor_index].reservoir.m == 0u) {
            continue;
        }

        // Do not spatially reuse environment-miss samples across pixels.
        if (input_reservoir[neighbor_index].sample.sample_position.w > 0.5) {
            continue;
        }

        let normal_similarity = dot(neighbor_normal, normal);
        let neighbor_position = textureLoad(gbuffer_position, neighbor_full_pixel_coord, 0u).xyz;
        // Use a relative depth metric (scaled by distance-to-camera) for robustness.
        let plane_distance = abs(dot(neighbor_position - center_position, normal));
        let camera_distance = max(length(center_position - camera_position), 0.001);

        let valid_for_reuse = normal_similarity > SPATIAL_NORMAL_THRESHOLD
            && (plane_distance / camera_distance) < SPATIAL_DEPTH_THRESHOLD;

        if (valid_for_reuse) {
            rng_state = random_seed(rng_state);

            let target_pdf = compute_reuse_target_pdf(input_reservoir[neighbor_index].sample, center_position);
            gi_reservoir_merge(
                &output.reservoir,
                candidate_count,
                input_reservoir[neighbor_index].reservoir,
                target_pdf,
                rand_float(rng_state),
                MAX_SPATIAL_SAMPLES
            );

            candidate_samples[candidate_count] = input_reservoir[neighbor_index].sample;
            candidate_count = candidate_count + 1u;
        }
    }
    #endif

    output.sample = candidate_samples[output.reservoir.selected_index];

    gi_reservoir_finalize(
        &output.reservoir,
        select(
            compute_reuse_target_pdf(output.sample, center_position),
            output.sample.sample_normal_target_pdf.w,
            output.sample.sample_position.w > 0.5
        )
    );

    if (candidate_count > 0u) {
        spatial_reservoir_curr[pixel_index] = output;
    } else {
        spatial_reservoir_curr[pixel_index] = create_empty();
    }
}


