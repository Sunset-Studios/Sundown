// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                 PER-PIXEL GI - SPATIAL RESERVOIR PASS                     ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Spatial resampling stage for ReSTIR GI. Reuses temporally filtered        ║
// ║  reservoirs from neighboring pixels and applies the Jacobian determinant   ║
// ║  to transform PDFs between different surface parameterizations.            ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

// =============================================================================
// DEFINES
// =============================================================================
// Uncomment to skip spatial reservoir resampling (pass through temporal sample)
// #define SKIP_SPATIAL_RESAMPLING

#include "common.wgsl"
#include "gi/gi_common.wgsl"
#include "raytracing/restir_common.wgsl"

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read> temporal_reservoir: array<GIReservoirData>;
@group(1) @binding(2) var<storage, read> spatial_reservoir_prev: array<GIReservoirData>;
@group(1) @binding(3) var<storage, read_write> spatial_reservoir_curr: array<GIReservoirData>;
@group(1) @binding(4) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(5) var gbuffer_normal: texture_2d<f32>;

// ─────────────────────────────────────────────────────────────────────────────
// Spatial Reuse Similarity Thresholds
// ─────────────────────────────────────────────────────────────────────────────
const SPATIAL_NORMAL_THRESHOLD: f32 = 0.95;  // ~18 degree threshold
const SPATIAL_DEPTH_THRESHOLD: f32 = 0.05;   // 5% relative geometric distance threshold (used for tangent-plane distance)

// =============================================================================
// HELPERS
// =============================================================================

fn create_empty(pixel_index: u32) -> GIReservoirData {
    var empty: GIReservoirData;
    empty.reservoir = gi_reservoir_init();
    empty.sample.visible_position_source_pdf = vec4<f32>(0.0);
    empty.sample.sample_position = vec4<f32>(0.0);
    empty.sample.sample_normal_target_pdf = vec4<f32>(0.0);
    empty.sample.outgoing_radiance = vec4<f32>(0.0);
    return empty;
}

// ─────────────────────────────────────────────────────────────────────────────
// Compute target PDF with Jacobian adjustment for spatial reuse
// When reusing a sample from a different surface, apply Jacobian to account
// for the change in solid angle measure between surfaces.
// ─────────────────────────────────────────────────────────────────────────────
fn compute_reuse_target_pdf(
    sample: GIReservoirSample,
    target_visible_position: vec3<f32>
) -> f32 {
    let source_visible_position = sample.visible_position_source_pdf.xyz;
    let sample_position = sample.sample_position.xyz;
    let sample_normal = safe_normalize(sample.sample_normal_target_pdf.xyz);

    let jacobian = compute_restir_gi_jacobian(
        sample_normal,
        source_visible_position,
        target_visible_position,
        sample_position
    );

    // Algorithm 4: beta_hat'_q = beta_hat_q / |J_{q->r}|
    // We use beta_hat = p_hat = luminance(f(y)), where f(y) is the *unweighted integrand*
    // stored in `sample.outgoing_radiance.xyz`.
    return sample.sample_normal_target_pdf.w / max(jacobian, 1e-6);
}

// ─────────────────────────────────────────────────────────────────────────────
// Stochastic Spatial Neighbor Sampling
// Generates a random neighbor offset using RNG based on pixel and frame index.
// Each sample index uses a unique seed for decorrelation between samples.
// ─────────────────────────────────────────────────────────────────────────────
fn get_spatial_neighbor(
    pixel_index: u32,
    sample_index: u32,
    frame_index: u32,
    max_radius: f32
) -> vec2<i32> {
    // Create unique seed combining pixel, sample, and frame for decorrelation
    let seed = hash(pixel_index ^ (sample_index * 0x9E3779B9u) ^ (frame_index * 0x85EBCA6Bu));
    
    // Generate two random values for angle and radius
    var rng = seed;
    rng = random_seed(rng);
    let rand_angle = rand_float(rng);
    rng = random_seed(rng);
    let rand_radius = rand_float(rng);
    
    // Convert to polar coordinates
    let angle = rand_angle * 2.0 * PI;
    
    // Square root for uniform disk sampling (area-preserving)
    // Add minimum radius to avoid self-sampling
    let min_radius = 1.0;
    let radius = min_radius + sqrt(rand_radius) * (max_radius - min_radius);
    
    // Convert polar to cartesian (float offset in pixels)
    let offset_f = vec2<f32>(cos(angle), sin(angle)) * radius;

    let offset_i = vec2<i32>(
        i32(round(offset_f.x)),
        i32(round(offset_f.y))
    );

    return offset_i;
}

// =============================================================================
// MAIN COMPUTE SHADER
// =============================================================================

@compute @workgroup_size(16, 16, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(gbuffer_normal);

    if (gid.x >= res.x || gid.y >= res.y) {
        return;
    }

    let pixel_index = gid.y * res.x + gid.x;
    let pixel_coord = vec2<i32>(gid.xy);
    
    let normal_sample = textureLoad(gbuffer_normal, pixel_coord, 0);
    let normal = safe_normalize(normal_sample.xyz);

    if (length(normal_sample.xyz) <= 0.0) {
        spatial_reservoir_curr[pixel_index] = create_empty(pixel_index);
        return;
    }

    // Current visible point position (used for Jacobian + similarity checks)
    let center_position = textureLoad(gbuffer_position, pixel_coord, 0).xyz;

    // ─────────────────────────────────────────────────────────────────────────
    // ReSTIR GI spatial resampling (paper Algorithm 4 / 5 style):
    // 1) Start from the temporally resampled reservoir at this pixel.
    // 2) Sample a small number of neighbor pixels in a disk.
    // 3) For each neighbor, compute Jacobian-corrected p_hat at the current pixel.
    // 4) Merge neighbor reservoirs into the output reservoir.
    // 5) Finalize reservoir weight for the selected sample.
    // ─────────────────────────────────────────────────────────────────────────
    var output: GIReservoirData;
    output.reservoir = gi_reservoir_init();
    
    // Store candidate samples for final selection
    var candidate_samples: array<GIReservoirSample, num_spatial_samples + 1>;
    var candidate_count = 0u;
    
    // =====================================================================
    // RNG Setup
    // =====================================================================
    var rng_state = hash(pixel_index ^ u32(frame_info.frame_index));

    // ─────────────────────────────────────────────────────────────────────────
    // Local temporal reservoir candidate (same surface, no Jacobian needed)
    // ─────────────────────────────────────────────────────────────────────────
    let temporal_entry = temporal_reservoir[pixel_index];
    if (temporal_entry.reservoir.m > 0u) {
        rng_state = random_seed(rng_state);

        let target_pdf = compute_reuse_target_pdf(temporal_entry.sample, center_position);
        gi_reservoir_merge(
            &output.reservoir,
            candidate_count,
            temporal_entry.reservoir,
            target_pdf,
            rand_float(rng_state),
            max_spatial_samples
        );

        candidate_samples[candidate_count] = temporal_entry.sample;
        candidate_count = candidate_count + 1u;
    }

    #ifndef SKIP_SPATIAL_RESAMPLING

    // ─────────────────────────────────────────────────────────────────────────
    // Stochastic spatial reuse from neighbors (disk sampling)
    // ─────────────────────────────────────────────────────────────────────────
    for (var i = 0u; i < num_spatial_samples; i = i + 1u) {
        // Generate a disk sample using the per-pixel RNG.
        rng_state = random_seed(rng_state);
        let rand_angle = rand_float(rng_state);
        rng_state = random_seed(rng_state);
        let rand_radius = rand_float(rng_state);

        let angle = rand_angle * 2.0 * PI;
        let radius = sqrt(rand_radius) * spatial_radius;

        let offset = vec2<i32>(
            i32(round(cos(angle) * radius)),
            i32(round(sin(angle) * radius))
        );

        // Avoid self-sampling.
        let safe_offset = select(offset, vec2<i32>(1, 0), offset.x == 0 && offset.y == 0);
        let neighbor = pixel_coord + safe_offset;

        if (neighbor.x < 0 || neighbor.y < 0 || neighbor.x >= i32(res.x) || neighbor.y >= i32(res.y)) {
            continue;
        }

        let neighbor_normal_sample = textureLoad(gbuffer_normal, neighbor, 0);
        let neighbor_normal = safe_normalize(neighbor_normal_sample.xyz);
        if (length(neighbor_normal_sample.xyz) <= 0.0) {
            continue;
        }
        
        // Normal similarity check - reject samples from surfaces with different orientations
        let normal_similarity = dot(neighbor_normal, normal);
        if (normal_similarity < SPATIAL_NORMAL_THRESHOLD) {
            continue;
        }
        
        // Tangent-plane distance check (geometric similarity):
        // Reject neighbors that are too far off the center pixel's tangent plane.
        // This is more robust than comparing depth differences (especially on
        // grazing surfaces and thin geometry), and better matches "same surface"
        // intent for spatial reuse.
        let neighbor_position = textureLoad(gbuffer_position, neighbor, 0).xyz;
        let delta_position = neighbor_position - center_position;
        let plane_distance = abs(dot(delta_position, normal));

        if (plane_distance > SPATIAL_DEPTH_THRESHOLD) {
            continue;
        }

        let neighbor_index = u32(neighbor.y) * res.x + u32(neighbor.x);
        let neighbor_entry = temporal_reservoir[neighbor_index];
        if (neighbor_entry.reservoir.m == 0u) {
            continue;
        }

        // Compute target PDF at this pixel with full Jacobian correction (paper Eq. 11)
        let target_pdf = compute_reuse_target_pdf(neighbor_entry.sample, center_position);
        
        gi_reservoir_merge(
            &output.reservoir,
            candidate_count,
            neighbor_entry.reservoir,
            target_pdf,
            rand_float(rng_state),
            max_spatial_samples
        );

        candidate_samples[candidate_count] = neighbor_entry.sample;
        candidate_count = candidate_count + 1u;
    }
    #endif

    // ─────────────────────────────────────────────────────────────────────────
    // Finalize reservoir
    // ─────────────────────────────────────────────────────────────────────────
    if (candidate_count == 0u) {
        spatial_reservoir_curr[pixel_index] = create_empty(pixel_index);
    } else {
        let selected_index = output.reservoir.selected_index;
        let selected_sample = candidate_samples[selected_index];
        
        let selected_target_pdf = compute_reuse_target_pdf(selected_sample, center_position);

        gi_reservoir_finalize(&output.reservoir, selected_target_pdf);
        
        // Keep the original sample payload (source visible pos + p_hat at source).
        // Target-pdf for this pixel can be recomputed via compute_reuse_target_pdf().
        output.sample = selected_sample;

        spatial_reservoir_curr[pixel_index] = output;
    }
}
