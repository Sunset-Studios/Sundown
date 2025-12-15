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
@group(1) @binding(1) var<storage, read> temporal_reservoir_curr: array<GIReservoirData>;
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
// MAIN COMPUTE SHADER
// =============================================================================

@compute @workgroup_size(16, 16, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(gbuffer_position);

    if (gid.x >= res.x || gid.y >= res.y) {
        return;
    }

    let pixel_index = gid.y * res.x + gid.x;
    let pixel_coord = vec2<u32>(gid.xy);
    
    let normal_sample = textureLoad(gbuffer_normal, pixel_coord, 0u);
    let normal = safe_normalize(normal_sample.xyz);

    if (length(normal_sample.xyz) <= 0.0) {
        spatial_reservoir_curr[pixel_index] = create_empty();
        return;
    }

    // Current visible point position (used for Jacobian + similarity checks)
    let center_position = textureLoad(gbuffer_position, pixel_coord, 0u).xyz;

    // ─────────────────────────────────────────────────────────────────────────
    // ReSTIR GI spatial resampling:
    // 1) Start from the temporally resampled reservoir at this pixel.
    // 2) Sample a small number of neighbor pixels in a disk.
    // 3) For each neighbor, compute Jacobian-corrected p_hat at the current pixel.
    // 4) Merge neighbor reservoirs into the output reservoir.
    // 5) Finalize reservoir weight for the selected sample.
    // ─────────────────────────────────────────────────────────────────────────
    // Store candidate samples for final selection
    var candidate_samples: array<GIReservoirSample, num_spatial_samples + 1>;
    var candidate_count = 0u;
    var output: GIReservoirData;
    output.reservoir = gi_reservoir_init();
    
    // =====================================================================
    // RNG Setup
    // =====================================================================
    var rng_state = hash(pixel_index ^ u32(frame_info.frame_index));

    // ─────────────────────────────────────────────────────────────────────────
    // Local temporal reservoir candidate (same surface, no Jacobian needed)
    // ─────────────────────────────────────────────────────────────────────────
    if (temporal_reservoir_curr[pixel_index].reservoir.m > 0u) {
        rng_state = random_seed(rng_state);

        let target_pdf = compute_reuse_target_pdf(temporal_reservoir_curr[pixel_index].sample, center_position);
        gi_reservoir_merge(
            &output.reservoir,
            candidate_count,
            temporal_reservoir_curr[pixel_index].reservoir,
            target_pdf,
            rand_float(rng_state),
            max_spatial_samples
        );

        candidate_samples[candidate_count] = temporal_reservoir_curr[pixel_index].sample;
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
        let angle = rand_angle * 2.0 * PI;

        // Sample neighbor positions by offsetting in WORLD SPACE (tangent-plane disk)
        // and projecting back to screen pixels.
        let tangent_basis = orthonormalize(normal);
        let disk_offset_xy = vec2<f32>(cos(angle), sin(angle)) * spatial_radius;

        let world_offset = tangent_basis[0] * disk_offset_xy.x + tangent_basis[1] * disk_offset_xy.y;
        let safe_world_offset = select(world_offset, tangent_basis[0] * 1e-4, dot(world_offset, world_offset) < 1e-10);

        let neighbor_world_position = center_position + safe_world_offset;

        let view_index = u32(frame_info.view_index);
        let view_proj = view_buffer[view_index].view_projection_matrix;
        let neighbor_clip = view_proj * vec4<f32>(neighbor_world_position, 1.0);

        // Reject points behind the camera / invalid projection.
        if (neighbor_clip.w <= 1e-6) {
            continue;
        }

        let neighbor_ndc = neighbor_clip.xy / neighbor_clip.w;
        var neighbor_uv = neighbor_ndc * 0.5 + vec2<f32>(0.5);
        neighbor_uv.y = 1.0 - neighbor_uv.y;

        let uv_in_bounds = neighbor_uv.x >= 0.0 && neighbor_uv.x < 1.0 && neighbor_uv.y >= 0.0 && neighbor_uv.y < 1.0;
        if (!uv_in_bounds) {
            continue;
        }

        let neighbor_coord_f = neighbor_uv * vec2<f32>(f32(res.x), f32(res.y));
        let neighbor = vec2<u32>(u32(neighbor_coord_f.x), u32(neighbor_coord_f.y));

        let neighbor_normal_sample = textureLoad(gbuffer_normal, neighbor, 0u);
        let neighbor_normal = safe_normalize(neighbor_normal_sample.xyz);
        if (length(neighbor_normal_sample.xyz) <= 0.0) {
            continue;
        }
        
        let neighbor_index = neighbor.y * res.x + neighbor.x;

        // Normal similarity check - reject samples from surfaces with different orientations
        let normal_similarity = dot(neighbor_normal, normal);
        // Tangent-plane distance check (geometric similarity):
        let neighbor_position = textureLoad(gbuffer_position, neighbor, 0u).xyz;
        let plane_distance = abs(dot(neighbor_position - center_position, normal));

        let is_self = neighbor.x == pixel_coord.x && neighbor.y == pixel_coord.y;

        let valid_for_reuse = normal_similarity > SPATIAL_NORMAL_THRESHOLD
            && plane_distance < SPATIAL_DEPTH_THRESHOLD
            && temporal_reservoir_curr[neighbor_index].reservoir.m > 0u
            && !is_self;

        if (valid_for_reuse) {
            // Compute target PDF at this pixel with full Jacobian correction (paper Eq. 11)
            let target_pdf = compute_reuse_target_pdf(temporal_reservoir_curr[neighbor_index].sample, center_position);
            gi_reservoir_merge(
                &output.reservoir,
                candidate_count,
                temporal_reservoir_curr[neighbor_index].reservoir,
                target_pdf,
                rand_float(rng_state),
                max_spatial_samples
            );

            candidate_samples[candidate_count] = temporal_reservoir_curr[neighbor_index].sample;
            candidate_count = candidate_count + 1u;
        }
    }
    #endif

    // ─────────────────────────────────────────────────────────────────────────
    // Finalize reservoir
    // ─────────────────────────────────────────────────────────────────────────
    output.sample = candidate_samples[output.reservoir.selected_index];
    
    gi_reservoir_finalize(
        &output.reservoir,
        compute_reuse_target_pdf(output.sample, center_position)
    );

    if (candidate_count == 0u) {
        spatial_reservoir_curr[pixel_index] = create_empty();
    } else {
        spatial_reservoir_curr[pixel_index] = output;
    }
}
