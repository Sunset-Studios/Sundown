// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                 PER-PIXEL GI - TEMPORAL RESERVOIR PASS                    ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Implements the temporal resampling stage of ReSTIR GI. Initial samples   ║
// ║  from the per-pixel path tracing pass are combined with a reprojected     ║
// ║  reservoir from the previous frame to build a temporally stable pool of   ║
// ║  candidates.                                                              ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

// =============================================================================
// DEFINES
// =============================================================================
// Uncomment to skip temporal reservoir resampling (pass through current sample)
// define SKIP_TEMPORAL_RESAMPLING

#include "common.wgsl"
#include "gi/gi_common.wgsl"
#include "raytracing/restir_common.wgsl"

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read> pixel_path_state: array<PixelPathState>;
@group(1) @binding(2) var<storage, read_write> temporal_reservoir_prev: array<GIReservoirData>;
@group(1) @binding(3) var<storage, read_write> temporal_reservoir_curr: array<GIReservoirData>;
@group(1) @binding(4) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(5) var gbuffer_position_prev: texture_2d<f32>;
@group(1) @binding(6) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(7) var gbuffer_motion: texture_2d<f32>;
@group(1) @binding(8) var gbuffer_normal_prev: texture_2d<f32>;

// ─────────────────────────────────────────────────────────────────────────────
// Temporal reprojection validation thresholds
// ─────────────────────────────────────────────────────────────────────────────
const TEMPORAL_NORMAL_THRESHOLD: f32 = 0.95;
const TEMPORAL_DEPTH_THRESHOLD: f32 = 0.05; // relative distance-to-camera threshold
const TEMPORAL_RADIANCE_RELATIVE_THRESHOLD: f32 = 0.95; // relative luminance mismatch threshold (0 = strict, 1 = permissive)
const MAX_TEMPORAL_SAMPLES = 10u;

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
    let normal_sample = textureLoad(gbuffer_normal, vec2<i32>(gid.xy), 0);
    let normal = safe_normalize(normal_sample.xyz);

    if (length(normal_sample.xyz) <= 0.0) {
        temporal_reservoir_curr[pixel_index] = create_empty();
        return;
    }

    let rays_per_tile = u32(gi_params.screen_ray_count);
    let upscale_factor = u32(gi_params.upscale_factor);
    let tile_x = gid.x / upscale_factor;
    let tile_y = gid.y / upscale_factor;
    let tile_grid_width = res.x / upscale_factor;
    let tile_index = tile_y * tile_grid_width + tile_x;

    // ─────────────────────────────────────────────────────────────────────────
    // Build output reservoir by merging current frame sample with history
    // ─────────────────────────────────────────────────────────────────────────
    // Store candidate samples for final selection
    var candidate_samples: array<GIReservoirSample, 2>;
    var candidate_count = 0u;
    var output: GIReservoirData;
    output.reservoir = gi_reservoir_init();

    // =====================================================================
    // RNG Setup
    // =====================================================================
    var rng_state = hash(pixel_index ^ u32(frame_info.frame_index));

    // ─────────────────────────────────────────────────────────────────────────
    // Current frame sample from the traced pixel (initial sampling buffer)
    //
    // IMPORTANT (unbiasedness):
    // - `path.throughput_*` are MC estimates which already include `1 / source_pdf`.
    // - ReSTIR/RIS expects the reservoir payload `f(y)` to be the *unweighted integrand*.
    //   Otherwise, multiplying by `reservoir.w` later will effectively apply `1 / pdf`
    //   twice and can bias/brighten the result.
    //
    // So we store:
    // - `sample.outgoing_radiance_*.xyz` = f(y) = (mc_estimate * source_pdf)
    // - `sample.sample_normal_target_pdf.w` = p_hat(y) (we use luminance(f(y)))
    // ─────────────────────────────────────────────────────────────────────────
    let visible_position = textureLoad(gbuffer_position, vec2<i32>(gid.xy), 0).xyz;
    let camera_position = view_buffer[u32(frame_info.view_index)].view_position.xyz;

    for (var i = 0u; i < rays_per_tile; i = i + 1u) {
        let ray_id = tile_index * rays_per_tile + i;
        let path = pixel_path_state[ray_id];

        // TODO: Should we skip this or fill the candidate with something meaningful?
        let traced_this_frame = u32(path.pixel_coords.x) == gid.x && u32(path.pixel_coords.y) == gid.y;
        if (!traced_this_frame) {
            continue;
        }

        let sample_count = max(path.rng_sample_count_frame_stamp.y, 1.0);
        let accumulated_avg_direct = path.throughput_direct.xyz / sample_count;
        let accumulated_avg_indirect_diffuse = path.throughput_indirect_diffuse.xyz / sample_count;
        let accumulated_avg_indirect_specular = path.throughput_indirect_specular.xyz / sample_count;

        // The proposal PDF for this candidate (used for RIS weights).
        let source_pdf = max(path.path_weight.w, 0.0001);

        // Convert MC estimate -> unweighted integrand for ReSTIR:
        // mc_estimate = f(y) / source_pdf  =>  f(y) = mc_estimate * source_pdf
        let integrand_direct = safe_clamp_vec3(accumulated_avg_direct * source_pdf);
        let integrand_indirect_diffuse = safe_clamp_vec3(accumulated_avg_indirect_diffuse * source_pdf);
        let integrand_indirect_specular = safe_clamp_vec3(accumulated_avg_indirect_specular * source_pdf);
        let integrand_total = integrand_direct + integrand_indirect_diffuse + integrand_indirect_specular;

        // Target function approximation p_hat(y). Must be computed from the same f(y).
        let target_pdf = max(luminance(integrand_total), 0.0);

        candidate_samples[candidate_count].visible_position_source_pdf = vec4<f32>(visible_position, source_pdf);
        candidate_samples[candidate_count].sample_position = vec4<f32>(path.origin_tmin.xyz, 0.0);
        candidate_samples[candidate_count].sample_normal_target_pdf = vec4<f32>(safe_normalize(path.normal_section_index.xyz), target_pdf);
        candidate_samples[candidate_count].outgoing_radiance_direct = vec4<f32>(integrand_direct, 0.0);
        candidate_samples[candidate_count].outgoing_radiance_indirect_diffuse = vec4<f32>(integrand_indirect_diffuse, 0.0);
        candidate_samples[candidate_count].outgoing_radiance_indirect_specular = vec4<f32>(integrand_indirect_specular, 0.0);
        
        // For a new sample: contribution weight = p_hat / p_source (RIS weight)
        gi_reservoir_update(
            &output.reservoir,
            candidate_count,
            target_pdf / source_pdf,
            &rng_state,
            MAX_TEMPORAL_SAMPLES
        );
        
        candidate_count = candidate_count + 1u;
    }

    #ifndef SKIP_TEMPORAL_RESAMPLING
    // ─────────────────────────────────────────────────────────────────────────
    // Reproject last frame's temporal reservoir using motion vectors
    // Use proper reservoir merging to preserve the temporal sample count
    // ─────────────────────────────────────────────────────────────────────────
    let motion_sample = textureLoad(gbuffer_motion, vec2<i32>(gid.xy), 0);
    let pixel_velocity = motion_sample.xy * vec2<f32>(f32(res.x), f32(res.y)) * vec2<f32>(0.5, -0.5);
    let pixel_center = vec2<f32>(gid.xy) + 0.5;
    let prev_coord = vec2<i32>(pixel_center - pixel_velocity);
    var has_valid_reprojection = false;

    if (prev_coord.x >= 0 && prev_coord.y >= 0 && prev_coord.x < i32(res.x) && prev_coord.y < i32(res.y)) {
        let prev_index = u32(prev_coord.y) * res.x + u32(prev_coord.x);
        let prev_reservoir_data = temporal_reservoir_prev[prev_index];
        let prev_position = textureLoad(gbuffer_position_prev, prev_coord, 0).xyz;
        let prev_normal_sample = textureLoad(gbuffer_normal_prev, prev_coord, 0);
        let prev_normal = safe_normalize(prev_normal_sample.xyz);

        if (prev_reservoir_data.reservoir.m > 0u && length(prev_normal_sample.xyz) > 0.0) {
            // Geometry validation: normal + depth similarity to reject disocclusion.
            let normal_similarity = dot(prev_normal, normal);
            let normal_valid = normal_similarity > TEMPORAL_NORMAL_THRESHOLD;

            let delta_position = prev_position - visible_position;
            let depth_valid = abs(dot(delta_position, normal)) < TEMPORAL_DEPTH_THRESHOLD;
            
            let radiance_valid = select(
                true,
                temporal_radiance_valid(prev_reservoir_data.sample, candidate_samples[0], TEMPORAL_RADIANCE_RELATIVE_THRESHOLD),
                candidate_count > 0u
            );

            if (normal_valid && depth_valid && radiance_valid) {
                rng_state = random_seed(rng_state);

                gi_reservoir_merge(
                    &output.reservoir,
                    candidate_count,
                    prev_reservoir_data.reservoir,
                    prev_reservoir_data.sample.sample_normal_target_pdf.w,
                    rand_float(rng_state),
                    MAX_TEMPORAL_SAMPLES
                );
                
                candidate_samples[candidate_count] = prev_reservoir_data.sample;
                candidate_count = candidate_count + 1u;
                has_valid_reprojection = true;
            }
        }
    }

    // Clear history when temporal reprojection fails (disocclusion)
    // This prevents stale samples from persisting indefinitely
    if (!has_valid_reprojection) {
        temporal_reservoir_prev[pixel_index] = create_empty();
    }
    #endif

    // ─────────────────────────────────────────────────────────────────────────
    // Finalize reservoir
    // ─────────────────────────────────────────────────────────────────────────
    output.sample = candidate_samples[output.reservoir.selected_index];
    gi_reservoir_finalize(&output.reservoir, output.sample.sample_normal_target_pdf.w);

    // Select the output reservoir or create an empty one if no candidates were found
    if (candidate_count > 0u) {
        temporal_reservoir_curr[pixel_index] = output;
    } else {
        temporal_reservoir_curr[pixel_index] = create_empty();
    }
}
