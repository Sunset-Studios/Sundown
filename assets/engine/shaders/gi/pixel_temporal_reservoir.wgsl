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
@group(1) @binding(2) var<storage, read> temporal_reservoir_prev: array<GIReservoirData>;
@group(1) @binding(3) var<storage, read_write> temporal_reservoir_curr: array<GIReservoirData>;
@group(1) @binding(4) var depth_texture: texture_2d<f32>;
@group(1) @binding(5) var prev_depth_texture: texture_2d<f32>;
@group(1) @binding(6) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(7) var gbuffer_motion: texture_2d<f32>;
@group(1) @binding(8) var gbuffer_normal_prev: texture_2d<f32>;
#if SPECULAR_MASK_ENABLED
@group(1) @binding(9) var specular_mask: texture_2d<u32>;
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Temporal reprojection validation thresholds
// ─────────────────────────────────────────────────────────────────────────────
const TEMPORAL_NORMAL_THRESHOLD: f32 = 0.95;
const TEMPORAL_DEPTH_THRESHOLD: f32 = 0.05; // relative distance-to-camera threshold
const TEMPORAL_RADIANCE_RELATIVE_THRESHOLD: f32 = 0.95; // relative luminance mismatch threshold (0 = strict, 1 = permissive)
const MAX_TEMPORAL_SAMPLES = 16u;

// =============================================================================
// GEOMETRIC NORMAL RECONSTRUCTION (screen-space)
// =============================================================================
// NOTE:
// - We intentionally use a *geometric* normal derived from reconstructed world positions
//   for temporal validation, instead of the GBuffer shading normal. Shading normals
//   can change due to normal maps, LOD changes, etc., and are too strict for reprojection.
// - This is only used for *safety checks* (dot + plane-distance), not for lighting.
fn compute_geometric_normal_from_depth_tex(
    depth_tex: texture_2d<f32>,
    full_pixel_coord: vec2<i32>,
    full_res: vec2<i32>,
    view_index: u32,
    use_prev_view: bool
) -> vec3<f32> {
    let coord_l = vec2<i32>(max(full_pixel_coord.x - 1, 0), full_pixel_coord.y);
    let coord_r = vec2<i32>(min(full_pixel_coord.x + 1, full_res.x - 1), full_pixel_coord.y);
    let coord_u = vec2<i32>(full_pixel_coord.x, max(full_pixel_coord.y - 1, 0));
    let coord_d = vec2<i32>(full_pixel_coord.x, min(full_pixel_coord.y + 1, full_res.y - 1));

    let depth_l = textureLoad(depth_tex, coord_l, 0).r;
    let depth_r = textureLoad(depth_tex, coord_r, 0).r;
    let depth_u = textureLoad(depth_tex, coord_u, 0).r;
    let depth_d = textureLoad(depth_tex, coord_d, 0).r;

    let neighbors_valid = (depth_l < 1.0) && (depth_r < 1.0) && (depth_u < 1.0) && (depth_d < 1.0);
    if (!neighbors_valid) {
        return vec3<f32>(0.0);
    }

    let uv_l = coord_to_uv(coord_l, vec2<u32>(full_res));
    let uv_r = coord_to_uv(coord_r, vec2<u32>(full_res));
    let uv_u = coord_to_uv(coord_u, vec2<u32>(full_res));
    let uv_d = coord_to_uv(coord_d, vec2<u32>(full_res));

    let pos_l = select(
        reconstruct_world_position(uv_l, depth_l, view_index),
        reconstruct_prev_world_position(uv_l, depth_l, view_index),
        use_prev_view
    );
    let pos_r = select(
        reconstruct_world_position(uv_r, depth_r, view_index),
        reconstruct_prev_world_position(uv_r, depth_r, view_index),
        use_prev_view
    );
    let pos_u = select(
        reconstruct_world_position(uv_u, depth_u, view_index),
        reconstruct_prev_world_position(uv_u, depth_u, view_index),
        use_prev_view
    );
    let pos_d = select(
        reconstruct_world_position(uv_d, depth_d, view_index),
        reconstruct_prev_world_position(uv_d, depth_d, view_index),
        use_prev_view
    );

    let dp_dx = pos_r - pos_l;
    let dp_dy = pos_d - pos_u; // screen Y increases downward

    return safe_normalize(cross(dp_dx, dp_dy));
}

// =============================================================================
// MAIN COMPUTE SHADER
// =============================================================================

@compute @workgroup_size(16, 16, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let full_res = vec2<u32>(u32(gi_params.full_resolution_x), u32(gi_params.full_resolution_y));
    let gi_res = vec2<u32>(u32(gi_params.gi_resolution_x), u32(gi_params.gi_resolution_y));
    let full_res_i32 = vec2<i32>(i32(full_res.x), i32(full_res.y));

    if (gid.x >= gi_res.x || gid.y >= gi_res.y) {
        return;
    }

    let pixel_index = gid.y * gi_res.x + gid.x;
    let upscale_factor = u32(gi_params.upscale_factor);
    let full_pixel_coord = gi_pixel_to_full_res_pixel_coord(gid.xy, upscale_factor, full_res);
    let view_index = u32(frame_info.view_index);

    let full_pixel_coord_i = vec2<i32>(full_pixel_coord);
    let visible_depth = textureLoad(depth_texture, full_pixel_coord_i, 0).r;
    if (visible_depth >= 1.0) {
        temporal_reservoir_curr[pixel_index] = create_empty();
        return;
    }

    let visible_position = reconstruct_world_position(
        coord_to_uv(full_pixel_coord_i, full_res),
        visible_depth,
        view_index
    );
    let geom_normal = compute_geometric_normal_from_depth_tex(
        depth_texture,
        full_pixel_coord_i,
        full_res_i32,
        view_index,
        false
    );

#if SPECULAR_MASK_ENABLED
    if (textureLoad(specular_mask, vec2<i32>(i32(gid.x), i32(gid.y)), 0).x == 0u) {
        temporal_reservoir_curr[pixel_index] = create_empty();
        return;
    }
#endif

    let rays_per_pixel = u32(gi_params.screen_ray_count);

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
    let camera_position = view_buffer[view_index].view_position.xyz;

    let base_ray_id = pixel_index * rays_per_pixel;
    for (var i = 0u; i < rays_per_pixel; i = i + 1u) {
        let ray_id = base_ray_id + i;
        let path = pixel_path_state[ray_id];

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

        // Mark environment-miss samples explicitly. They do not have a finite sample point,
        // so later Jacobian-based spatial reuse must avoid treating them like on-surface hits.
        let is_environment_sample = path.state_u32.w == 0xffffffffu;
        let sample_kind = select(0.0, 1.0, is_environment_sample);

        candidate_samples[candidate_count].visible_position_source_pdf = vec4<f32>(visible_position, source_pdf);
        candidate_samples[candidate_count].sample_position = vec4<f32>(path.origin_tmin.xyz, sample_kind);
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
    let motion_sample = textureLoad(gbuffer_motion, vec2<i32>(full_pixel_coord), 0);
    let full_pixel_velocity = vec2<f32>(-0.5 * motion_sample.x, 0.5 * motion_sample.y) * vec2<f32>(f32(full_res.x) - 1.0, f32(full_res.y) - 1.0);
    let gi_pixel_velocity = full_pixel_velocity / max(f32(upscale_factor), 1.0);
    let pixel_center = vec2<f32>(gid.xy) + 0.5;
    let prev_coord = vec2<i32>(pixel_center + gi_pixel_velocity);
    var has_valid_reprojection = false;

    if (prev_coord.x >= 0 && prev_coord.y >= 0 && prev_coord.x < i32(gi_res.x) && prev_coord.y < i32(gi_res.y)) {
        let prev_index = u32(prev_coord.y) * gi_res.x + u32(prev_coord.x);
        let prev_reservoir_data = temporal_reservoir_prev[prev_index];
        let prev_full_pixel_coord = gi_pixel_to_full_res_pixel_coord(vec2<u32>(prev_coord), upscale_factor, full_res);
        let prev_full_pixel_coord_i = vec2<i32>(prev_full_pixel_coord);
        let prev_depth = textureLoad(prev_depth_texture, prev_full_pixel_coord_i, 0).r;
        let prev_position = reconstruct_prev_world_position(
            coord_to_uv(prev_full_pixel_coord_i, full_res),
            prev_depth,
            view_index
        );
        let prev_geom_normal = compute_geometric_normal_from_depth_tex(
            prev_depth_texture,
            prev_full_pixel_coord_i,
            full_res_i32,
            view_index,
            true
        );

        if (prev_reservoir_data.reservoir.m > 0u && prev_depth < 1.0 && length(prev_geom_normal) > 0.0 && length(geom_normal) > 0.0) {
            // Geometry validation: geometric normal + depth similarity to reject disocclusion.
            // Use abs(dot) to avoid rejecting due to sign flips from screen-space reconstruction.
            let normal_similarity = abs(dot(prev_geom_normal, geom_normal));
            let normal_valid = normal_similarity > TEMPORAL_NORMAL_THRESHOLD;

            let delta_position = prev_position - visible_position;
            // Use a relative depth metric (scaled by distance-to-camera) for robustness.
            let plane_distance = abs(dot(delta_position, geom_normal));
            let camera_distance = max(length(visible_position - camera_position), 0.001);
            let depth_valid = (plane_distance / camera_distance) < TEMPORAL_DEPTH_THRESHOLD;

            if (normal_valid && depth_valid) {
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
