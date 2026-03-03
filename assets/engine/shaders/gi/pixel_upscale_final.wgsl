// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                 PTGI - FINAL FULL-RES UPSCALE PASS                         ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Upscales low-resolution GI outputs into full-resolution textures.         ║
// ║  Uses a lightweight edge-aware (normal+depth) weighting to preserve        ║
// ║  geometric discontinuities (crisp contact edges) while remaining stable.   ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

#include "common.wgsl"
#include "gi/gi_common.wgsl"

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var gi_low_direct: texture_2d<f32>;
@group(1) @binding(2) var gi_low_indirect_diffuse: texture_2d<f32>;
@group(1) @binding(3) var gi_low_indirect_specular: texture_2d<f32>;
@group(1) @binding(4) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(5) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(6) var gbuffer_smra: texture_2d<f32>;
@group(1) @binding(7) var out_direct: texture_storage_2d<rgba16float, write>;
@group(1) @binding(8) var out_indirect_diffuse: texture_storage_2d<rgba16float, write>;
@group(1) @binding(9) var out_indirect_specular: texture_storage_2d<rgba16float, write>;
#if SPECULAR_MASK_ENABLED
@group(1) @binding(10) var specular_mask: texture_2d<u32>;
#endif

// =============================================================================
// CONSTANTS
// =============================================================================

const DEPTH_SIGMA: f32 = 0.05;
const NORMAL_POWER: f32 = 64.0;
const SPECULAR_EIGHT_TAP_ROUGHNESS_THRESHOLD: f32 = 0.15;
const EIGHT_TAP_DISTANCE_FALLOFF: f32 = 0.75;

// =============================================================================
// HELPERS
// =============================================================================

fn compute_edge_weight(
    center_depth: f32,
    center_normal: vec3<f32>,
    sample_depth: f32,
    sample_normal: vec3<f32>
) -> f32 {
    let depth_diff = abs(center_depth - sample_depth) / max(center_depth, 0.001);
    let depth_weight = exp(-depth_diff * depth_diff / (2.0 * DEPTH_SIGMA * DEPTH_SIGMA));

    let normal_dot = max(dot(center_normal, sample_normal), 0.0);
    let normal_weight = pow(normal_dot, NORMAL_POWER);

    return depth_weight * normal_weight;
}

fn clamp_i32(v: vec2<i32>, lo: vec2<i32>, hi: vec2<i32>) -> vec2<i32> {
    return vec2<i32>(clamp(v.x, lo.x, hi.x), clamp(v.y, lo.y, hi.y));
}

// =============================================================================
// MAIN
// =============================================================================

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let full_res = vec2<u32>(u32(gi_params.full_resolution_x), u32(gi_params.full_resolution_y));
    let gi_res = vec2<u32>(u32(gi_params.gi_resolution_x), u32(gi_params.gi_resolution_y));
    let upscale_factor = u32(gi_params.upscale_factor);

    if (gid.x >= full_res.x || gid.y >= full_res.y) {
        return;
    }

    let full_pixel_coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let center_normal_data = textureLoad(gbuffer_normal, full_pixel_coord, 0);
    let center_normal = safe_normalize(center_normal_data.xyz);

    // Sky pixel: no geometry => no GI.
    if (length(center_normal_data.xyz) <= 0.0) {
        textureStore(out_direct, full_pixel_coord, vec4<f32>(0.0));
        textureStore(out_indirect_diffuse, full_pixel_coord, vec4<f32>(0.0));
        textureStore(out_indirect_specular, full_pixel_coord, vec4<f32>(0.0));
        return;
    }

#if SPECULAR_MASK_ENABLED
    // Mask is in GI resolution. Use the corresponding GI texel for this full-res pixel.
    let gi_coord = vec2<u32>(
        min(u32(gid.x) / max(u32(upscale_factor), 1u), gi_res.x - 1u),
        min(u32(gid.y) / max(u32(upscale_factor), 1u), gi_res.y - 1u)
    );
    if (textureLoad(specular_mask, vec2<i32>(gi_coord), 0).x == 0u) {
        textureStore(out_direct, full_pixel_coord, vec4<f32>(0.0));
        textureStore(out_indirect_diffuse, full_pixel_coord, vec4<f32>(0.0));
        textureStore(out_indirect_specular, full_pixel_coord, vec4<f32>(0.0));
        return;
    }
#endif

    let view = view_buffer[u32(frame_info.view_index)];
    let camera_position = view.view_position.xyz;
    let center_position = textureLoad(gbuffer_position, full_pixel_coord, 0).xyz;
    let center_depth = length(center_position - camera_position);
    let roughness = clamp(textureLoad(gbuffer_smra, full_pixel_coord, 0).g, 0.0, 1.0);
    let should_use_eight_tap = roughness < SPECULAR_EIGHT_TAP_ROUGHNESS_THRESHOLD;

    // Map full-res pixel to low-res GI space (floating point for bilinear taps).
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) / vec2<f32>(f32(full_res.x), f32(full_res.y));
    let gi_pos = uv * vec2<f32>(f32(gi_res.x), f32(gi_res.y)) - vec2<f32>(0.5);

    let base_i = vec2<i32>(floor(gi_pos));
    let frac = gi_pos - vec2<f32>(base_i);

    let max_gi_i = vec2<i32>(i32(gi_res.x) - 1, i32(gi_res.y) - 1);
    let c00_i = clamp_i32(base_i, vec2<i32>(0, 0), max_gi_i);
    let c10_i = clamp_i32(base_i + vec2<i32>(1, 0), vec2<i32>(0, 0), max_gi_i);
    let c01_i = clamp_i32(base_i + vec2<i32>(0, 1), vec2<i32>(0, 0), max_gi_i);
    let c11_i = clamp_i32(base_i + vec2<i32>(1, 1), vec2<i32>(0, 0), max_gi_i);

    let w00 = (1.0 - frac.x) * (1.0 - frac.y);
    let w10 = frac.x * (1.0 - frac.y);
    let w01 = (1.0 - frac.x) * frac.y;
    let w11 = frac.x * frac.y;

    // Compute edge weights by comparing the center full-res geometry to the
    // representative full-res geometry of each GI texel tap.
    let c00_full = gi_pixel_to_full_res_pixel_coord(vec2<u32>(c00_i), upscale_factor, full_res);
    let c10_full = gi_pixel_to_full_res_pixel_coord(vec2<u32>(c10_i), upscale_factor, full_res);
    let c01_full = gi_pixel_to_full_res_pixel_coord(vec2<u32>(c01_i), upscale_factor, full_res);
    let c11_full = gi_pixel_to_full_res_pixel_coord(vec2<u32>(c11_i), upscale_factor, full_res);

    let c00_pos = textureLoad(gbuffer_position, vec2<i32>(c00_full), 0).xyz;
    let c10_pos = textureLoad(gbuffer_position, vec2<i32>(c10_full), 0).xyz;
    let c01_pos = textureLoad(gbuffer_position, vec2<i32>(c01_full), 0).xyz;
    let c11_pos = textureLoad(gbuffer_position, vec2<i32>(c11_full), 0).xyz;

    let c00_n = safe_normalize(textureLoad(gbuffer_normal, vec2<i32>(c00_full), 0).xyz);
    let c10_n = safe_normalize(textureLoad(gbuffer_normal, vec2<i32>(c10_full), 0).xyz);
    let c01_n = safe_normalize(textureLoad(gbuffer_normal, vec2<i32>(c01_full), 0).xyz);
    let c11_n = safe_normalize(textureLoad(gbuffer_normal, vec2<i32>(c11_full), 0).xyz);

    let e00 = compute_edge_weight(center_depth, center_normal, length(c00_pos - camera_position), c00_n);
    let e10 = compute_edge_weight(center_depth, center_normal, length(c10_pos - camera_position), c10_n);
    let e01 = compute_edge_weight(center_depth, center_normal, length(c01_pos - camera_position), c01_n);
    let e11 = compute_edge_weight(center_depth, center_normal, length(c11_pos - camera_position), c11_n);

    let ww00 = w00 * e00;
    let ww10 = w10 * e10;
    let ww01 = w01 * e01;
    let ww11 = w11 * e11;

    let weight_sum = ww00 + ww10 + ww01 + ww11;
    let inv_weight_sum = 1.0 / max(weight_sum, 1e-6);

    // Fallback to pure bilinear weights if edge weights collapse (e.g. bad data).
    let use_edge_weights = weight_sum > 1e-6;
    let fw00 = select(w00, ww00 * inv_weight_sum, use_edge_weights);
    let fw10 = select(w10, ww10 * inv_weight_sum, use_edge_weights);
    let fw01 = select(w01, ww01 * inv_weight_sum, use_edge_weights);
    let fw11 = select(w11, ww11 * inv_weight_sum, use_edge_weights);

    // Sample low-res GI (radiance in rgb, sample_count in a).
    let s00_d = textureLoad(gi_low_direct, c00_i, 0);
    let s10_d = textureLoad(gi_low_direct, c10_i, 0);
    let s01_d = textureLoad(gi_low_direct, c01_i, 0);
    let s11_d = textureLoad(gi_low_direct, c11_i, 0);

    let s00_id = textureLoad(gi_low_indirect_diffuse, c00_i, 0);
    let s10_id = textureLoad(gi_low_indirect_diffuse, c10_i, 0);
    let s01_id = textureLoad(gi_low_indirect_diffuse, c01_i, 0);
    let s11_id = textureLoad(gi_low_indirect_diffuse, c11_i, 0);

    let s00_is = textureLoad(gi_low_indirect_specular, c00_i, 0);
    let s10_is = textureLoad(gi_low_indirect_specular, c10_i, 0);
    let s01_is = textureLoad(gi_low_indirect_specular, c01_i, 0);
    let s11_is = textureLoad(gi_low_indirect_specular, c11_i, 0);

    let out_d = s00_d * fw00 + s10_d * fw10 + s01_d * fw01 + s11_d * fw11;
    let out_id = s00_id * fw00 + s10_id * fw10 + s01_id * fw01 + s11_id * fw11;
    let out_is = s00_is * fw00 + s10_is * fw10 + s01_is * fw01 + s11_is * fw11;

    var out_d_8 = vec4<f32>(0.0);
    var out_id_8 = vec4<f32>(0.0);
    var out_is_8 = vec4<f32>(0.0);
    var weight_sum_8 = 0.0;

    if (should_use_eight_tap) {
        let center_gi_i = clamp_i32(vec2<i32>(round(gi_pos)), vec2<i32>(0, 0), max_gi_i);
        let tap_offsets = array<vec2<i32>, 8>(
            vec2<i32>(0, 0),
            vec2<i32>(1, 0),
            vec2<i32>(-1, 0),
            vec2<i32>(0, 1),
            vec2<i32>(0, -1),
            vec2<i32>(1, 1),
            vec2<i32>(-1, 1),
            vec2<i32>(1, -1)
        );

        for (var tap_index = 0u; tap_index < 8u; tap_index = tap_index + 1u) {
            let tap_i = clamp_i32(center_gi_i + tap_offsets[tap_index], vec2<i32>(0, 0), max_gi_i);
            let tap_full = gi_pixel_to_full_res_pixel_coord(vec2<u32>(tap_i), upscale_factor, full_res);

            let tap_position = textureLoad(gbuffer_position, vec2<i32>(tap_full), 0).xyz;
            let tap_normal = safe_normalize(textureLoad(gbuffer_normal, vec2<i32>(tap_full), 0).xyz);
            let tap_depth = length(tap_position - camera_position);

            let delta = vec2<f32>(tap_i) - gi_pos;
            let distance_weight = exp(-dot(delta, delta) * EIGHT_TAP_DISTANCE_FALLOFF);
            let edge_weight = compute_edge_weight(center_depth, center_normal, tap_depth, tap_normal);
            let tap_weight = distance_weight * edge_weight;

            out_d_8 = out_d_8 + textureLoad(gi_low_direct, tap_i, 0) * tap_weight;
            out_id_8 = out_id_8 + textureLoad(gi_low_indirect_diffuse, tap_i, 0) * tap_weight;
            out_is_8 = out_is_8 + textureLoad(gi_low_indirect_specular, tap_i, 0) * tap_weight;
            weight_sum_8 = weight_sum_8 + tap_weight;
        }
    }

    let use_eight_tap_resolve = should_use_eight_tap && (weight_sum_8 > 1e-6);
    out_d_8 = out_d_8 / max(weight_sum_8, 1e-6);
    out_id_8 = out_id_8 / max(weight_sum_8, 1e-6);
    out_is_8 = out_is_8 / max(weight_sum_8, 1e-6);

    let final_out_d = select(out_d, out_d_8, use_eight_tap_resolve);
    let final_out_id = select(out_id, out_id_8, use_eight_tap_resolve);
    let final_out_is = select(out_is, out_is_8, use_eight_tap_resolve);

    let direct = safe_clamp_vec3_max(final_out_d.xyz, MAX_RADIANCE_LUMINANCE);
    let indirect_diffuse = safe_clamp_vec3_max(final_out_id.xyz, MAX_RADIANCE_LUMINANCE);
    let indirect_specular = safe_clamp_vec3_max(final_out_is.xyz, MAX_RADIANCE_LUMINANCE);

    textureStore(out_direct, full_pixel_coord, vec4<f32>(direct, final_out_d.w));
    textureStore(out_indirect_diffuse, full_pixel_coord, vec4<f32>(indirect_diffuse, final_out_id.w));
    textureStore(out_indirect_specular, full_pixel_coord, vec4<f32>(indirect_specular, final_out_is.w));
}


