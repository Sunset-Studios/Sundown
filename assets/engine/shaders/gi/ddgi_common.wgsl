#include "gi/gi_common.wgsl"
#include "sh_common.wgsl"

// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                    DDGI PROBE COMMON DEFINITIONS                          ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  This file contains shared structures and utilities for DDGI probes:      ║
// ║  • Octahedral atlas layout for radiance/depth storage                     ║
// ║  • Spherical harmonics (SH) probe representation                          ║
// ║  • Probe grid indexing and coordinate conversion                          ║
// ║  • Sampling helpers for both octahedral and SH representations            ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

const GOLDEN_RATIO_CONJUGATE = 0.6180339887498948;

// =============================================================================
// OCTAHEDRAL DEPTH MOMENTS CONSTANTS (16x16)
// =============================================================================
const DDGI_PROBE_DEPTH_RES = 8u;
const DDGI_DEPTH_TEXEL_COUNT = 64u;

// =============================================================================
// DEPTH MOMENTS VISIBILITY CONSTANTS (octahedral, temporally accumulated)
// =============================================================================
const DDGI_VISIBILITY_MIN_VARIANCE = 1e-4;
const DDGI_VISIBILITY_POWER = 2.0;
const DDGI_VISIBILITY_CONFIDENCE_MIN = 1e-3;
const DDGI_VISBILITY_DISTANCE_THICKNESS_BIAS = 0.01;
const DDGI_PERCEPTUAL_FALLOFF_THRESHOLD = 0.05;

// =============================================================================
// SPHERICAL HARMONICS PROBE CONSTANTS
// =============================================================================
// SH probes store L1 RGB coefficients (4 coefficients × 3 channels = 12 floats)
// packed into 6 u32 values using f16 packing for efficient storage.
// =============================================================================
const DDGI_SH_PROBE_SIZE_U32 = 6u;    // Size of packed SH L1 RGB in u32 units
const DDGI_SH_PROBE_SIZE_F32 = 12u;   // Size of unpacked SH L1 RGB in f32 units

struct DDGIParams {
    probe_counts: vec4<f32>,      // x=probe_count, y=rays_per_probe, z=probes_per_frame, w=probe_spacing
    probe_grid_dims: vec4<f32>,   // x=dim_x, y=dim_y, z=dim_z, w=probe_radius
    probe_grid_origin: vec4<f32>, // xyz = grid origin, w = unused
    probe_grid_log2: vec4<f32>,   // xyz = log2(dim_*), w = unused
    probe_grid_mask: vec4<f32>,   // xyz = (dim_*-1), w = unused
    probe_grid_snap_delta: vec4<f32>, // xyz = delta in probe cells, w = active (1/0)
    frame_index: u32,
    indirect_boost: f32,
    self_shadow_bias: f32,
    _pad0: f32,
};

struct DDGIProbeRayData {
    hit_pos_t: vec4<f32>,         // xyz = world hit position, w = t (>=0) or -1 for miss
    ray_dir_prim: vec4<f32>,      // xyz = ray direction, w = ray PDF (set by init; preserved by hit)
    world_n_section: vec4<f32>,   // xyz = world geometric normal, w = section_index as f32
    world_t_uvx: vec4<f32>,       // xyz = world tangent, w = uv.x
    world_b_uvy: vec4<f32>,       // xyz = world bitangent, w = uv.y
    state_u32: vec4<u32>,         // x = prim_store, y = alive|flags, z = shadow_visible, w = tri_id_local
    radiance: vec4<f32>,          // xyz = shaded ray radiance, w = 1.0 (or unused)
};

// ─────────────────────────────────────────────────────────────────────────────
// Storage structure for packed SH L1 RGB probe data
// Uses f16 packing: 12 floats (4 coefficients × 3 channels) → 6 u32 values
// ─────────────────────────────────────────────────────────────────────────────
struct DDGISHProbe {
    data: array<u32, 6>,
}

fn ddgi_probe_coord_from_index(ddgi_params: ptr<uniform, DDGIParams>, probe_index: u32) -> vec3<u32> {
    let shift_x = u32((*ddgi_params).probe_grid_log2.x);
    let shift_y = u32((*ddgi_params).probe_grid_log2.y);

    let mask_x = u32((*ddgi_params).probe_grid_mask.x);
    let mask_y = u32((*ddgi_params).probe_grid_mask.y);
    let mask_z = u32((*ddgi_params).probe_grid_mask.z);

    let x = probe_index & mask_x;
    let y = (probe_index >> shift_x) & mask_y;
    let z = (probe_index >> (shift_x + shift_y)) & mask_z;

    return vec3<u32>(x, y, z);
}

fn ddgi_probe_index_from_coord(ddgi_params: ptr<uniform, DDGIParams>, coord: vec3<u32>) -> u32 {
    let shift_x = u32((*ddgi_params).probe_grid_log2.x);
    let shift_y = u32((*ddgi_params).probe_grid_log2.y);
    return coord.x | (coord.y << shift_x) | (coord.z << (shift_x + shift_y));
}

// ─────────────────────────────────────────────────────────────────────────────
// 3D depth atlas cell coordinates: XZ within layer, Y as layer index
// ─────────────────────────────────────────────────────────────────────────────
fn ddgi_probe_world_position_from_index(ddgi_params: ptr<uniform, DDGIParams>, probe_index: u32) -> vec3<f32> {
    let spacing = (*ddgi_params).probe_counts.w;
    let origin = (*ddgi_params).probe_grid_origin.xyz;
    let coord = ddgi_probe_coord_from_index(ddgi_params, probe_index);
    return origin + vec3<f32>(f32(coord.x), f32(coord.y), f32(coord.z)) * spacing;
}

fn ddgi_probe_world_position_from_coord(ddgi_params: ptr<uniform, DDGIParams>, coord: vec3<u32>) -> vec3<f32> {
    let spacing = (*ddgi_params).probe_counts.w;
    let origin = (*ddgi_params).probe_grid_origin.xyz;
    return origin + vec3<f32>(f32(coord.x), f32(coord.y), f32(coord.z)) * spacing;
}

// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                 SPHERICAL HARMONICS PROBE HELPERS                         ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// Read packed SH probe from buffer and unpack to L1 RGB
// ─────────────────────────────────────────────────────────────────────────────
fn ddgi_sh_probe_read(
    buffer: ptr<storage, array<u32>, read>,
    probe_index: u32
) -> SH_L1_RGB {
    let base_offset = probe_index * DDGI_SH_PROBE_SIZE_U32;
    
    var packed: SH_L1_RGB_Packed;
    packed.data[0] = (*buffer)[base_offset + 0u];
    packed.data[1] = (*buffer)[base_offset + 1u];
    packed.data[2] = (*buffer)[base_offset + 2u];
    packed.data[3] = (*buffer)[base_offset + 3u];
    packed.data[4] = (*buffer)[base_offset + 4u];
    packed.data[5] = (*buffer)[base_offset + 5u];
    
    return sh_l1_rgb_unpack(packed);
}

// ─────────────────────────────────────────────────────────────────────────────
// Pack L1 RGB and write to buffer
// ─────────────────────────────────────────────────────────────────────────────
fn ddgi_sh_probe_write(
    buffer: ptr<storage, array<u32>, read_write>,
    probe_index: u32,
    sh: SH_L1_RGB
) {
    let base_offset = probe_index * DDGI_SH_PROBE_SIZE_U32;
    let packed = sh_l1_rgb_pack(sh);
    
    (*buffer)[base_offset + 0u] = packed.data[0];
    (*buffer)[base_offset + 1u] = packed.data[1];
    (*buffer)[base_offset + 2u] = packed.data[2];
    (*buffer)[base_offset + 3u] = packed.data[3];
    (*buffer)[base_offset + 4u] = packed.data[4];
    (*buffer)[base_offset + 5u] = packed.data[5];
}

// ─────────────────────────────────────────────────────────────────────────────
// Project a radiance sample onto SH basis
// direction: normalized direction of the sample
// radiance: radiance value in that direction
// weight: sample weight (typically 1 / PDF for Monte Carlo integration)
// ─────────────────────────────────────────────────────────────────────────────
fn ddgi_sh_project_sample(
    direction: vec3<f32>,
    radiance: vec3<f32>,
    weight: f32
) -> SH_L1_RGB {
    // Project the weighted radiance onto SH basis functions
    // For Monte Carlo integration: E[f(x)] ≈ (1/N) Σ f(x_i) / p(x_i)
    // Here weight = 1/p(x) for importance sampling
    return sh_project_onto_l1_rgb(direction, radiance * weight);
}

// ─────────────────────────────────────────────────────────────────────────────
// Evaluate irradiance from SH probe in a given direction
// Uses cosine lobe convolution for diffuse lighting
// ─────────────────────────────────────────────────────────────────────────────
fn ddgi_sh_evaluate_irradiance(
    sh: SH_L1_RGB,
    normal: vec3<f32>
) -> vec3<f32> {
    // Calculate irradiance using the Geometrics non-linear fit
    // This provides better quality than linear evaluation for L1
    return sh_l1_rgb_calculate_irradiance_geometrics(sh, normal);
}

// ─────────────────────────────────────────────────────────────────────────────
// Evaluate radiance from SH probe in a given direction
// Direct evaluation without cosine convolution (for debug/specular)
// ─────────────────────────────────────────────────────────────────────────────
fn ddgi_sh_evaluate_radiance(
    sh: SH_L1_RGB,
    direction: vec3<f32>
) -> vec3<f32> {
    return max(sh_l1_rgb_evaluate(sh, direction), vec3<f32>(0.0));
}

// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                  PROBE DEPTH MOMENTS VISIBILITY HELPERS                   ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================
fn ddgi_depth_texel_id(texel_coord: vec2<u32>) -> u32 {
    return texel_coord.x + texel_coord.y * DDGI_PROBE_DEPTH_RES;
}

fn ddgi_depth_clamp_texel_coord(texel_coord: vec2<i32>) -> vec2<u32> {
    let max_i = i32(DDGI_PROBE_DEPTH_RES) - 1;
    return vec2<u32>(
        u32(clamp(texel_coord.x, 0, max_i)),
        u32(clamp(texel_coord.y, 0, max_i))
    );
}

// Returns accumulated (mean_t, mean_t2, confidence) bilinearly sampled from the
// per-probe octahedral moments atlas.
fn ddgi_probe_depth_moments_sample_accum(
    probe_depth_moments: ptr<storage, array<vec4<f32>>, read>,
    probe_index: u32,
    direction_from_probe: vec3<f32>
) -> vec3<f32> {
    let res_f = f32(DDGI_PROBE_DEPTH_RES);
    let uv = encode_octahedral(direction_from_probe);
    let uv_f = uv * res_f - 0.5;

    let base_f = floor(uv_f);
    let frac = uv_f - base_f;
    let base_i = vec2<i32>(i32(base_f.x), i32(base_f.y));

    let w00 = (1.0 - frac.x) * (1.0 - frac.y);
    let w10 = frac.x * (1.0 - frac.y);
    let w01 = (1.0 - frac.x) * frac.y;
    let w11 = frac.x * frac.y;

    let t00_xy = ddgi_depth_clamp_texel_coord(base_i + vec2<i32>(0, 0));
    let t10_xy = ddgi_depth_clamp_texel_coord(base_i + vec2<i32>(1, 0));
    let t01_xy = ddgi_depth_clamp_texel_coord(base_i + vec2<i32>(0, 1));
    let t11_xy = ddgi_depth_clamp_texel_coord(base_i + vec2<i32>(1, 1));

    let base_offset = probe_index * DDGI_DEPTH_TEXEL_COUNT;

    let v00 = (*probe_depth_moments)[base_offset + ddgi_depth_texel_id(t00_xy)];
    let v10 = (*probe_depth_moments)[base_offset + ddgi_depth_texel_id(t10_xy)];
    let v01 = (*probe_depth_moments)[base_offset + ddgi_depth_texel_id(t01_xy)];
    let v11 = (*probe_depth_moments)[base_offset + ddgi_depth_texel_id(t11_xy)];

    // x = mean_t, y = mean_t2, z = confidence
    let a00 = vec3<f32>(v00.x, v00.y, v00.z) * w00;
    let a10 = vec3<f32>(v10.x, v10.y, v10.z) * w10;
    let a01 = vec3<f32>(v01.x, v01.y, v01.z) * w01;
    let a11 = vec3<f32>(v11.x, v11.y, v11.z) * w11;
    return a00 + a10 + a01 + a11;
}

fn ddgi_visibility_chebyshev(dist: f32, mean_d: f32, mean_d2: f32) -> f32 {
    // Moment shadow mapping / Chebyshev upper bound.
    // If the queried distance is in front of the mean, treat as visible.
    let variance = max(mean_d2 - mean_d * mean_d, DDGI_VISIBILITY_MIN_VARIANCE);
    let delta = dist - mean_d;
    let p_max = variance / (variance + delta * delta);
    let v = select(p_max, 1.0, dist <= mean_d);
    return clamp(v, 0.0, 1.0);
}

fn ddgi_visibility_weight_from_moments(
    probe_depth_moments: ptr<storage, array<vec4<f32>>, read>,
    probe_index: u32,
    direction_from_probe: vec3<f32>,
    dist: f32
) -> f32 {
    let biased_dist = dist + DDGI_VISBILITY_DISTANCE_THICKNESS_BIAS;
    let accum = ddgi_probe_depth_moments_sample_accum(probe_depth_moments, probe_index, direction_from_probe);
    // Chebyshev visibility with a mild power curve for sharper occluder rejection.
    var v = ddgi_visibility_chebyshev(biased_dist, accum.x, accum.y);
    // Confidence fade (low confidence -> treat as visible).
    return pow(mix(1.0, v, accum.z), DDGI_VISIBILITY_POWER);
}

// ─────────────────────────────────────────────────────────────────────────────
// Sample SH irradiance from probes
// ─────────────────────────────────────────────────────────────────────────────
fn ddgi_sample_sh_irradiance(
    ddgi_params: ptr<uniform, DDGIParams>,
    sh_probes: ptr<storage, array<u32>, read>,
    probe_depth_moments: ptr<storage, array<vec4<f32>>, read>,
    position: vec3<f32>,
    normal_ws: vec3<f32>
) -> vec3<f32> {
    let spacing = (*ddgi_params).probe_counts.w;
    let dims = vec3<u32>(
        u32((*ddgi_params).probe_grid_dims.x),
        u32((*ddgi_params).probe_grid_dims.y),
        u32((*ddgi_params).probe_grid_dims.z)
    );
    let origin = (*ddgi_params).probe_grid_origin.xyz;
    let probe_radius = (*ddgi_params).probe_grid_dims.w;

    let rel = (position - origin) / spacing;
    let base_f = floor(rel);
    let frac = rel - base_f;

    let max_base = vec3<f32>(
        f32(select(0u, dims.x - 2u, dims.x > 1u)),
        f32(select(0u, dims.y - 2u, dims.y > 1u)),
        f32(select(0u, dims.z - 2u, dims.z > 1u))
    );
    let base_clamped = clamp(base_f, vec3<f32>(0.0), max_base);
    let base = vec3<u32>(base_clamped);
    let frac_clamped = clamp(frac, vec3<f32>(0.0), vec3<f32>(1.0));

    let view_index = u32(frame_info.view_index);
    let camera_position = view_buffer[view_index].view_position.xyz;
    let w_o = safe_normalize(camera_position - position);
    let bias_offset = (0.2 * normal_ws + 0.8 * w_o) * (0.75 * spacing) * (*ddgi_params).self_shadow_bias;

    var sh_sum = sh_l1_rgb_zero();
    var weight_sum = 0.0;

    for (var z = 0u; z < 2u; z = z + 1u) {
        for (var y = 0u; y < 2u; y = y + 1u) {
            for (var x = 0u; x < 2u; x = x + 1u) {
                let coord = base + vec3<u32>(x, y, z);
                let clamped_coord = clamp(coord, vec3<u32>(0u), dims - vec3<u32>(1u));
                let probe_index = ddgi_probe_index_from_coord(ddgi_params, clamped_coord);

                let tri_weight =
                    select(1.0 - frac_clamped.x, frac_clamped.x, x == 1u) *
                    select(1.0 - frac_clamped.y, frac_clamped.y, y == 1u) *
                    select(1.0 - frac_clamped.z, frac_clamped.z, z == 1u);

                let probe_pos = ddgi_probe_world_position_from_coord(ddgi_params, clamped_coord);
                let dir_to_probe = safe_normalize(probe_pos - position);

                let backface = clamp(dot(normal_ws, dir_to_probe), 0.0, 1.0);
                let backface_weight = backface * backface;

                let offset_pos = position + bias_offset;
                let dir_from_probe = safe_normalize(offset_pos - probe_pos);
                let dist = length(offset_pos - probe_pos);

                // -----------------------------------------------------------------
                // Probe visibility weight from depth moments
                // -----------------------------------------------------------------
                let visibility_weight = ddgi_visibility_weight_from_moments(
                    probe_depth_moments,
                    probe_index,
                    dir_from_probe,
                    dist
                );

                let probe_sh = ddgi_sh_probe_read(sh_probes, probe_index);
                let preview_irradiance = ddgi_sh_evaluate_irradiance(probe_sh, normal_ws);
                let perceptual_linear = luminance(preview_irradiance) / DDGI_PERCEPTUAL_FALLOFF_THRESHOLD;
                let perceptual_weight = max(perceptual_linear * perceptual_linear, 1e-5);

                let weight = tri_weight * backface_weight * perceptual_weight * visibility_weight;

                sh_sum = sh_l1_rgb_add(sh_sum, sh_l1_rgb_multiply_scalar(probe_sh, weight));
                weight_sum = weight_sum + weight;
            }
        }
    }

    let inv_weight_sum = 1.0 / max(weight_sum, 1e-6);
    let sh_interpolated = sh_l1_rgb_multiply_scalar(sh_sum, inv_weight_sum);
    var irradiance = ddgi_sh_evaluate_irradiance(sh_interpolated, normal_ws) * (*ddgi_params).indirect_boost;
    return max(irradiance, vec3<f32>(0.0));
}