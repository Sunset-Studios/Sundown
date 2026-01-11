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

// =============================================================================
// OCTAHEDRAL ATLAS CONSTANTS
// =============================================================================
// - Each probe stores directional irradiance in an octahedral map (paper: 8x8)
// - Each probe tile is padded with a 1-texel duplicated gutter to avoid bilinear
//   filtering artifacts across atlas tile boundaries.
// - Each probe stores depth moments in an octahedral map (paper: 16x16)
// =============================================================================
const DDGI_PROBE_IRRADIANCE_RES = 8u;
const DDGI_PROBE_DEPTH_RES = 16u;
const DDGI_PROBE_ATLAS_GUTTER = 1u;
const PROBE_SAMPLE_CAP = 32.0;
const GOLDEN_RATIO_CONJUGATE = 0.6180339887498948;

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

struct DDGIProbeRayHit {
    hit_pos_t: vec4<f32>,         // xyz = world hit position, w = t (>=0) or -1 for miss
    ray_dir_prim: vec4<f32>,      // xyz = ray direction, w = prim_store as f32 (undefined if miss)
    world_n_section: vec4<f32>,   // xyz = world geometric normal, w = section_index as f32
    world_t_uvx: vec4<f32>,       // xyz = world tangent, w = uv.x
    world_b_uvy: vec4<f32>,       // xyz = world bitangent, w = uv.y
    state_u32: vec4<u32>,         // x = lobe_type (0 = diffuse, 1 = specular), y = alive, z = shadow_visible, w = tri_id
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

// ─────────────────────────────────────────────────────────────────────────────
// Sample SH irradiance from probes
// ─────────────────────────────────────────────────────────────────────────────
fn ddgi_sample_sh_irradiance(
    ddgi_params: ptr<uniform, DDGIParams>,
    sh_probes: ptr<storage, array<u32>, read>,
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

    // -------------------------------------------------------------------------
    // Unified self-shadow bias (paper-style)
    //
    // Bias_vector = (n * 0.2 + w_o * 0.8) * (0.75 * D) * B
    //
    // - n   : surface normal (world)
    // - w_o : direction from surface point to camera (world)
    // - D   : minimum axial distance between probes (probe spacing)
    // - B   : user-tunable scalar (`ddgi_params.self_shadow_bias`)
    //
    // We apply this world-space bias to the *visibility query point* (not the
    // shading point itself) to reduce shadow leaking near the mean of the depth
    // distribution in the probe depth moments atlas.
    // -------------------------------------------------------------------------
    let view_index = u32(frame_info.view_index);
    let camera_position = view_buffer[view_index].view_position.xyz;
    let w_o = safe_normalize(camera_position - position);
    let b = (*ddgi_params).self_shadow_bias;
    let bias_vector = (normal_ws * 0.2 + w_o * 0.8) * (0.75 * spacing) * b;

    let perceptual_threshold = max(0.05 * MAX_RADIANCE_LUMINANCE, 1e-4);

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

                let biased_pos = position + bias_vector;
                let dir_from_probe = safe_normalize(biased_pos - probe_pos);
                let dist = length(biased_pos - probe_pos);

                let probe_sh = ddgi_sh_probe_read(sh_probes, probe_index);
                let preview_irradiance = max(ddgi_sh_evaluate_irradiance(probe_sh, normal_ws), vec3<f32>(0.0));
                let probe_luma = luminance(preview_irradiance);
                let perceptual_linear = clamp(probe_luma / perceptual_threshold, 0.0, 1.0);
                // Avoid "black holes" from perceptual weight reaching 0.0 everywhere.
                let perceptual_weight = max(perceptual_linear * perceptual_linear, 0.05);

                let weight = tri_weight * backface_weight * perceptual_weight;

                sh_sum = sh_l1_rgb_add(sh_sum, sh_l1_rgb_multiply_scalar(probe_sh, weight));
                weight_sum = weight_sum + weight;
            }
        }
    }

    let inv_weight_sum = 1.0 / weight_sum;
    let sh_interpolated = sh_l1_rgb_multiply_scalar(sh_sum, inv_weight_sum);
    var irradiance = ddgi_sh_evaluate_irradiance(sh_interpolated, normal_ws) * (*ddgi_params).indirect_boost;
    return max(irradiance, vec3<f32>(0.0));
}