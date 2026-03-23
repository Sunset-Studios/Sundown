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
// ║  • Dead probe detection                                                   ║
// ║                                                                           ║
// ║  To minimize calculations, we reuse ray-tracing information.              ║
// ║  Backface hits are encoded with negative distances in the ray data        ║
// ║  (hit_pos_t.w < 0). A probe is considered "dead" (inside geometry) when   ║
// ║  the fraction of backface hits exceeds PROBE_STATE_BACKFACE_THRESHOLD.    ║
// ║  This approach handles non-manifold and intersecting geometry robustly by ║
// ║  counting multiple rays rather than relying on a single ray test.         ║
// ║                                                                           ║
// ║  Based on "Improving Probes in Dynamic Diffuse Global Illumination" by D  ║
// ║  Rohacek et al.                                                           ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

const GOLDEN_RATIO_CONJUGATE = 0.6180339887498948;
const DDGI_VISIBILITY_MIN_VARIANCE = 1e-4;
const DDGI_CASCADE_BLEND_WINDOW_PROBES = 4.0;

// SH probes store L1 RGB coefficients (4 coefficients × 3 channels = 12 floats)
// packed into 6 u32 values using f16 packing for efficient storage.
const DDGI_SH_PROBE_SIZE_U32 = 6u;    // Size of packed SH L1 RGB in u32 units
const DDGI_SH_PROBE_SIZE_F32 = 12u;   // Size of unpacked SH L1 RGB in f32 units

// Probe states - stored as u32 per probe
const PROBE_STATE_UNINITIALIZED: u32 = 0u;   // Default - needs classification
const PROBE_STATE_OFF: u32           = 1u;   // Inside static geometry - never trace
const PROBE_STATE_SLEEPING: u32      = 2u;   // No geometry nearby - skip tracing
const PROBE_STATE_NEWLY_AWAKE: u32   = 3u;   // Just woken by dynamic - fast converge
const PROBE_STATE_NEWLY_VIGILANT: u32= 4u;   // Just initialized near static - fast converge
const PROBE_STATE_VIGILANT: u32      = 5u;   // Near static geometry - always trace
const PROBE_STATE_AWAKE: u32         = 6u;   // Near dynamic geometry - trace while active

// Classification parameters
const PROBE_STATE_INIT_FRAMES: u32         = 1u;   // Frames of tracing for classification
const PROBE_STATE_CONVERGENCE_FRAMES: u32  = 2u;   // Frames for "Newly" states to converge
const PROBE_STATE_CONVERGENCE_READINESS_MULTIPLIER_START: f32 = 8.0;
const PROBE_STATE_CONVERGENCE_READINESS_MULTIPLIER_END: f32 = 1.0;
const PROBE_STATE_CONVERGENCE_READINESS_MULTIPLIER_RAMP_FRAMES: u32 = 16u;
const PROBE_STATE_BACKFACE_THRESHOLD: f32  = 0.5;  // Fraction of backface hits = inside geometry
const PROBE_STATE_NEAR_GEOMETRY_DIST: f32  = 1.0;  // Multiplier of probe_spacing for "near"
const PROBE_STATE_FLAG_SURFACE_VISIBLE: u32 = 1u;  // Bit 0 of flags byte (bit 24 of packed_state)

// Maximum number of DDGI cascades supported
const DDGI_MAX_CASCADES: u32 = 4u;

struct DDGICascadeData {
    origin_spacing: vec4<f32>, // xyz = cascade origin, w = probe spacing
    scroll_offset: vec4<f32>,  // xyz = ring buffer scroll offset (probe cells), w = unused
    snap_delta: vec4<f32>,     // xyz = delta in probe cells, w = active (1/0)
    depth_atlas_info: vec4<f32>, // x = depth_res, y = depth_texel_count_per_probe, z = depth_base_texel_offset, w = unused
};

struct DDGIParams {
    probe_counts: vec4<f32>,      // x=probe_count_total, y=max_rays_per_probe, z=probes_per_frame, w=probe_spacing_base
    probe_grid_dims: vec4<f32>,   // x=dim_x, y=dim_y, z=dim_z, w=probe_radius
    probe_grid_origin: vec4<f32>, // xyz = grid origin, w = unused
    probe_grid_log2: vec4<f32>,   // xyz = log2(dim_*), w = unused
    probe_grid_mask: vec4<f32>,   // xyz = (dim_*-1), w = unused
    probe_grid_snap_delta: vec4<f32>, // xyz = delta in probe cells, w = active (1/0)
    frame_index: f32,
    indirect_boost: f32,
    cascade_count: f32,
    probe_update_culled_ratio: f32,
    permutation_stride: f32,       // Precomputed coprime stride for probe cycling (CPU-computed)
    permutation_base_offset: f32,  // Precomputed base offset for permutation (CPU-computed)
    permutation_frame_stride: f32, // Precomputed frame stride for temporal offset (CPU-computed)
    min_rays_per_probe: f32,       // min_rays_per_probe
    cascades: array<DDGICascadeData, DDGI_MAX_CASCADES>, // Per-cascade data (origin, scroll, snap)
};

struct DDGIProbeRayData {
    hit_pos_t: vec4<f32>,
    ray_dir_prim: vec4<f32>,      // xyz = ray direction, w = ray PDF (set by init; preserved by hit)
    nee_light_dir_type: vec4<f32>, // xyz = selected NEE light dir, w = 0 analytic / 1 emissive
    nee_light_radiance: vec4<f32>, // xyz = selected NEE radiance scale (without n_dot_l), w = unused
    world_n_section: vec4<f32>,   // xyz = world geometric normal, w = section_index as f32
    world_t_uvx: vec4<f32>,       // xyz = world tangent, w = uv.x
    world_b_uvy: vec4<f32>,       // xyz = world bitangent, w = uv.y
    state_u32: vec4<u32>,         // x = prim_store, y = alive, z = shadow_visible, w = tri_id_local
    radiance: vec4<f32>,          // xyz = shaded ray radiance, w = 1.0 (or unused)
    meta_u32: vec4<u32>,          // x = probe_index, yzw = reserved
};

struct ProbeStateData {
    packed_state: u32,        // state | init_frame_count | convergence_frame_count | flags (bit 0 = cull_visible)
    sample_count: u32,        // sample count
}

struct DDGIProbeRayDataHeader {
    active_ray_count: atomic<u32>,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct DDGIProbeRayDataHeaderReadOnly {
    active_ray_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct DDGIProbeRayDataBuffer {
    header: DDGIProbeRayDataHeader,
    rays: array<DDGIProbeRayData>,
};

struct DDGIProbeRayDataBufferReadOnlyHeader {
    header: DDGIProbeRayDataHeaderReadOnly,
    rays: array<DDGIProbeRayData>,
};

// ─────────────────────────────────────────────────────────────────────────────
// Result struct for sampling with readiness tracking
// Used for cascade fallback blending during probe initialization
// ─────────────────────────────────────────────────────────────────────────────
struct DDGISampleResult {
    irradiance: vec3<f32>,
    readiness: f32,  // Weighted average of probe readiness (0.0 to 1.0)
}

fn ddgi_probe_state_get_sample_count(probe_state: ptr<storage, ProbeStateData, read_write>) -> u32 {
    return (*probe_state).sample_count;
}

fn ddgi_probe_state_set_sample_count(probe_state: ptr<storage, ProbeStateData, read_write>, count: u32) {
    (*probe_state).sample_count = count;
}


fn ddgi_max_rays_per_probe(ddgi_params: ptr<uniform, DDGIParams>) -> u32 {
    return max(1u, u32((*ddgi_params).probe_counts.y));
}

fn ddgi_min_rays_per_probe(ddgi_params: ptr<uniform, DDGIParams>) -> u32 {
    return min(ddgi_max_rays_per_probe(ddgi_params), max(1u, u32((*ddgi_params).min_rays_per_probe)));
}

fn ddgi_probe_count_per_cascade(ddgi_params: ptr<uniform, DDGIParams>) -> u32 {
    let dims = vec3<u32>(
        u32((*ddgi_params).probe_grid_dims.x),
        u32((*ddgi_params).probe_grid_dims.y),
        u32((*ddgi_params).probe_grid_dims.z)
    );
    return dims.x * dims.y * dims.z;
}

fn ddgi_cascade_count(ddgi_params: ptr<uniform, DDGIParams>) -> u32 {
    return max(1u, u32((*ddgi_params).cascade_count));
}

fn ddgi_probe_cascade_index(ddgi_params: ptr<uniform, DDGIParams>, probe_index: u32) -> u32 {
    let probes_per_cascade = ddgi_probe_count_per_cascade(ddgi_params);
    return probe_index / max(probes_per_cascade, 1u);
}

fn ddgi_probe_index_in_cascade(ddgi_params: ptr<uniform, DDGIParams>, probe_index: u32) -> u32 {
    let probes_per_cascade = ddgi_probe_count_per_cascade(ddgi_params);
    return probe_index - ddgi_probe_cascade_index(ddgi_params, probe_index) * probes_per_cascade;
}

fn ddgi_cascade_origin(ddgi_params: ptr<uniform, DDGIParams>, cascade_index: u32) -> vec3<f32> {
    return (*ddgi_params).cascades[cascade_index].origin_spacing.xyz;
}

fn ddgi_cascade_spacing(ddgi_params: ptr<uniform, DDGIParams>, cascade_index: u32) -> f32 {
    return (*ddgi_params).cascades[cascade_index].origin_spacing.w;
}

fn ddgi_depth_resolution_for_cascade(ddgi_params: ptr<uniform, DDGIParams>, cascade_index: u32) -> u32 {
    return u32((*ddgi_params).cascades[cascade_index].depth_atlas_info.x);
}

fn ddgi_depth_texel_count_for_cascade(ddgi_params: ptr<uniform, DDGIParams>, cascade_index: u32) -> u32 {
    return u32((*ddgi_params).cascades[cascade_index].depth_atlas_info.y);
}

fn ddgi_depth_base_offset_for_cascade(ddgi_params: ptr<uniform, DDGIParams>, cascade_index: u32) -> u32 {
    return u32((*ddgi_params).cascades[cascade_index].depth_atlas_info.z);
}

fn ddgi_probe_spacing_from_index(ddgi_params: ptr<uniform, DDGIParams>, probe_index: u32) -> f32 {
    return ddgi_cascade_spacing(ddgi_params, ddgi_probe_cascade_index(ddgi_params, probe_index));
}

fn ddgi_depth_resolution_for_probe(ddgi_params: ptr<uniform, DDGIParams>, probe_index: u32) -> u32 {
    let cascade_index = ddgi_probe_cascade_index(ddgi_params, probe_index);
    return ddgi_depth_resolution_for_cascade(ddgi_params, cascade_index);
}

fn ddgi_depth_texel_count_for_probe(ddgi_params: ptr<uniform, DDGIParams>, probe_index: u32) -> u32 {
    let cascade_index = ddgi_probe_cascade_index(ddgi_params, probe_index);
    return ddgi_depth_texel_count_for_cascade(ddgi_params, cascade_index);
}

fn ddgi_depth_base_for_probe(ddgi_params: ptr<uniform, DDGIParams>, probe_index: u32) -> u32 {
    let cascade_index = ddgi_probe_cascade_index(ddgi_params, probe_index);
    let local_probe_index = ddgi_probe_index_in_cascade(ddgi_params, probe_index);
    let cascade_base_offset = ddgi_depth_base_offset_for_cascade(ddgi_params, cascade_index);
    let depth_texel_count = ddgi_depth_texel_count_for_cascade(ddgi_params, cascade_index);
    return cascade_base_offset + local_probe_index * depth_texel_count;
}

fn ddgi_cascade_scroll_offset(ddgi_params: ptr<uniform, DDGIParams>, cascade_index: u32) -> vec3<u32> {
    return vec3<u32>(
        u32((*ddgi_params).cascades[cascade_index].scroll_offset.x),
        u32((*ddgi_params).cascades[cascade_index].scroll_offset.y),
        u32((*ddgi_params).cascades[cascade_index].scroll_offset.z)
    );
}

// =============================================================================
// STOCHASTIC (BUT DETERMINISTIC) PROBE CYCLING
// =============================================================================
// We want a selection pattern that:
// - Looks "random" to avoid structured artifacts (better temporal distribution)
// - Is deterministic (given probe_count and frame_index)
// - Does not miss probes: every probe index must be visited eventually
//
// Approach:
// - Treat the per-frame probe picks as a walk over Z_n (n = probe_count).
// - Use an affine map:   idx(k) = (offset + k * stride) mod n
// - If gcd(stride, n) = 1, this is a permutation: k=0..n-1 visits every probe once.
// - We set k = frame_index * probes_per_frame + local_id so the walk advances by
//   probes_per_frame each frame without gaps.
//
// OPTIMIZATION: The stride, base_offset, and frame_stride values are UNIFORM
// across all shader invocations (they only depend on probe_count). These are
// precomputed on the CPU and passed via DDGIParams, eliminating expensive
// GCD computation loops that previously ran per-thread.
//
// The permutation formula is:
//   probe_index = (base_offset + (frame_stride * frame_index + slot) * stride) % probe_count
fn ddgi_probe_index_from_permuted_slot(
    slot: u32,
    probe_count: u32,
    frame_index_u32: u32,
    stride: u32,
    base_offset: u32,
    frame_stride: u32
) -> u32 {
    let safe_probe_count = max(probe_count, 1u);
    let frame_shift = frame_index_u32 * frame_stride;
    let k = frame_shift + slot;
    return (base_offset + k * stride) % safe_probe_count;
}


fn ddgi_probe_storage_coord_from_index(
    ddgi_params: ptr<uniform, DDGIParams>,
    probe_index: u32
) -> vec3<u32> {
    let shift_x = u32((*ddgi_params).probe_grid_log2.x);
    let shift_y = u32((*ddgi_params).probe_grid_log2.y);

    let mask_x = u32((*ddgi_params).probe_grid_mask.x);
    let mask_y = u32((*ddgi_params).probe_grid_mask.y);
    let mask_z = u32((*ddgi_params).probe_grid_mask.z);

    let local_index = ddgi_probe_index_in_cascade(ddgi_params, probe_index);
    let x = local_index & mask_x;
    let y = (local_index >> shift_x) & mask_y;
    let z = (local_index >> (shift_x + shift_y)) & mask_z;

    return vec3<u32>(x, y, z);
}

fn ddgi_probe_coord_from_index(ddgi_params: ptr<uniform, DDGIParams>, probe_index: u32) -> vec3<u32> {
    let mask = vec3<u32>(
        u32((*ddgi_params).probe_grid_mask.x),
        u32((*ddgi_params).probe_grid_mask.y),
        u32((*ddgi_params).probe_grid_mask.z)
    );
    let dims = mask + vec3<u32>(1u);

    let cascade_index = ddgi_probe_cascade_index(ddgi_params, probe_index);
    let scroll = ddgi_cascade_scroll_offset(ddgi_params, cascade_index);
    let storage_coord = ddgi_probe_storage_coord_from_index(ddgi_params, probe_index);
    let world_coord = (storage_coord + dims - scroll) & mask;

    return world_coord;
}

fn ddgi_probe_index_from_coord(
    ddgi_params: ptr<uniform, DDGIParams>,
    cascade_index: u32,
    coord: vec3<u32>
) -> u32 {
    let shift_x = u32((*ddgi_params).probe_grid_log2.x);
    let shift_y = u32((*ddgi_params).probe_grid_log2.y);
    let mask = vec3<u32>(
        u32((*ddgi_params).probe_grid_mask.x),
        u32((*ddgi_params).probe_grid_mask.y),
        u32((*ddgi_params).probe_grid_mask.z)
    );
    let scroll = ddgi_cascade_scroll_offset(ddgi_params, cascade_index);
    let storage_coord = (coord + scroll) & mask;
    let local_index =
        storage_coord.x |
        (storage_coord.y << shift_x) |
        (storage_coord.z << (shift_x + shift_y));
    let probes_per_cascade = ddgi_probe_count_per_cascade(ddgi_params);
    return cascade_index * probes_per_cascade + local_index;
}

// ─────────────────────────────────────────────────────────────────────────────
// 3D depth atlas cell coordinates: XZ within layer, Y as layer index
// ─────────────────────────────────────────────────────────────────────────────
fn ddgi_probe_world_position_from_index(ddgi_params: ptr<uniform, DDGIParams>, probe_index: u32) -> vec3<f32> {
    let cascade_index = ddgi_probe_cascade_index(ddgi_params, probe_index);
    let spacing = ddgi_cascade_spacing(ddgi_params, cascade_index);
    let origin = ddgi_cascade_origin(ddgi_params, cascade_index);
    let coord = ddgi_probe_coord_from_index(ddgi_params, probe_index);
    return origin + vec3<f32>(f32(coord.x), f32(coord.y), f32(coord.z)) * spacing;
}

fn ddgi_probe_world_position_from_coord(
    ddgi_params: ptr<uniform, DDGIParams>,
    cascade_index: u32,
    coord: vec3<u32>
) -> vec3<f32> {
    let spacing = ddgi_cascade_spacing(ddgi_params, cascade_index);
    let origin = ddgi_cascade_origin(ddgi_params, cascade_index);
    return origin + vec3<f32>(f32(coord.x), f32(coord.y), f32(coord.z)) * spacing;
}

// ─────────────────────────────────────────────────────────────────────────────
// Clipmap-style cascade shell selection
// Returns true if a probe belongs to its cascade's "shell" (toroidal region).
// For cascade 0: always true (innermost cascade covers entire bounds)
// For cascade N > 0: true only if probe is outside cascade N-1's bounds
// ─────────────────────────────────────────────────────────────────────────────
fn ddgi_probe_in_cascade_shell(
    ddgi_params: ptr<uniform, DDGIParams>,
    probe_index: u32
) -> bool {
    let cascade_index = ddgi_probe_cascade_index(ddgi_params, probe_index);
    
    // Cascade 0 always includes all its probes (innermost)
    if (cascade_index == 0u) {
        return true;
    }
    
    // Get probe world position
    let probe_pos = ddgi_probe_world_position_from_index(ddgi_params, probe_index);
    
    // Get inner cascade's (N-1) bounds
    let inner_cascade = cascade_index - 1u;
    let inner_origin = ddgi_cascade_origin(ddgi_params, inner_cascade);
    let inner_spacing = ddgi_cascade_spacing(ddgi_params, inner_cascade);
    
    let dims = vec3<f32>(
        (*ddgi_params).probe_grid_dims.x,
        (*ddgi_params).probe_grid_dims.y,
        (*ddgi_params).probe_grid_dims.z
    );
    
    // Compute inner cascade's AABB
    let inner_min = inner_origin;
    let inner_max = inner_origin + (dims - vec3<f32>(1.0)) * inner_spacing;
    
    // Probe is in the shell if it's outside the inner cascade's bounds
    let inside_inner = all(probe_pos >= inner_min) && all(probe_pos <= inner_max);
    
    return !inside_inner;
}

fn ddgi_probe_in_cascade(
    ddgi_params: ptr<uniform, DDGIParams>,
    probe_index: u32
) -> bool {
    let cascade_index = ddgi_probe_cascade_index(ddgi_params, probe_index);
    let probe_pos = ddgi_probe_world_position_from_index(ddgi_params, probe_index);
    let origin = ddgi_cascade_origin(ddgi_params, cascade_index);
    let spacing = ddgi_cascade_spacing(ddgi_params, cascade_index);
    let dims = vec3<f32>(
        (*ddgi_params).probe_grid_dims.x,
        (*ddgi_params).probe_grid_dims.y,
        (*ddgi_params).probe_grid_dims.z
    );
    let max_bound = origin + (dims - vec3<f32>(1.0)) * spacing;
    return all(probe_pos <= max_bound);
}

fn ddgi_position_inside_cascade_bounds(
    ddgi_params: ptr<uniform, DDGIParams>,
    cascade_index: u32,
    position: vec3<f32>
) -> bool {
    let dims_f = vec3<f32>(
        (*ddgi_params).probe_grid_dims.x,
        (*ddgi_params).probe_grid_dims.y,
        (*ddgi_params).probe_grid_dims.z
    );
    let spacing = ddgi_cascade_spacing(ddgi_params, cascade_index);
    let origin = ddgi_cascade_origin(ddgi_params, cascade_index);
    let max_bound = origin + (dims_f - vec3<f32>(1.0)) * spacing;
    return
        position.x >= origin.x && position.y >= origin.y && position.z >= origin.z &&
        position.x <= max_bound.x && position.y <= max_bound.y && position.z <= max_bound.z;
}

// ─────────────────────────────────────────────────────────────────────────────
// Get the cascade index that contains a given world position.
// Returns the finest (lowest index) cascade whose bounds contain the position.
// Falls back to cascade 0 if the position is outside all cascades.
// ─────────────────────────────────────────────────────────────────────────────
fn ddgi_cascade_index_for_position(
    ddgi_params: ptr<uniform, DDGIParams>,
    position: vec3<f32>
) -> u32 {
    let cascade_count = ddgi_cascade_count(ddgi_params);
    
    var cascade_index = 0u;
    for (var c = 0u; c < cascade_count; c = c + 1u) {
        let inside = ddgi_position_inside_cascade_bounds(ddgi_params, c, position);
        cascade_index = select(cascade_index, c, inside);
        if (inside) {
            break;
        }
    }
    
    return cascade_index;
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
    buffer: ptr<storage, array<u32>, read_write>,
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
    return sh_l1_rgb_calculate_irradiance(sh, normal);
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

// ─────────────────────────────────────────────────────────────────────────────
// Depth moments are packed into a single u32 per texel using f16 packing:
//   bits [0..15]  = mean distance     (f16)
//   bits [16..31] = mean distance²    (f16)
// This halves storage compared to the previous vec4<f32> layout while
// retaining sufficient precision for Chebyshev visibility testing.
// ─────────────────────────────────────────────────────────────────────────────
fn ddgi_depth_moments_pack(mean_t: f32, mean_t2: f32) -> u32 {
    return pack2x16float(vec2<f32>(mean_t, mean_t2));
}

fn ddgi_depth_moments_unpack(packed: u32) -> vec2<f32> {
    return unpack2x16float(packed);
}

fn ddgi_depth_texel_id(texel_coord: vec2<u32>, depth_res: u32) -> u32 {
    return texel_coord.x + texel_coord.y * depth_res;
}

fn ddgi_visibility_weight_from_moments(
    ddgi_params: ptr<uniform, DDGIParams>,
    probe_depth_moments: ptr<storage, array<u32>, read>,
    probe_index: u32,
    direction_from_probe: vec3<f32>,
    dist: f32
) -> f32 {
    let depth_res = ddgi_depth_resolution_for_probe(ddgi_params, probe_index);

    // Map direction to octahedral UV in texel space with half-texel offset
    // so that texel centers align with integer coordinates for bilinear filtering
    let uv = encode_octahedral(direction_from_probe) * f32(depth_res) - 0.5;
    let base_f = floor(uv);
    let frac = uv - base_f;
    let base_i = vec2<i32>(base_f);

    let base_idx = ddgi_depth_base_for_probe(ddgi_params, probe_index);
    let max_coord = i32(depth_res) - 1;

    // Bilinear sample with clamped coordinates to prevent out-of-bounds reads
    let c00 = vec2<u32>(clamp(base_i, vec2<i32>(0), vec2<i32>(max_coord)));
    let c10 = vec2<u32>(clamp(base_i + vec2<i32>(1, 0), vec2<i32>(0), vec2<i32>(max_coord)));
    let c01 = vec2<u32>(clamp(base_i + vec2<i32>(0, 1), vec2<i32>(0), vec2<i32>(max_coord)));
    let c11 = vec2<u32>(clamp(base_i + vec2<i32>(1, 1), vec2<i32>(0), vec2<i32>(max_coord)));

    let m00 = ddgi_depth_moments_unpack((*probe_depth_moments)[base_idx + ddgi_depth_texel_id(c00, depth_res)]);
    let m10 = ddgi_depth_moments_unpack((*probe_depth_moments)[base_idx + ddgi_depth_texel_id(c10, depth_res)]);
    let m01 = ddgi_depth_moments_unpack((*probe_depth_moments)[base_idx + ddgi_depth_texel_id(c01, depth_res)]);
    let m11 = ddgi_depth_moments_unpack((*probe_depth_moments)[base_idx + ddgi_depth_texel_id(c11, depth_res)]);

    let moments = mix(mix(m00, m10, frac.x), mix(m01, m11, frac.x), frac.y);

    let mean_d = moments.x;
    let mean_d2 = moments.y;

    let variance = max(mean_d2 - mean_d * mean_d, DDGI_VISIBILITY_MIN_VARIANCE);

    let delta = max(0.0, dist - mean_d);
    var chebyshev_weight = variance / (variance + delta * delta);
    
    // Softer contrast (square instead of cube) to reduce banding
    chebyshev_weight = chebyshev_weight * chebyshev_weight;

    return select(chebyshev_weight, 1.0, dist <= mean_d);
}

// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                    PROBE STATE INTEGRATION                                ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Defines probe states for adaptive update scheduling. Not all probes in   ║
// ║  a uniform grid contribute equally - many may be inside walls, in open    ║
// ║  space, or far from visible surfaces. This system classifies probes to    ║
// ║  skip unnecessary work while maintaining visual quality.                  ║
// ║                                                                           ║
// ║  State Machine:                                                           ║
// ║  ┌─────────────────────────────────────────────────────────────────────┐  ║
// ║  │                                                                     │  ║
// ║  │    ┌──────────────────────────────────────────────────────────┐     │  ║
// ║  │    │             UNINITIALIZED (0)                            │     │  ║
// ║  │    │      (default state, needs classification)               │     │  ║
// ║  │    └───────────────────────┬──────────────────────────────────┘     │  ║
// ║  │                            │ after init rays                        │  ║
// ║  │              ┌─────────────┼─────────────┐                          │  ║
// ║  │              ▼             ▼             ▼                          │  ║
// ║  │    ┌─────────────┐  ┌───────────┐  ┌──────────────────┐             │  ║
// ║  │    │  OFF (1)    │  │ SLEEPING  │  │  NEWLY_VIGILANT  │             │  ║
// ║  │    │  (in wall)  │  │   (2)     │  │      (4)         │             │  ║
// ║  │    │  never      │  │  no       │  │  fast hysteresis │             │  ║
// ║  │    │  trace      │  │  nearby   │  └────────┬─────────┘             │  ║
// ║  │    └─────────────┘  │  geometry │           │                       │  ║
// ║  │                     └───────────┘           │ after convergence     │  ║
// ║  │                                             ▼                       │  ║
// ║  │                                       ┌──────────────┐              │  ║
// ║  │                                       │  VIGILANT    │              │  ║
// ║  │                                       │    (5)       │              │  ║
// ║  │                                       │  static geo  │              │  ║
// ║  │                                       │  shading     │              │  ║
// ║  │                                       └──────────────┘              │  ║
// ║  │                                                                     │  ║
// ║  │  Note: NEWLY_AWAKE (3) and AWAKE (6) states are reserved for        │  ║
// ║  │  future dynamic object handling.                                    │  ║
// ║  │                                                                     │  ║
// ║  └─────────────────────────────────────────────────────────────────────┘  ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// Extract the probe state from packed data
// ─────────────────────────────────────────────────────────────────────────────
fn probe_state_get_state(packed: u32) -> u32 {
    return packed & 0xFFu;
}

// ─────────────────────────────────────────────────────────────────────────────
// Extract the initialization frame count (0-255)
// ─────────────────────────────────────────────────────────────────────────────
fn probe_state_get_init_frames(packed: u32) -> u32 {
    return (packed >> 8u) & 0xFFu;
}

// ─────────────────────────────────────────────────────────────────────────────
// Extract the convergence frame count (0-255)
// ─────────────────────────────────────────────────────────────────────────────
fn probe_state_get_convergence_frames(packed: u32) -> u32 {
    return (packed >> 16u) & 0xFFu;
}

// ─────────────────────────────────────────────────────────────────────────────
// Extract flags (top 8 bits)
// ─────────────────────────────────────────────────────────────────────────────
fn probe_state_get_flags(packed: u32) -> u32 {
    return (packed >> 24u) & 0xFFu;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pack state data into a single u32
// ─────────────────────────────────────────────────────────────────────────────
fn probe_state_pack(state: u32, init_frames: u32, convergence_frames: u32, flags: u32) -> u32 {
    return (state & 0xFFu) |
           ((init_frames & 0xFFu) << 8u) |
           ((convergence_frames & 0xFFu) << 16u) |
           ((flags & 0xFFu) << 24u);
}

// ─────────────────────────────────────────────────────────────────────────────
// Check if a probe is in a "newly" state (needs fast convergence)
// ─────────────────────────────────────────────────────────────────────────────
fn probe_state_is_newly(state: u32) -> bool {
    return state == PROBE_STATE_NEWLY_AWAKE || state == PROBE_STATE_NEWLY_VIGILANT;
}

// ─────────────────────────────────────────────────────────────────────────────
// Compute probe readiness weight (0.0 to 1.0) based on state and convergence
// Used for cascade fallback blending during probe initialization
// ─────────────────────────────────────────────────────────────────────────────
fn ddgi_probe_readiness_weight(state_data: ProbeStateData) -> f32 {
    let state = probe_state_get_state(state_data.packed_state);
    let convergence_frames = probe_state_get_convergence_frames(state_data.packed_state);
    
    // UNINITIALIZED and OFF probes have no valid data
    if (state == PROBE_STATE_OFF || state == PROBE_STATE_SLEEPING) {
        return 0.0;
    }

    let readiness_multiplier_t = min(
        1.0,
        f32(convergence_frames) / f32(max(1u, PROBE_STATE_CONVERGENCE_READINESS_MULTIPLIER_RAMP_FRAMES))
    );
    let readiness_multiplier = mix(
        PROBE_STATE_CONVERGENCE_READINESS_MULTIPLIER_START,
        PROBE_STATE_CONVERGENCE_READINESS_MULTIPLIER_END,
        readiness_multiplier_t
    );
    let readiness_frames_target = f32(PROBE_STATE_CONVERGENCE_FRAMES) * readiness_multiplier;

    // VIGILANT and AWAKE are fully ready
    return min(1.0, f32(convergence_frames) / readiness_frames_target);
}

// ─────────────────────────────────────────────────────────────────────────────
// Check if a probe should be traced (updated with new rays)
// ─────────────────────────────────────────────────────────────────────────────
fn probe_state_is_active(state: u32) -> bool {
    // Active for tracing if: UNINITIALIZED, NEWLY_AWAKE, NEWLY_VIGILANT, VIGILANT, or AWAKE
    // Don't trace if: OFF or SLEEPING (OFF is inside geometry, SLEEPING has no nearby geometry)
    return state != PROBE_STATE_OFF && state != PROBE_STATE_SLEEPING;
}

// ─────────────────────────────────────────────────────────────────────────────
// Check if a probe should be used for shading/sampling
// Only probes in active states (VIGILANT, AWAKE, NEWLY_*) have valid SH data.
// SLEEPING and OFF probes are not traced and have no meaningful data.
// NEWLY_* states are valid - their partial contribution is handled by
// ddgi_probe_readiness_weight() for cascade fallback blending.
// ─────────────────────────────────────────────────────────────────────────────
fn probe_state_is_valid_for_sampling(state_data: ProbeStateData) -> bool {
    let state = probe_state_get_state(state_data.packed_state);
    return state == PROBE_STATE_VIGILANT
        || state == PROBE_STATE_AWAKE
        || state == PROBE_STATE_NEWLY_VIGILANT
        || state == PROBE_STATE_NEWLY_AWAKE;
}

// ─────────────────────────────────────────────────────────────────────────────
// Transition a probe from UNINITIALIZED based on classification results
// Called after PROBE_STATE_INIT_FRAMES of tracing
// ─────────────────────────────────────────────────────────────────────────────
fn probe_state_classify_initial(backface_ratio: f32) -> u32 {
    // If most rays hit backfaces, probe is inside geometry
    if (backface_ratio > PROBE_STATE_BACKFACE_THRESHOLD) {
        return PROBE_STATE_OFF;
    }
    
    // Otherwise, probe is near static geometry
    return PROBE_STATE_NEWLY_VIGILANT;
}

// ─────────────────────────────────────────────────────────────────────────────
// Transition from NEWLY_* to stable state after convergence
// ─────────────────────────────────────────────────────────────────────────────
fn probe_state_after_convergence(current_state: u32) -> u32 {
    if (current_state == PROBE_STATE_NEWLY_AWAKE) {
        return PROBE_STATE_AWAKE;
    }
    if (current_state == PROBE_STATE_NEWLY_VIGILANT) {
        return PROBE_STATE_VIGILANT;
    }
    return current_state;
}

// ─────────────────────────────────────────────────────────────────────────────
// Check if a probe is valid for shading based on its state
// Returns 1.0 for active probes, 0.0 for inactive/sleeping/off probes
// ─────────────────────────────────────────────────────────────────────────────
fn ddgi_probe_state_weight(
    probe_states: ptr<storage, array<ProbeStateData>, read>,
    probe_index: u32
) -> f32 {
    let state = probe_state_get_state(probe_states[probe_index].packed_state);
    return select(0.0, 1.0, probe_state_is_active(state));
}

fn ddgi_cascade_blend_weight(
    ddgi_params: ptr<uniform, DDGIParams>,
    cascade_index: u32,
    position: vec3<f32>
) -> f32 {
    let dims_f = vec3<f32>(
        (*ddgi_params).probe_grid_dims.x,
        (*ddgi_params).probe_grid_dims.y,
        (*ddgi_params).probe_grid_dims.z
    );
    let spacing = ddgi_cascade_spacing(ddgi_params, cascade_index);
    let origin = ddgi_cascade_origin(ddgi_params, cascade_index);
    let max_bound = origin + (dims_f - vec3<f32>(1.0)) * spacing;

    let dist_to_min = position - origin;
    let dist_to_max = max_bound - position;
    let dist_to_edge = min(dist_to_min, dist_to_max);
    let min_edge_dist = min(dist_to_edge.x, min(dist_to_edge.y, dist_to_edge.z));
    let edge_dist_probes = min_edge_dist / max(spacing, 1e-6);
    let blend_window = max(DDGI_CASCADE_BLEND_WINDOW_PROBES, 1e-6);

    return clamp(1.0 - edge_dist_probes / blend_window, 0.0, 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Sample SH irradiance from probes with state awareness and offset-aware filtering
//
// This function incorporates:
// - Probe state weighting (OFF and SLEEPING probes are excluded)
// - Offset-aware trilinear interpolation that accounts for probe displacement
//   from the uniform grid (via spiral optimizer for dead probe relocation)
//
// The offset-aware filtering ensures:
// - Weights remain in [0,1] range (no oversaturation or light subtraction)
// - Proper interpolation even when probes are moved from grid positions
// ─────────────────────────────────────────────────────────────────────────────
fn ddgi_sample_sh_irradiance_single_cascade_internal(
    ddgi_params: ptr<uniform, DDGIParams>,
    sh_probes: ptr<storage, array<u32>, read_write>,
    probe_states: ptr<storage, array<ProbeStateData>, read_write>,
    probe_depth_moments: ptr<storage, array<u32>, read>,
    position: vec3<f32>,
    normal_ws: vec3<f32>,
    cascade_index: u32,
    mark_surface_visible: bool
) -> DDGISampleResult {
    let dims = vec3<u32>(
        u32((*ddgi_params).probe_grid_dims.x),
        u32((*ddgi_params).probe_grid_dims.y),
        u32((*ddgi_params).probe_grid_dims.z)
    );

    let spacing = ddgi_cascade_spacing(ddgi_params, cascade_index);
    let origin = ddgi_cascade_origin(ddgi_params, cascade_index);

    let view_index = u32(frame_info.view_index);
    let camera_position = view_buffer[view_index].view_position.xyz;
    let bias_offset = (normal_ws * 0.2 + normalize(camera_position - position) * 0.8) * (0.75 * spacing);
    let offset_pos = position + bias_offset;

    let rel = (offset_pos - origin) / spacing;
    let base = floor(rel);
    let alpha = fract(rel);

    let trilinear_index_offsets: array<vec3<f32>, 8> = array<vec3<f32>, 8>(
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(1.0, 1.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
        vec3<f32>(0.0, 1.0, 1.0),
        vec3<f32>(1.0, 1.0, 1.0),
        vec3<f32>(1.0, 0.0, 1.0),
    );

    var sh_sum = sh_l1_rgb_zero();
    var weight_sum = 0.0;
    var readiness_weighted_sum = 0.0;

    // Do trilinear interpolation for sampling
    for (var i = 0; i < 8; i = i + 1) {
        let coord = vec3<u32>(base) + vec3<u32>(trilinear_index_offsets[i]);
        let clamped_coord = clamp(coord, vec3<u32>(0u), dims - vec3<u32>(1u));
        let probe_index = ddgi_probe_index_from_coord(ddgi_params, cascade_index, clamped_coord);

        // Skip probes that don't have valid data for sampling
        if (!probe_state_is_valid_for_sampling((*probe_states)[probe_index])) {
            continue;
        }

        // Get probe readiness for cascade fallback blending
        let probe_readiness = ddgi_probe_readiness_weight((*probe_states)[probe_index]);

        var weight = 1.0;

        let probe_pos = ddgi_probe_world_position_from_coord(
            ddgi_params,
            cascade_index,
            clamped_coord
        );

        let dir_to_probe = normalize(probe_pos - position);

        let to_probe = offset_pos - probe_pos;
        let dist = length(to_probe);
        let dir_from_probe = to_probe / dist;

        // Backface weight
        {
            let backface = max(0.00001, (dot(normal_ws, dir_to_probe) + 1.0) * 0.5);
            weight *= (backface * backface) + 0.2;
        }

        // Probe visibility weight from depth moments
        {
            weight *= ddgi_visibility_weight_from_moments(
                ddgi_params,
                probe_depth_moments,
                probe_index,
                dir_from_probe,
                dist
            );
        }

        // Perceptual weight
        {
            let crush_threshold = 0.95;
            if (weight < crush_threshold) {
                weight *= (weight * weight) / (crush_threshold * crush_threshold);
            }
        }

        // Trilinear weight
        {
            let trilinear_weight = mix(vec3<f32>(1.0) - alpha, alpha, trilinear_index_offsets[i]);
            weight *= trilinear_weight.x * trilinear_weight.y * trilinear_weight.z;
        }

        let probe_sh = ddgi_sh_probe_read(sh_probes, probe_index);
        sh_sum = sh_l1_rgb_add(sh_sum, sh_l1_rgb_multiply_scalar(probe_sh, weight));
        weight_sum += weight;
        readiness_weighted_sum += weight * probe_readiness;
    }

    // Mark probes along trilinear neighborhood of surfaces as active
    if (mark_surface_visible) {
        for (var i = 0; i < 8; i = i + 1) {
            let coord = vec3<u32>(base) + vec3<u32>(trilinear_index_offsets[i]);
            let clamped_coord = clamp(coord, vec3<u32>(0u), dims - vec3<u32>(1u));
            let probe_index = ddgi_probe_index_from_coord(ddgi_params, cascade_index, clamped_coord);

            let state = probe_state_get_state(probe_states[probe_index].packed_state);
            var flags = probe_state_get_flags(probe_states[probe_index].packed_state);

            if (state == PROBE_STATE_SLEEPING || state == PROBE_STATE_OFF) {
                flags = flags | PROBE_STATE_FLAG_SURFACE_VISIBLE;
                probe_states[probe_index].packed_state = probe_state_pack(PROBE_STATE_UNINITIALIZED, 0u, 0u, flags);
            }
        }
    }

    var result: DDGISampleResult;
    
    if (weight_sum > 1e-6) {
        let sh_interpolated = sh_l1_rgb_multiply_scalar(sh_sum, 1.0 / weight_sum);
        result.irradiance = max(ddgi_sh_evaluate_irradiance(sh_interpolated, normal_ws) * (*ddgi_params).indirect_boost, vec3<f32>(0.0));
        result.readiness = saturate(readiness_weighted_sum / weight_sum);
    } else {
        result.irradiance = vec3<f32>(0.0);
        result.readiness = 0.0;
    }
    
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Sample SH irradiance with cascade fallback for initializing probes
// When fine cascade probes aren't fully ready, blends with coarser cascade data
// ─────────────────────────────────────────────────────────────────────────────
fn ddgi_sample_sh_irradiance_with_fallback(
    ddgi_params: ptr<uniform, DDGIParams>,
    sh_probes: ptr<storage, array<u32>, read_write>,
    probe_states: ptr<storage, array<ProbeStateData>, read_write>,
    probe_depth_moments: ptr<storage, array<u32>, read>,
    position: vec3<f32>,
    normal_ws: vec3<f32>,
    start_cascade: u32,
    mark_surface_visible: bool
) -> vec3<f32> {
    let cascade_count = ddgi_cascade_count(ddgi_params);
    
    // Sample the starting cascade
    let fine_result = ddgi_sample_sh_irradiance_single_cascade_internal(
        ddgi_params,
        sh_probes,
        probe_states,
        probe_depth_moments,
        position,
        normal_ws,
        start_cascade,
        mark_surface_visible
    );
    
    // If no coarser cascade, return as-is
    let next_cascade = start_cascade + 1u;
    if (next_cascade >= cascade_count) {
        return fine_result.irradiance;
    }
    
    // Sample coarser cascade for fallback
    // Use iteration instead of recursion (WGSL limitation)
    var accumulated_irradiance = fine_result.irradiance * fine_result.readiness;
    var accumulated_weight = fine_result.readiness;
    var remaining_weight = 1.0 - fine_result.readiness;
    var current_cascade = next_cascade;
    
    // Iterate through coarser cascades until we have full coverage
    //for (var iter = 0u; remaining_weight > 0.0001 && current_cascade < cascade_count; iter = iter + 1u) {
    if (remaining_weight > 0.0001 && current_cascade < cascade_count) {
        let coarse_result = ddgi_sample_sh_irradiance_single_cascade_internal(
            ddgi_params,
            sh_probes,
            probe_states,
            probe_depth_moments,
            position,
            normal_ws,
            current_cascade,
            false /* mark_surface_visible */
        );
        
        // Contribute proportionally to the remaining weight needed
        let contribute_weight = remaining_weight * coarse_result.readiness;
        accumulated_irradiance = accumulated_irradiance + coarse_result.irradiance * contribute_weight;
        accumulated_weight = accumulated_weight + contribute_weight;
    }
    
    // Normalize by total weight (handles case where not all cascades are ready)
    return select(
        fine_result.irradiance,
        accumulated_irradiance / accumulated_weight,
        accumulated_weight > 0.0001
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main sampling entry point with state awareness and cascade fallback
// Handles both:
// - Readiness-based fallback to coarser cascades for initializing probes
// - Edge blending between cascades for smooth transitions
// ─────────────────────────────────────────────────────────────────────────────
fn ddgi_sample_sh_irradiance_with_states(
    ddgi_params: ptr<uniform, DDGIParams>,
    sh_probes: ptr<storage, array<u32>, read_write>,
    probe_states: ptr<storage, array<ProbeStateData>, read_write>,
    probe_depth_moments: ptr<storage, array<u32>, read>,
    position: vec3<f32>,
    normal_ws: vec3<f32>
) -> vec3<f32> {
    let cascade_count = ddgi_cascade_count(ddgi_params);
    let cascade_index = ddgi_cascade_index_for_position(ddgi_params, position);
    
    // Use fallback-aware sampling to handle initializing probes
    let irradiance_fine = ddgi_sample_sh_irradiance_with_fallback(
        ddgi_params,
        sh_probes,
        probe_states,
        probe_depth_moments,
        position,
        normal_ws,
        cascade_index,
        true /* mark_surface_visible */
    );

    // Edge blending between cascades for smooth spatial transitions
    let coarser_index = cascade_index + 1u;
    let has_coarser = coarser_index < cascade_count;
    let blend_weight = select(0.0, ddgi_cascade_blend_weight(ddgi_params, cascade_index, position), has_coarser);

    if (blend_weight > 0.0) {
        // Also use fallback-aware sampling for the coarser cascade
        let irradiance_coarse = ddgi_sample_sh_irradiance_with_fallback(
            ddgi_params,
            sh_probes,
            probe_states,
            probe_depth_moments,
            position,
            normal_ws,
            coarser_index,
            false /* mark_surface_visible */
        );
        return mix(irradiance_fine, irradiance_coarse, blend_weight);
    }

    return irradiance_fine;
}
