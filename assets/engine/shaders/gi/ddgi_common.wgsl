#include "gi/gi_common.wgsl"
#include "sh_common.wgsl"

const GOLDEN_RATIO_CONJUGATE = 0.6180339887498948;
const DDGI_VISIBILITY_MIN_VARIANCE = 1e-4;
const DDGI_CASCADE_ACTIVE_OVERLAP_ROWS = 4.0;
const DDGI_CASCADE_BLEND_WINDOW_PROBES = 4.0;

// SH probes store L1 RGB coefficients (4 coefficients × 3 channels = 12 floats)
// packed into 6 u32 values using f16 packing for efficient storage.
const DDGI_SH_PROBE_SIZE_U32 = 6u;    // Size of packed SH L1 RGB in u32 units
const DDGI_SH_PROBE_SIZE_F32 = 12u;   // Size of unpacked SH L1 RGB in f32 units
const DDGI_MSME_STATS_SIZE_U32 = 8u;  // Packed short SH mean + scalar variance/inconsistency

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
const PROBE_STATE_CONVERGENCE_READINESS_MULTIPLIER_START: f32 = 2.0;
const PROBE_STATE_CONVERGENCE_READINESS_MULTIPLIER_END: f32 = 1.0;
const PROBE_STATE_CONVERGENCE_READINESS_MULTIPLIER_RAMP_FRAMES: u32 = 8u;
const PROBE_STATE_GATHER_STABLE_SAMPLE_COUNT_START: f32 = 2.0;
const PROBE_STATE_GATHER_STABLE_SAMPLE_COUNT_END: f32 = 8.0;
const PROBE_STATE_BACKFACE_THRESHOLD: f32  = 0.5;  // Fraction of backface hits = inside geometry
const PROBE_STATE_NEAR_GEOMETRY_DIST: f32  = 1.0;  // Multiplier of probe_spacing for "near"

// Probe scheduler priorities. IDs are intentionally spaced so new priorities
// can be inserted or reordered without renumbering every existing bucket.
const DDGI_PROBE_SCHEDULE_PRIORITY_NONE: u32 = 0u;
const DDGI_PROBE_SCHEDULE_PRIORITY_NORMAL: u32 = 100u;
const DDGI_PROBE_SCHEDULE_PRIORITY_FRESH: u32 = 200u;
const DDGI_PROBE_SCHEDULE_PRIORITY_BUCKET_NORMAL: u32 = 0u;
const DDGI_PROBE_SCHEDULE_PRIORITY_BUCKET_FRESH: u32 = 1u;
const DDGI_PROBE_SCHEDULE_PRIORITY_COUNT: u32 = 2u;

// Maximum number of DDGI cascades supported
const DDGI_MAX_CASCADES: u32 = 6u;

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
    depth_slot_params: vec4<f32>, // x=slot_count, y=words_per_slot, z=max_texels_per_probe, w=retention_frames
    frame_index: f32,
    indirect_boost: f32,
    cascade_count: f32,
    max_ray_length: f32,
    permutation_stride: f32,       // Precomputed coprime stride for probe cycling (CPU-computed)
    permutation_base_offset: f32,  // Precomputed base offset for permutation (CPU-computed)
    permutation_frame_stride: f32, // Precomputed frame stride for temporal offset (CPU-computed)
    _unused1: f32,
    cascades: array<DDGICascadeData, DDGI_MAX_CASCADES>, // Per-cascade data (origin, scroll, snap)
};

struct DDGIProbeRayData {
    // Keep this record scalar-aligned. These 13 words are the values that are
    // either expensive to regenerate in downstream passes or identify mutable
    // scene data. Probe/ray indices, hit position, and surface attributes are
    // reconstructed from the flat ray index and the data below.
    ray_dir_x: f32,
    ray_dir_y: f32,
    ray_dir_z: f32,
    hit_distance: f32,            // Negative for backfaces, zero for misses
    prim_store: u32,              // INVALID_IDX for misses
    vertex_index_0: u32,
    vertex_index_1: u32,
    vertex_index_2: u32,
    barycentric_u: f32,
    barycentric_v: f32,
    radiance_r: f32,              // Visible NEE before shade; final radiance after shade
    radiance_g: f32,
    radiance_b: f32,
};

struct ProbeStateData {
    packed_state: u32,        // state | init_frame_count | convergence_frame_count | flags
    sample_count: u32,        // sample count
}

struct DDGIMSMEProbeStats {
    short_mean: SH_L1_RGB_Packed,
    variance: f32,
    inconsistency: f32,
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

struct DDGIDepthSlotAllocatorState {
    free_count: atomic<u32>,
    allocation_failures: atomic<u32>,
    _pad0: atomic<u32>,
    _pad1: atomic<u32>,
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

fn ddgi_msme_stats_reset(
    stats_buffer: ptr<storage, array<DDGIMSMEProbeStats>, read_write>,
    probe_index: u32
) {
    stats_buffer[probe_index].short_mean = sh_l1_rgb_pack(sh_l1_rgb_zero());
    stats_buffer[probe_index].variance = 0.0;
    stats_buffer[probe_index].inconsistency = 0.0;
}


fn ddgi_max_rays_per_probe(ddgi_params: ptr<uniform, DDGIParams>) -> u32 {
    return max(1u, u32((*ddgi_params).probe_counts.y));
}

fn ddgi_probe_ray_direction(
    ddgi_params: ptr<uniform, DDGIParams>,
    probe_index: u32,
    ray_index_in_probe: u32,
    rays_per_probe: u32
) -> vec3<f32> {
    let ray_count = max(rays_per_probe, 1u);
    let ray_index = min(ray_index_in_probe, ray_count - 1u);
    let frame_index = u32((*ddgi_params).frame_index);
    var probe_rng = hash(
        probe_index
            ^ (frame_index * 0x9E3779B9u)
            ^ 0xA511E9B3u
    );
    let rotation_01 = rand_float(probe_rng);

    probe_rng = random_seed(probe_rng);
    let r1 = rand_float(probe_rng);
    probe_rng = random_seed(probe_rng);
    let r2 = rand_float(probe_rng);

    let z = 1.0 - 2.0 * r1;
    let rot_phi = 2.0 * PI * r2;
    let r_xy = sqrt(max(1.0 - z * z, 0.0));
    let z_axis = vec3<f32>(cos(rot_phi) * r_xy, sin(rot_phi) * r_xy, z);

    let u = (f32(ray_index) + 0.5) / f32(ray_count);
    let cos_theta = 1.0 - 2.0 * u;
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let phi = 2.0 * PI * fract(f32(ray_index) * GOLDEN_RATIO_CONJUGATE + rotation_01);
    let dir_local = vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    return orthonormalize(z_axis) * dir_local;
}

fn ddgi_probe_ray_stored_direction(ray: DDGIProbeRayData) -> vec3<f32> {
    return vec3<f32>(ray.ray_dir_x, ray.ray_dir_y, ray.ray_dir_z);
}

fn ddgi_probe_ray_radiance(ray: DDGIProbeRayData) -> vec3<f32> {
    return vec3<f32>(ray.radiance_r, ray.radiance_g, ray.radiance_b);
}

fn ddgi_probe_ray_set_radiance(ray: ptr<storage, DDGIProbeRayData, read_write>, radiance: vec3<f32>) {
    (*ray).radiance_r = radiance.x;
    (*ray).radiance_g = radiance.y;
    (*ray).radiance_b = radiance.z;
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

fn ddgi_depth_slot_count(ddgi_params: ptr<uniform, DDGIParams>) -> u32 {
    return u32((*ddgi_params).depth_slot_params.x);
}

fn ddgi_depth_words_per_slot(ddgi_params: ptr<uniform, DDGIParams>) -> u32 {
    return u32((*ddgi_params).depth_slot_params.y);
}

fn ddgi_depth_base_for_slot(ddgi_params: ptr<uniform, DDGIParams>, slot_index: u32) -> u32 {
    return slot_index * ddgi_depth_words_per_slot(ddgi_params);
}

fn ddgi_depth_slot_for_probe(
    probe_depth_slots: ptr<storage, array<u32>, read>,
    probe_index: u32
) -> u32 {
    let encoded_slot = (*probe_depth_slots)[probe_index];
    return select(INVALID_IDX, encoded_slot - 1u, encoded_slot != 0u);
}

fn ddgi_probe_miss_distance(
    ddgi_params: ptr<uniform, DDGIParams>,
    _probe_index: u32
) -> f32 {
    return max((*ddgi_params).max_ray_length, 0.001);
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
// Probe update compaction keeps the first `probes_per_frame` active slots, so
// the slot-to-probe mapping acts as the per-frame priority order. Use a
// frame-seeded bijective permutation instead of an affine walk to avoid visible
// update waves while still ensuring one slot maps to one probe.

fn ddgi_probe_permutation_hash(x: u32) -> u32 {
    var y = x;
    y = y ^ (y >> 16u);
    y = y * 0x85ebca6bu;
    y = y ^ (y >> 13u);
    y = y * 0xc2b2ae35u;
    y = y ^ (y >> 16u);
    return y;
}

fn ddgi_probe_permutation_domain_mask(probe_count: u32) -> u32 {
    var mask = max(probe_count - 1u, 1u);
    mask = mask | (mask >> 1u);
    mask = mask | (mask >> 2u);
    mask = mask | (mask >> 4u);
    mask = mask | (mask >> 8u);
    mask = mask | (mask >> 16u);
    return mask;
}

fn ddgi_probe_permute_power_of_two_domain(value: u32, mask: u32, seed: u32) -> u32 {
    let multiplier_a = (ddgi_probe_permutation_hash(seed ^ 0x27d4eb2du) | 1u) & mask;
    let multiplier_b = (ddgi_probe_permutation_hash(seed ^ 0x165667b1u) | 1u) & mask;

    var x = (value + (ddgi_probe_permutation_hash(seed ^ 0x9e3779b9u) & mask)) & mask;
    x = (x ^ (x >> 16u)) & mask;
    x = (x * multiplier_a) & mask;
    x = (x ^ (x >> 15u)) & mask;
    x = (x * multiplier_b) & mask;
    x = (x ^ (x >> 16u)) & mask;
    x = (x + (ddgi_probe_permutation_hash(seed ^ 0x85ebca6bu) & mask)) & mask;
    return x;
}

fn ddgi_probe_index_from_permuted_slot(
    slot: u32,
    probe_count: u32,
    frame_index_u32: u32,
    stride: u32,
    base_offset: u32,
    frame_stride: u32
) -> u32 {
    let safe_probe_count = max(probe_count, 1u);
    if (safe_probe_count <= 1u) {
        return 0u;
    }

    let domain_mask = ddgi_probe_permutation_domain_mask(safe_probe_count);
    let seed = ddgi_probe_permutation_hash(
        base_offset ^
        (stride * 0x9e3779b9u) ^
        (frame_stride * 0x85ebca6bu) ^
        (frame_index_u32 * 0xc2b2ae35u)
    );

    var candidate = slot & domain_mask;
    loop {
        candidate = ddgi_probe_permute_power_of_two_domain(candidate, domain_mask, seed);
        if (candidate < safe_probe_count) {
            return candidate;
        }
    }
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
fn ddgi_probe_index_from_world_position(
    ddgi_params: ptr<uniform, DDGIParams>,
    cascade_index: u32,
    position: vec3<f32>
) -> u32 {
    let spacing = ddgi_cascade_spacing(ddgi_params, cascade_index);
    let origin = ddgi_cascade_origin(ddgi_params, cascade_index);
    let max_coord = vec3<i32>(
        i32((*ddgi_params).probe_grid_mask.x),
        i32((*ddgi_params).probe_grid_mask.y),
        i32((*ddgi_params).probe_grid_mask.z)
    );
    let relative_position = (position - origin) / spacing;
    let nearest_coord = vec3<u32>(
        clamp(vec3<i32>(round(relative_position)), vec3<i32>(0), max_coord)
    );
    return ddgi_probe_index_from_coord(ddgi_params, cascade_index, nearest_coord);
}

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
// For cascade N > 0: true if the probe is outside cascade N-1's core. The
// core is contracted by at least one row of cascade N probes, so active probes
// for coarser cascades overlap inward over the next finer cascade edge.
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
    
    // Compute inner cascade's contracted core AABB.
    let inner_min = inner_origin;
    let inner_max = inner_origin + (dims - vec3<f32>(1.0)) * inner_spacing;
    let overlap_distance = ddgi_cascade_spacing(ddgi_params, cascade_index) * DDGI_CASCADE_ACTIVE_OVERLAP_ROWS;
    let inner_core_min = inner_min + vec3<f32>(overlap_distance);
    let inner_core_max = inner_max - vec3<f32>(overlap_distance);
    let has_inner_core = all(inner_core_min <= inner_core_max);
    
    // Probe is in the shell if it is outside the contracted inner core.
    let inside_inner = has_inner_core
        && all(probe_pos >= inner_core_min)
        && all(probe_pos <= inner_core_max);
    
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

fn ddgi_cascade_edge_distance_world(
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
    return min(dist_to_edge.x, min(dist_to_edge.y, dist_to_edge.z));
}

fn ddgi_position_in_coarser_active_overlap(
    ddgi_params: ptr<uniform, DDGIParams>,
    inner_cascade_index: u32,
    coarser_cascade_index: u32,
    position: vec3<f32>
) -> bool {
    let edge_distance = ddgi_cascade_edge_distance_world(ddgi_params, inner_cascade_index, position);
    let overlap_distance = ddgi_cascade_spacing(ddgi_params, coarser_cascade_index) * DDGI_CASCADE_ACTIVE_OVERLAP_ROWS;
    return edge_distance >= 0.0 && edge_distance <= overlap_distance;
}

// ─────────────────────────────────────────────────────────────────────────────
// Get the cascade index that contains a given world position.
// Returns the finest (lowest index) cascade whose bounds contain the position.
// Falls back to largest cascade if the position is outside all cascades.
// ─────────────────────────────────────────────────────────────────────────────
fn ddgi_cascade_index_for_position(
    ddgi_params: ptr<uniform, DDGIParams>,
    position: vec3<f32>
) -> u32 {
    let cascade_count = i32(ddgi_cascade_count(ddgi_params));
    
    var cascade_index = cascade_count - 1;
    for (var c = cascade_count - 1; c >= 0; c = c - 1) {
        let inside = ddgi_position_inside_cascade_bounds(ddgi_params, u32(c), position);
        cascade_index = select(cascade_index, c, inside);
    }
    
    return u32(cascade_index);
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
// Depth moments use one 16-bit mean/spread code per directional texel.
// Two texels share a u32; logarithmic quantization preserves precision near
// the probe while still covering the full miss-distance range.
// ─────────────────────────────────────────────────────────────────────────────
fn ddgi_depth_log_encode(value: f32, spacing: f32, miss_distance: f32) -> f32 {
    let safe_spacing = max(spacing, 1e-4);
    let log_range = max(log2(1.0 + miss_distance / safe_spacing), 1e-4);
    return saturate(log2(1.0 + clamp(value, 0.0, miss_distance) / safe_spacing) / log_range);
}

fn ddgi_depth_log_decode(value: f32, spacing: f32, miss_distance: f32) -> f32 {
    let safe_spacing = max(spacing, 1e-4);
    let log_range = max(log2(1.0 + miss_distance / safe_spacing), 1e-4);
    return safe_spacing * (exp2(saturate(value) * log_range) - 1.0);
}

// One 16-bit texel stores an 11-bit log mean and a 5-bit log standard
// deviation. The second moment is reconstructed as mean^2 + spread^2.
fn ddgi_depth_moments_pack(mean_t: f32, mean_t2: f32, spacing: f32, miss_distance: f32) -> u32 {
    let variance = max(mean_t2 - mean_t * mean_t, 0.0);
    let spread = sqrt(variance);
    let mean_q = u32(round(ddgi_depth_log_encode(mean_t, spacing, miss_distance) * 2047.0));
    let spread_q = u32(round(ddgi_depth_log_encode(spread, spacing, miss_distance) * 31.0));
    return mean_q | (spread_q << 11u);
}

fn ddgi_depth_moments_unpack(
    packed_texel: u32,
    spacing: f32,
    miss_distance: f32
) -> vec2<f32> {
    let mean_n = f32(packed_texel & 0x7ffu) / 2047.0;
    let spread_n = f32((packed_texel >> 11u) & 0x1fu) / 31.0;
    let mean_t = ddgi_depth_log_decode(mean_n, spacing, miss_distance);
    let spread = ddgi_depth_log_decode(spread_n, spacing, miss_distance);
    return vec2<f32>(mean_t, mean_t * mean_t + spread * spread);
}

fn ddgi_depth_word_texel(word: u32, texel_index: u32) -> u32 {
    let shift = (texel_index & 1u) * 16u;
    return (word >> shift) & 0xffffu;
}

fn ddgi_depth_word_replace_texel(word: u32, texel_index: u32, packed_texel: u32) -> u32 {
    let shift = (texel_index & 1u) * 16u;
    let mask = 0xffffu << shift;
    return (word & ~mask) | ((packed_texel & 0xffffu) << shift);
}

fn ddgi_depth_moments_load(
    probe_depth_moments: ptr<storage, array<u32>, read>,
    slot_base: u32,
    texel_index: u32,
    spacing: f32,
    miss_distance: f32
) -> vec2<f32> {
    let word = (*probe_depth_moments)[slot_base + texel_index / 2u];
    return ddgi_depth_moments_unpack(
        ddgi_depth_word_texel(word, texel_index),
        spacing,
        miss_distance
    );
}

fn ddgi_depth_texel_id(texel_coord: vec2<u32>, depth_res: u32) -> u32 {
    return texel_coord.x + texel_coord.y * depth_res;
}

fn ddgi_visibility_weight_from_moments(
    ddgi_params: ptr<uniform, DDGIParams>,
    probe_depth_moments: ptr<storage, array<u32>, read>,
    probe_depth_slots: ptr<storage, array<u32>, read>,
    probe_index: u32,
    direction_from_probe: vec3<f32>,
    dist: f32
) -> f32 {
    let depth_slot = ddgi_depth_slot_for_probe(probe_depth_slots, probe_index);
    if (depth_slot == INVALID_IDX) {
        return 0.0;
    }

    let depth_res = ddgi_depth_resolution_for_probe(ddgi_params, probe_index);

    // Map direction to octahedral UV in texel space with half-texel offset
    // so that texel centers align with integer coordinates for bilinear filtering
    let uv = encode_octahedral(direction_from_probe) * f32(depth_res) - 0.5;
    let base_f = floor(uv);
    let frac = uv - base_f;
    let base_i = vec2<i32>(base_f);

    let base_idx = ddgi_depth_base_for_slot(ddgi_params, depth_slot);
    let max_coord = i32(depth_res) - 1;
    let spacing = ddgi_probe_spacing_from_index(ddgi_params, probe_index);
    let miss_distance = ddgi_probe_miss_distance(ddgi_params, probe_index);

    // Bilinear sample with clamped coordinates to prevent out-of-bounds reads
    let c00 = vec2<u32>(clamp(base_i, vec2<i32>(0), vec2<i32>(max_coord)));
    let c10 = vec2<u32>(clamp(base_i + vec2<i32>(1, 0), vec2<i32>(0), vec2<i32>(max_coord)));
    let c01 = vec2<u32>(clamp(base_i + vec2<i32>(0, 1), vec2<i32>(0), vec2<i32>(max_coord)));
    let c11 = vec2<u32>(clamp(base_i + vec2<i32>(1, 1), vec2<i32>(0), vec2<i32>(max_coord)));

    let m00 = ddgi_depth_moments_load(probe_depth_moments, base_idx, ddgi_depth_texel_id(c00, depth_res), spacing, miss_distance);
    let m10 = ddgi_depth_moments_load(probe_depth_moments, base_idx, ddgi_depth_texel_id(c10, depth_res), spacing, miss_distance);
    let m01 = ddgi_depth_moments_load(probe_depth_moments, base_idx, ddgi_depth_texel_id(c01, depth_res), spacing, miss_distance);
    let m11 = ddgi_depth_moments_load(probe_depth_moments, base_idx, ddgi_depth_texel_id(c11, depth_res), spacing, miss_distance);

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

fn ddgi_probe_schedule_priority_for_state(state: u32, convergence_frames: u32) -> u32 {
    if (
        state == PROBE_STATE_UNINITIALIZED ||
        probe_state_is_newly(state) ||
        (convergence_frames > 0u && convergence_frames <= PROBE_STATE_CONVERGENCE_FRAMES)
    ) {
        return DDGI_PROBE_SCHEDULE_PRIORITY_FRESH;
    }

    return DDGI_PROBE_SCHEDULE_PRIORITY_NORMAL;
}

fn ddgi_probe_schedule_priority_bucket(priority: u32) -> u32 {
    if (priority == DDGI_PROBE_SCHEDULE_PRIORITY_FRESH) {
        return DDGI_PROBE_SCHEDULE_PRIORITY_BUCKET_FRESH;
    }

    return DDGI_PROBE_SCHEDULE_PRIORITY_BUCKET_NORMAL;
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

    let state_readiness = min(1.0, f32(convergence_frames) / readiness_frames_target);
    let stability_readiness = smoothstep(PROBE_STATE_GATHER_STABLE_SAMPLE_COUNT_START, PROBE_STATE_GATHER_STABLE_SAMPLE_COUNT_END, f32(state_data.sample_count));

    return min(state_readiness, stability_readiness);
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
    probe_depth_slots: ptr<storage, array<u32>, read>,
    position: vec3<f32>,
    normal_ws: vec3<f32>,
    cascade_index: u32
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
    var ready_weight_sum = 0.0;

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
                probe_depth_slots,
                probe_index,
                dir_from_probe,
                dist
            );
        }

        // Perceptual weight
        {
            let crush_threshold = 0.9;
            if (weight < crush_threshold) {
                weight *= (weight * weight) / (crush_threshold * crush_threshold);
            }
        }

        // Trilinear weight
        {
            let trilinear_weight = mix(vec3<f32>(1.0) - alpha, alpha, trilinear_index_offsets[i]);
            weight *= trilinear_weight.x * trilinear_weight.y * trilinear_weight.z;
        }

        let ready_weight = weight * probe_readiness;
        let probe_sh = ddgi_sh_probe_read(sh_probes, probe_index);
        sh_sum = sh_l1_rgb_add(sh_sum, sh_l1_rgb_multiply_scalar(probe_sh, ready_weight));
        weight_sum += weight;
        ready_weight_sum += ready_weight;
    }

    var result: DDGISampleResult;
    
    if (ready_weight_sum > 1e-6) {
        let sh_interpolated = sh_l1_rgb_multiply_scalar(sh_sum, 1.0 / ready_weight_sum);
        result.irradiance = max(ddgi_sh_evaluate_irradiance(sh_interpolated, normal_ws) * (*ddgi_params).indirect_boost, vec3<f32>(0.0));
        result.readiness = saturate(ready_weight_sum / max(weight_sum, 1e-6));
    } else {
        result.irradiance = vec3<f32>(0.0);
        result.readiness = 0.0;
    }
    
    return result;
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
    probe_depth_slots: ptr<storage, array<u32>, read>,
    position: vec3<f32>,
    normal_ws: vec3<f32>
) -> vec3<f32> {
    let cascade_count = ddgi_cascade_count(ddgi_params);
    let cascade_index = ddgi_cascade_index_for_position(ddgi_params, position);
    
    let irradiance_fine = ddgi_sample_sh_irradiance_single_cascade_internal(
        ddgi_params,
        sh_probes,
        probe_states,
        probe_depth_moments,
        probe_depth_slots,
        position,
        normal_ws,
        cascade_index
    );

    // Edge blending between cascades for smooth spatial transitions
    let coarser_index = cascade_index + 1u;
    let has_coarser = coarser_index < cascade_count;
    let blend_weight = select(0.0, ddgi_cascade_blend_weight(ddgi_params, cascade_index, position), has_coarser);

    if (blend_weight > 0.0) {
        let irradiance_coarse = ddgi_sample_sh_irradiance_single_cascade_internal(
            ddgi_params,
            sh_probes,
            probe_states,
            probe_depth_moments,
            probe_depth_slots,
            position,
            normal_ws,
            coarser_index
        );
        return mix(irradiance_fine.irradiance, irradiance_coarse.irradiance, blend_weight);
    }

    return irradiance_fine.irradiance;
}
