#include "gi/gi_common.wgsl"
#include "sh_common.wgsl"

// One unified hash table stores every LOD. A wider set-associative bucket
// keeps local hash pressure from turning into visible allocation holes.
const SURFACE_CACHE_BUCKET_SIZE: u32 = 16u;
const SURFACE_CACHE_LOD_EXTENT_CELLS: f32 = 128.0;
const SURFACE_CACHE_PATCH_EMPTY: u32 = 0u;
const SURFACE_CACHE_UPDATE_LOCKED: u32 = 0xffffffffu;
const SURFACE_CACHE_MIN_QUERY_SAMPLES: f32 = 2.0;
const SURFACE_CACHE_MAX_RADIANCE: f32 = 10.0;
const SURFACE_CACHE_SH_PATCH_SIZE_U32: u32 = 6u;

struct SurfaceCacheParams {
    surface_cache_size: f32,
    surface_cache_cell_size: f32,
    surface_cache_lod_count: f32,
    total_patch_count: f32,
    full_resolution_x: f32,
    full_resolution_y: f32,
    frame_index: f32,
    max_ray_length: f32,
    history_hysteresis: f32,
    max_history_samples: f32,
    indirect_boost: f32,
    importance_sample_count: f32,
    cache_entry_lifetime: f32,
    importance_exploration: f32,
    padding3: f32,
    padding2: f32,
};

struct SurfacePatch {
    position_frame: vec4<f32>,
    normal_lod: vec4<f32>,
    grid_key: vec4<i32>,
    metadata: vec4<f32>,
    history: vec4<f32>,
    fingerprint: atomic<u32>,
    update_frame: atomic<u32>,
    padding1: u32,
    padding2: u32,
};

struct SurfacePatchReadOnly {
    position_frame: vec4<f32>,
    normal_lod: vec4<f32>,
    grid_key: vec4<i32>,
    metadata: vec4<f32>,
    history: vec4<f32>,
    fingerprint: u32,
    update_frame: u32,
    padding1: u32,
    padding2: u32,
};

struct SurfaceCacheCounters {
    active_patch_count: atomic<u32>,
    padding0: atomic<u32>,
    padding1: atomic<u32>,
    padding2: atomic<u32>,
};

struct SurfaceCacheCountersReadOnly {
    active_patch_count: u32,
    padding0: u32,
    padding1: u32,
    padding2: u32,
};

// Compact geometric result for one active surface-cache ray. The hit-position
// w lane carries the RIS correction into the SH update; radiance remains in a
// separate buffer so geometry-only consumers do not pull radiance cache lines.
struct SurfaceCacheHitInfo {
    hit_position_sampling_weight: vec4<f32>,
    ray_direction_primitive: vec4<f32>,
    normal_section_index: vec4<f32>,
    hit_attr0: vec4<f32>,
    hit_attr1: vec4<f32>,
    shadow_origin: vec4<f32>,
    shadow_direction: vec4<f32>,
};

// Radiance and its small cross-pass state machine are kept separately from hit
// geometry. shadow_radiance.w is 0 for none, 1 for pending visibility, and 2
// for visible; sample_radiance.w marks a valid traced sample.
struct SurfaceCacheRadianceInfo {
    shadow_radiance: vec4<f32>,
    sample_radiance: vec4<f32>,
};

fn surface_cache_sh_patch_read(
    buffer: ptr<storage, array<u32>, read_write>,
    patch_index: u32
) -> SH_L1_RGB {
    let base_offset = patch_index * SURFACE_CACHE_SH_PATCH_SIZE_U32;
    var packed: SH_L1_RGB_Packed;
    for (var coefficient = 0u; coefficient < SURFACE_CACHE_SH_PATCH_SIZE_U32; coefficient = coefficient + 1u) {
        packed.data[coefficient] = (*buffer)[base_offset + coefficient];
    }
    return sh_l1_rgb_unpack(packed);
}

fn surface_cache_sh_patch_write(
    buffer: ptr<storage, array<u32>, read_write>,
    patch_index: u32,
    sh: SH_L1_RGB
) {
    let base_offset = patch_index * SURFACE_CACHE_SH_PATCH_SIZE_U32;
    let packed = sh_l1_rgb_pack(sh);
    for (var coefficient = 0u; coefficient < SURFACE_CACHE_SH_PATCH_SIZE_U32; coefficient = coefficient + 1u) {
        (*buffer)[base_offset + coefficient] = packed.data[coefficient];
    }
}

fn surface_cache_hemisphere_frame(normal: vec3<f32>) -> mat3x3<f32> {
    return orthonormalize(safe_normalize(normal));
}

fn surface_cache_world_to_hemisphere(direction: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    return transpose(surface_cache_hemisphere_frame(normal)) * direction;
}

fn surface_cache_rotate_sh_between_hemispheres(
    sh: SH_L1_RGB,
    source_normal: vec3<f32>,
    receiver_normal: vec3<f32>
) -> SH_L1_RGB {
    let source_to_world = surface_cache_hemisphere_frame(source_normal);
    let world_to_receiver = transpose(surface_cache_hemisphere_frame(receiver_normal));
    return sh_l1_rgb_rotate(sh, world_to_receiver * source_to_world);
}

fn surface_cache_evaluate_local_sh_irradiance(sh: SH_L1_RGB) -> vec3<f32> {
    return max(
        // orthonormalize() constructs a frame whose Z axis is the supplied
        // normal. All traced directions and stored SH use that convention.
        sh_l1_rgb_calculate_irradiance(sh, vec3<f32>(0.0, 0.0, 1.0)),
        vec3<f32>(0.0)
    );
}

fn surface_cache_full_resolution(params: SurfaceCacheParams) -> vec2<u32> {
    return vec2<u32>(u32(params.full_resolution_x), u32(params.full_resolution_y));
}

fn surface_cache_lod_value(position: vec3<f32>, camera_position: vec3<f32>, params: SurfaceCacheParams) -> f32 {
    let distance = length(position - camera_position);
    let base_extent = max(
        params.surface_cache_cell_size * SURFACE_CACHE_LOD_EXTENT_CELLS,
        params.surface_cache_cell_size
    );
    let maximum_lod = max(params.surface_cache_lod_count - 1.0, 0.0);
    return clamp(log2(max(distance / base_extent, 1.0)), 0.0, maximum_lod);
}

fn surface_cache_select_lod(position: vec3<f32>, camera_position: vec3<f32>, params: SurfaceCacheParams) -> u32 {
    // Admission chooses the nearest logarithmic level. Lookup blends adjacent
    // levels, so camera motion cannot create a hard level switch.
    return u32(floor(surface_cache_lod_value(position, camera_position, params) + 0.5));
}

fn surface_cache_lod_cell_size(lod: u32, params: SurfaceCacheParams) -> f32 {
    return params.surface_cache_cell_size * f32(1 << lod);
}

fn surface_cache_quantize_position(position: vec3<f32>, lod: u32, params: SurfaceCacheParams) -> vec3<i32> {
    return vec3<i32>(floor(position / surface_cache_lod_cell_size(lod, params) + vec3<f32>(0.0001)));
}

// Return the world-space center of a quantized cell constrained to the patch's
// surface plane. A quantized coordinate only identifies an axis-aligned world
// grid cell, so surface_position and surface_normal supply the plane needed to
// recover a surface-aligned center. The two axes tangent to the dominant normal
// component remain at their grid-cell centers; the dominant coordinate is
// solved from the plane equation. Choosing the dominant component also keeps
// the division well-conditioned for every valid normal.
fn surface_cache_world_cell_center(
    quantized_position: vec3<i32>,
    surface_position: vec3<f32>,
    surface_normal: vec3<f32>,
    lod: u32,
    params: SurfaceCacheParams
) -> vec3<f32> {
    let cell_size = surface_cache_lod_cell_size(lod, params);
    var cell_center = (vec3<f32>(quantized_position) + vec3<f32>(0.5)) * cell_size;
    let normal = safe_normalize(surface_normal);
    let absolute_normal = abs(normal);
    let dominant_axis = select(
        select(2u, 1u, absolute_normal.y >= absolute_normal.z),
        0u,
        absolute_normal.x >= max(absolute_normal.y, absolute_normal.z)
    );

    if (dominant_axis == 0u) {
        cell_center.x = surface_position.x - (
            normal.y * (cell_center.y - surface_position.y) +
            normal.z * (cell_center.z - surface_position.z)
        ) / normal.x;
    } else if (dominant_axis == 1u) {
        cell_center.y = surface_position.y - (
            normal.x * (cell_center.x - surface_position.x) +
            normal.z * (cell_center.z - surface_position.z)
        ) / normal.y;
    } else {
        cell_center.z = surface_position.z - (
            normal.x * (cell_center.x - surface_position.x) +
            normal.y * (cell_center.y - surface_position.y)
        ) / normal.z;
    }

    return cell_center;
}

fn surface_cache_quantize_normal(normal: vec3<f32>) -> vec2<i32> {
    let n = safe_normalize(normal);
    let normal_octant =
        select(0, 1, n.x >= 0.0) |
        select(0, 2, n.y >= 0.0) |
        select(0, 4, n.z >= 0.0);
    return vec2<i32>(normal_octant, 0);
}

fn surface_cache_bucket_start(position: vec3<i32>, normal: vec2<i32>, lod: u32, params: SurfaceCacheParams) -> u32 {
    let position_hash = (position.x * 73856093) ^ (position.y * 19349663) ^ (position.z * 83492791);
    let descriptor_hash = (normal.x * 50331653) ^ (i32(lod) * 25165843);
    let combined = bitcast<u32>(position_hash ^ descriptor_hash);
    let capacity = max(u32(params.total_patch_count), SURFACE_CACHE_BUCKET_SIZE);
    let bucket_count = max(capacity / SURFACE_CACHE_BUCKET_SIZE, 1u);
    return (combined % bucket_count) * SURFACE_CACHE_BUCKET_SIZE;
}

fn surface_cache_hash_fingerprint(position: vec3<i32>, normal: vec2<i32>, lod: u32) -> u32 {
    let position_hash = (position.x * 25165843) ^ (position.y * 50331653) ^ (position.z * 73856093);
    let normal_hash = (normal.x * 83492791) ^ (normal.y * 19349663);
    let value = bitcast<u32>(position_hash ^ normal_hash ^ (i32(lod) * 393241));
    return select(value, 1u, value == 0u);
}

fn surface_cache_make_grid_key(
    quantized_position: vec3<i32>,
    quantized_normal: vec2<i32>,
    lod: u32
) -> vec4<i32> {
    return vec4<i32>(quantized_position, i32(lod * 8u) + quantized_normal.x);
}

fn surface_cache_grid_key_lod(grid_key: vec4<i32>) -> u32 {
    return u32(max(grid_key.w, 0)) / 8u;
}

fn surface_cache_patch_descriptor_matches(
    patch_grid_key: vec4<i32>,
    quantized_position: vec3<i32>,
    quantized_normal: vec2<i32>,
    lod: u32
) -> bool {
    return all(patch_grid_key == surface_cache_make_grid_key(
        quantized_position,
        quantized_normal,
        lod
    ));
}

fn surface_cache_patch_rng(patch_index: u32, patch_fingerprint: u32) -> u32 {
    let rng = hash(patch_index ^ patch_fingerprint ^ 0x9e3779b9u);
    return random_seed(rng);
}
