#include "gi/gi_common.wgsl"
#include "sh_common.wgsl"

const SURFACE_CACHE_BUCKET_SIZE: u32 = 8u;
const SURFACE_CACHE_LOD_EXTENT_CELLS: f32 = 128.0;
const SURFACE_CACHE_NORMAL_QUANTIZATION: f32 = 1.0;
const SURFACE_CACHE_PATCH_EMPTY: u32 = 0u;
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
    normal_unused: vec4<f32>,
    albedo_roughness: vec4<f32>,
    material_props: vec4<f32>,
    history: vec4<f32>,
    fingerprint: atomic<u32>,
    update_frame: atomic<u32>,
    padding1: u32,
    padding2: u32,
};

struct SurfacePatchReadOnly {
    position_frame: vec4<f32>,
    normal_unused: vec4<f32>,
    albedo_roughness: vec4<f32>,
    material_props: vec4<f32>,
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
        sh_l1_rgb_calculate_irradiance(sh, vec3<f32>(0.0, 1.0, 0.0)),
        vec3<f32>(0.0)
    );
}

fn surface_cache_full_resolution(params: SurfaceCacheParams) -> vec2<u32> {
    return vec2<u32>(u32(params.full_resolution_x), u32(params.full_resolution_y));
}

fn surface_cache_select_lod(position: vec3<f32>, camera_position: vec3<f32>, params: SurfaceCacheParams) -> u32 {
    let delta = abs(position - camera_position);
    let square_distance = max(delta.x, max(delta.y, delta.z));
    let base_extent = max(
        params.surface_cache_cell_size * SURFACE_CACHE_LOD_EXTENT_CELLS,
        params.surface_cache_cell_size
    );
    let raw_lod = ceil(log2(max(square_distance / base_extent, 0.001)));
    return u32(clamp(i32(raw_lod), 0, i32(params.surface_cache_lod_count) - 1));
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
    let inv_l1 = 1.0 / max(abs(n.x) + abs(n.y) + abs(n.z), 1e-6);
    var oct = n.xy * inv_l1;
    let sign_oct = vec2<f32>(select(-1.0, 1.0, oct.x >= 0.0), select(-1.0, 1.0, oct.y >= 0.0));
    oct = select(oct, (vec2<f32>(1.0) - abs(oct.yx)) * sign_oct, n.z < 0.0);
    return vec2<i32>(floor(oct * SURFACE_CACHE_NORMAL_QUANTIZATION + vec2<f32>(0.5)));
}

fn surface_cache_bucket_start(position: vec3<i32>, normal: vec2<i32>, lod: u32, params: SurfaceCacheParams) -> u32 {
    let position_hash = (position.x * 73856093) ^ (position.y * 19349663) ^ (position.z * 83492791);
    let normal_hash = (normal.x * 50331653) ^ (normal.y * 25165843);
    let combined = bitcast<u32>(position_hash ^ normal_hash);
    let patches_per_lod = max(u32(params.surface_cache_size), SURFACE_CACHE_BUCKET_SIZE);
    let bucket_count = max(patches_per_lod / SURFACE_CACHE_BUCKET_SIZE, 1u);
    let lod_start = lod * patches_per_lod;
    return lod_start + (combined % bucket_count) * SURFACE_CACHE_BUCKET_SIZE;
}

fn surface_cache_hash_fingerprint(position: vec3<i32>, normal: vec2<i32>, lod: u32) -> u32 {
    let position_hash = (position.x * 25165843) ^ (position.y * 50331653) ^ (position.z * 73856093);
    let normal_hash = (normal.x * 83492791) ^ (normal.y * 19349663);
    let value = bitcast<u32>(position_hash ^ normal_hash ^ (i32(lod) * 393241));
    return select(value, 1u, value == 0u);
}

fn surface_cache_patch_descriptor_matches(
    patch_position: vec3<f32>,
    patch_normal: vec3<f32>,
    quantized_position: vec3<i32>,
    quantized_normal: vec2<i32>,
    lod: u32,
    params: SurfaceCacheParams
) -> bool {
    return
        all(surface_cache_quantize_position(patch_position, lod, params) == quantized_position) &&
        all(surface_cache_quantize_normal(patch_normal) == quantized_normal);
}

fn surface_cache_patch_rng(patch_index: u32, patch_fingerprint: u32) -> u32 {
    let rng = hash(patch_index ^ patch_fingerprint ^ 0x9e3779b9u);
    return random_seed(rng);
}
