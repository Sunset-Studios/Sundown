#include "gi/gi_common.wgsl"
#include "sh_common.wgsl"
#include "hashmap.wgsl"

const SURFACE_CACHE_MIN_QUERY_SAMPLES: f32 = 2.0;
const SURFACE_CACHE_MAX_RADIANCE: f32 = 10.0;
const SURFACE_CACHE_SH_PATCH_SIZE_U32: u32 = 6u;
const SURFACE_CACHE_NORMAL_BIN_COUNT: u32 = 7u;
const SURFACE_CACHE_NORMAL_DESCRIPTOR_COUNT: u32 = 343u;
const SURFACE_CACHE_CELL_EXPONENT_BIAS: i32 = 16;
const SURFACE_CACHE_MIN_CELL_EXPONENT: i32 = -16;
const SURFACE_CACHE_MAX_CELL_EXPONENT: i32 = 15;
const SURFACE_CACHE_LEVEL_BLEND_START: f32 = 0.2;
const SURFACE_CACHE_LEVEL_BLEND_END: f32 = 0.8;
const SURFACE_CACHE_LEVEL_CONFIDENCE_SAMPLES: f32 = 32.0;

struct SurfaceCacheParams {
    surface_cache_size: f32,
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
    rays_per_patch: f32,
    cache_pixel_footprint: f32,
    hash_search_count: f32,
    cache_normal_bias: f32,
    stable_update_interval: f32,
    stable_update_min_samples: f32,
    stable_update_variance_threshold: f32,
    padding0: f32,
};

struct SurfaceCacheCellLevels {
    fine_exponent: i32,
    coarse_exponent: i32,
    blend: f32,
};

struct SurfacePatch {
    position_frame: vec4<f32>,
    normal_cell_exponent: vec4<f32>,
    grid_key: vec4<i32>,
    metadata: vec4<f32>,
    history: vec4<f32>,
};

struct SurfacePatchReadOnly {
    position_frame: vec4<f32>,
    normal_cell_exponent: vec4<f32>,
    grid_key: vec4<i32>,
    metadata: vec4<f32>,
    history: vec4<f32>,
};

struct SurfaceCacheCounters {
    active_patch_count: atomic<u32>,
    update_patch_count: atomic<u32>,
    padding1: atomic<u32>,
    padding2: atomic<u32>,
};

struct SurfaceCacheCountersReadOnly {
    active_patch_count: u32,
    update_patch_count: u32,
    padding1: u32,
    padding2: u32,
};

// Exact compact primary-hit record. Attribute reconstruction stays in the
// material pass, keeping traversal output to three naturally aligned vectors.
struct SurfaceCacheHitInfo {
    ray_direction_sampling_weight: vec4<f32>,
    hit_identity: vec4<u32>,
    hit_barycentrics_t: vec4<f32>,
};

// Shading owns radiance and the shadow ray it generates. shadow_radiance.w is
// 0 for none, 1 for pending visibility, and 2 for visible;
// sample_radiance.w marks a valid traced sample.
struct SurfaceCacheRadianceInfo {
    shadow_radiance: vec4<f32>,
    sample_radiance: vec4<f32>,
    shadow_origin: vec4<f32>,
    shadow_direction: vec4<f32>,
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

fn surface_cache_rays_per_patch(params: SurfaceCacheParams) -> u32 {
    return max(u32(params.rays_per_patch), 1u);
}

// Convert the configured screen-space feature size into a continuous
// world-space exponent. Keeping the fractional component lets adjacent cache
// levels overlap instead of replacing one another at a hard power-of-two line.
fn surface_cache_cell_exponent_value(
    position: vec3<f32>,
    params: SurfaceCacheParams
) -> f32 {
    let view = view_buffer[u32(frame_info.view_index)];
    let projection_y_scale = max(abs(view.projection_matrix[1][1]), 1e-6);
    let view_depth = abs((view.view_matrix * vec4<f32>(position, 1.0)).z);
    let is_perspective = abs(view.projection_matrix[3][3]) < 0.5;
    // projection_y_scale is cot(vertical_fov / 2), so the perspective branch
    // is the article's d * tan(vertical_fov / 2) half-height calculation.
    let half_view_height = select(
        1.0 / projection_y_scale,
        view_depth / projection_y_scale,
        is_perspective
    );
    let world_space_footprint = max(params.cache_pixel_footprint, 1.0) * (
        (half_view_height * 2.0) / max(params.full_resolution_y, 1.0)
    );
    let minimum_cell_size = exp2(f32(SURFACE_CACHE_MIN_CELL_EXPONENT));
    let exponent_value = log2(max(
        world_space_footprint,
        minimum_cell_size
    ));
    return clamp(
        exponent_value,
        f32(SURFACE_CACHE_MIN_CELL_EXPONENT),
        f32(SURFACE_CACHE_MAX_CELL_EXPONENT)
    );
}

fn surface_cache_cell_levels(
    position: vec3<f32>,
    params: SurfaceCacheParams
) -> SurfaceCacheCellLevels {
    let exponent_value = surface_cache_cell_exponent_value(position, params);
    let fine_exponent = i32(floor(exponent_value));
    let coarse_exponent = min(
        fine_exponent + 1,
        SURFACE_CACHE_MAX_CELL_EXPONENT
    );
    let level_fraction = fract(exponent_value);
    let blend = select(
        smoothstep(
            SURFACE_CACHE_LEVEL_BLEND_START,
            SURFACE_CACHE_LEVEL_BLEND_END,
            level_fraction
        ),
        0.0,
        fine_exponent == coarse_exponent
    );
    return SurfaceCacheCellLevels(fine_exponent, coarse_exponent, blend);
}

fn surface_cache_cell_exponent(position: vec3<f32>, params: SurfaceCacheParams) -> i32 {
    return surface_cache_cell_levels(position, params).fine_exponent;
}

fn surface_cache_cell_size(cell_exponent: i32) -> f32 {
    return exp2(f32(cell_exponent));
}

fn surface_cache_encode_cell_exponent(cell_exponent: i32) -> u32 {
    return u32(cell_exponent + SURFACE_CACHE_CELL_EXPONENT_BIAS);
}

fn surface_cache_decode_cell_exponent(encoded_exponent: u32) -> i32 {
    return i32(encoded_exponent) - SURFACE_CACHE_CELL_EXPONENT_BIAS;
}

fn surface_cache_descriptor_offset(
    normal: vec3<f32>,
    cell_size: f32,
    params: SurfaceCacheParams
) -> vec3<f32> {
    let normalized = safe_normalize(normal);
    let absolute_normal = abs(normalized);
    let dominant_axis = select(
        select(2u, 1u, absolute_normal.y >= absolute_normal.z),
        0u,
        absolute_normal.x >= max(absolute_normal.y, absolute_normal.z)
    );
    let bias = max(params.cache_normal_bias, 0.0) * cell_size;

    if (dominant_axis == 0u) {
        return vec3<f32>(select(-bias, bias, normalized.x >= 0.0), 0.0, 0.0);
    }
    if (dominant_axis == 1u) {
        return vec3<f32>(0.0, select(-bias, bias, normalized.y >= 0.0), 0.0);
    }
    return vec3<f32>(0.0, 0.0, select(-bias, bias, normalized.z >= 0.0));
}

fn surface_cache_quantize_position(
    position: vec3<f32>,
    normal: vec3<f32>,
    cell_exponent: i32,
    params: SurfaceCacheParams
) -> vec3<i32> {
    let cell_size = surface_cache_cell_size(cell_exponent);
    let descriptor_position = position + surface_cache_descriptor_offset(
        normal,
        cell_size,
        params
    );
    return vec3<i32>(floor(descriptor_position / cell_size));
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
    cell_exponent: i32
) -> vec3<f32> {
    let cell_size = surface_cache_cell_size(cell_exponent);
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

fn surface_cache_quantize_normal(normal: vec3<f32>) -> vec3<i32> {
    let normalized = clamp(
        safe_normalize(normal),
        vec3<f32>(-1.0),
        vec3<f32>(1.0)
    );
    return vec3<i32>(floor(normalized * 3.0));
}

fn surface_cache_hash_key(
    quantized_position: vec3<i32>,
    quantized_normal: vec3<i32>,
    cell_exponent: i32
) -> HashMapKey {
    // The encoded exponent is the exact identity of the automatically selected
    // power-of-two cell size, including sizes smaller than one world unit.
    let encoded_exponent = surface_cache_encode_cell_exponent(cell_exponent);

    var index_hash = hashmap_pcg(bitcast<u32>(quantized_normal.z));
    index_hash = hashmap_pcg_combine(bitcast<u32>(quantized_normal.y), index_hash);
    index_hash = hashmap_pcg_combine(bitcast<u32>(quantized_normal.x), index_hash);
    index_hash = hashmap_pcg_combine(bitcast<u32>(quantized_position.z), index_hash);
    index_hash = hashmap_pcg_combine(bitcast<u32>(quantized_position.y), index_hash);
    index_hash = hashmap_pcg_combine(bitcast<u32>(quantized_position.x), index_hash);
    index_hash = hashmap_pcg_combine(encoded_exponent, index_hash);

    var checksum = hashmap_xxhash32(bitcast<u32>(quantized_normal.z));
    checksum = hashmap_xxhash32_combine(bitcast<u32>(quantized_normal.y), checksum);
    checksum = hashmap_xxhash32_combine(bitcast<u32>(quantized_normal.x), checksum);
    checksum = hashmap_xxhash32_combine(bitcast<u32>(quantized_position.z), checksum);
    checksum = hashmap_xxhash32_combine(bitcast<u32>(quantized_position.y), checksum);
    checksum = hashmap_xxhash32_combine(bitcast<u32>(quantized_position.x), checksum);
    checksum = hashmap_xxhash32_combine(encoded_exponent, checksum);

    return HashMapKey(index_hash, hashmap_sanitize_checksum(checksum));
}

fn surface_cache_make_grid_key(
    quantized_position: vec3<i32>,
    quantized_normal: vec3<i32>,
    cell_exponent: i32
) -> vec4<i32> {
    let normal_descriptor =
        u32(quantized_normal.x + 3) +
        SURFACE_CACHE_NORMAL_BIN_COUNT * u32(quantized_normal.y + 3) +
        SURFACE_CACHE_NORMAL_BIN_COUNT * SURFACE_CACHE_NORMAL_BIN_COUNT *
            u32(quantized_normal.z + 3);
    return vec4<i32>(
        quantized_position,
        i32(
            surface_cache_encode_cell_exponent(cell_exponent) *
            SURFACE_CACHE_NORMAL_DESCRIPTOR_COUNT + normal_descriptor
        )
    );
}

fn surface_cache_grid_key_cell_exponent(grid_key: vec4<i32>) -> i32 {
    let encoded_exponent =
        u32(max(grid_key.w, 0)) / SURFACE_CACHE_NORMAL_DESCRIPTOR_COUNT;
    return surface_cache_decode_cell_exponent(encoded_exponent);
}

fn surface_cache_patch_descriptor_matches(
    patch_grid_key: vec4<i32>,
    quantized_position: vec3<i32>,
    quantized_normal: vec3<i32>,
    cell_exponent: i32
) -> bool {
    return all(patch_grid_key == surface_cache_make_grid_key(
        quantized_position,
        quantized_normal,
        cell_exponent
    ));
}

fn surface_cache_hash_search_count(params: SurfaceCacheParams) -> u32 {
    return hashmap_search_count(
        max(u32(params.hash_search_count), 1u),
        max(u32(params.total_patch_count), 1u)
    );
}

fn surface_cache_patch_rng(patch_index: u32, grid_key: vec4<i32>) -> u32 {
    let descriptor_seed =
        bitcast<u32>(grid_key.x) ^
        hash(bitcast<u32>(grid_key.y)) ^
        hash(bitcast<u32>(grid_key.z)) ^
        hash(bitcast<u32>(grid_key.w));
    let rng = hash(patch_index ^ descriptor_seed ^ 0x9e3779b9u);
    return random_seed(rng);
}
