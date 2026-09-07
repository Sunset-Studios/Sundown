#include "gi/gi_common.wgsl"
#include "sh_common.wgsl"
#include "hashmap.wgsl"

const SURFACE_CACHE_MIN_QUERY_SAMPLES: f32 = 4.0;
const SURFACE_CACHE_MAX_RADIANCE: f32 = 10.0;
const SURFACE_CACHE_SH_PATCH_SIZE_U32: u32 = 6u;
// Keep every octahedral bin narrower than the lookup normal threshold. A
// coarse directional key can merge surfaces whose normals reconstruction must
// reject, leaving the patch's arbitrary feedback winner to decide whether the
// cache is visible on a given frame. Changing this static layout alters every
// cache key and therefore requires a cache reset.
const SURFACE_CACHE_DIRECTIONAL_BIN_RESOLUTION: u32 = 8u;
const SURFACE_CACHE_DIRECTIONAL_BIN_COUNT: u32 =
    SURFACE_CACHE_DIRECTIONAL_BIN_RESOLUTION *
    SURFACE_CACHE_DIRECTIONAL_BIN_RESOLUTION;
const SURFACE_CACHE_CELL_EXPONENT_BIAS: i32 = 16;
const SURFACE_CACHE_MIN_CELL_EXPONENT: i32 = -32;
const SURFACE_CACHE_MAX_CELL_EXPONENT: i32 = 31;
const SURFACE_CACHE_LEVEL_BLEND_START: f32 = 0.0;
const SURFACE_CACHE_LEVEL_BLEND_END: f32 = 1.0;
const SURFACE_CACHE_EMISSIVE_LUMA_SOFT_CAP: f32 = 2.0;
const SURFACE_CACHE_EMISSIVE_OVERFLOW_SCALE: f32 = 0.1;
const SURFACE_CACHE_COSINE_PROBABILITY: f32 = 0.5;
const SURFACE_CACHE_BACKFACE_HIT: u32 = INVALID_IDX - 1u;

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
    rays_per_patch: f32,
    cache_entry_lifetime: f32,
    cache_pixel_footprint: f32,
    hash_search_count: f32,
    cache_normal_bias: f32,
    maximum_ray_count_per_frame: f32,
    native_promotion_start_confidence: f32,
    native_promotion_end_confidence: f32,
    padding0: f32,
    padding1: f32,
    padding2: f32,
};

struct SurfaceCacheCellLevels {
    fine_exponent: i32,
    coarse_exponent: i32,
    blend: f32,
    exponent_value: f32,
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
    schedule_offset: atomic<u32>,
    available_update_patch_count: atomic<u32>,
    feedback_miss_count: atomic<u32>,
    scheduled_rays_per_patch: atomic<u32>,
};

struct SurfaceCacheCountersReadOnly {
    active_patch_count: u32,
    update_patch_count: u32,
    schedule_offset: u32,
    available_update_patch_count: u32,
    feedback_miss_count: u32,
    scheduled_rays_per_patch: u32,
};

struct SurfaceCacheHitInfo {
    ray_direction_sampling_weight: vec4<f32>,
    hit_identity: vec4<u32>,
    hit_barycentrics_t: vec4<f32>,
};

struct SurfaceCacheRadianceInfo {
    shadow_radiance: vec4<f32>, // .w = 0 for none, 1 for pending visiblity, 2 for visible
    sample_radiance: vec4<f32>, // .w = valid sample
    shadow_origin: vec4<f32>,
    shadow_direction: vec4<f32>,
};

struct SurfaceCacheRaySample {
    direction: vec3<f32>,
    sampling_weight: f32,
};

struct SurfaceCacheRayWork {
    active_index: u32,
    ray_index_in_patch: u32,
    data_index: u32,
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

fn surface_cache_sh_patch_read_only(
    buffer: ptr<storage, array<u32>, read>,
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

fn surface_cache_full_resolution(params: SurfaceCacheParams) -> vec2<u32> {
    return vec2<u32>(u32(params.full_resolution_x), u32(params.full_resolution_y));
}

fn surface_cache_ray_count(counters: SurfaceCacheCountersReadOnly) -> u32 {
    return counters.update_patch_count * max(counters.scheduled_rays_per_patch, 1u);
}

fn surface_cache_total_ray_count(
    counters: SurfaceCacheCountersReadOnly
) -> u32 {
    return surface_cache_ray_count(counters);
}

fn surface_cache_schedule_index(
    scheduled_index: u32,
    counters: SurfaceCacheCountersReadOnly
) -> u32 {
    let available_patch_count = max(
        counters.available_update_patch_count,
        1u
    );
    if (counters.update_patch_count >= available_patch_count) {
        return scheduled_index;
    }

    let scheduled_patch_count = max(counters.update_patch_count, 1u);
    let base_step = available_patch_count / scheduled_patch_count;
    let remainder = available_patch_count % scheduled_patch_count;
    let remainder_step = u32(floor(
        f32(scheduled_index) *
            (f32(remainder) / f32(scheduled_patch_count))
    ));
    let distributed_index =
        scheduled_index * base_step + remainder_step;
    let rotated_index = distributed_index + counters.schedule_offset;
    return select(
        rotated_index,
        rotated_index - available_patch_count,
        rotated_index >= available_patch_count
    );
}

fn surface_cache_ray_work(
    work_index: u32,
    counters: SurfaceCacheCountersReadOnly
) -> SurfaceCacheRayWork {
    let rays_per_patch = max(counters.scheduled_rays_per_patch, 1u);
    return SurfaceCacheRayWork(
        work_index / rays_per_patch,
        work_index % rays_per_patch,
        work_index
    );
}

fn surface_cache_commit_accumulation(
    cache_buffer: ptr<storage, array<SurfacePatch>, read_write>,
    sh_buffer: ptr<storage, array<u32>, read_write>,
    params: SurfaceCacheParams,
    patch_index: u32,
    sample_sh_sum: SH_L1_RGB,
    luminance_sum: f32,
    luminance_squared_sum: f32,
    valid_sample_count: f32
) {
    if (valid_sample_count <= 0.0) {
        return;
    }

    let inverse_sample_count = 1.0 / valid_sample_count;
    let sample_sh = sh_l1_rgb_multiply_scalar(
        sample_sh_sum,
        inverse_sample_count
    );
    let sample_luminance = luminance_sum * inverse_sample_count;
    let sample_luminance_squared =
        luminance_squared_sum * inverse_sample_count;

    let history = (*cache_buffer)[patch_index].history;
    let previous_sample_count = history.x;
    let maximum_history_samples = max(params.max_history_samples, 1.0);
    let effective_previous_sample_count = min(
        previous_sample_count,
        maximum_history_samples
    );
    let next_sample_count =
        effective_previous_sample_count + valid_sample_count;
    let next_sequence = f32(
        (u32(history.y) + u32(valid_sample_count)) & 4095u
    );
    let running_alpha = min(
        valid_sample_count / max(next_sample_count, 1.0),
        1.0
    );

    let previous_variance = max(
        history.w - history.z * history.z,
        0.0
    );
    let sample_variance = max(
        sample_luminance_squared - sample_luminance * sample_luminance,
        0.0
    );
    let mean_variance =
        previous_variance / max(effective_previous_sample_count, 1.0) +
        sample_variance / max(valid_sample_count, 1.0);
    let change_threshold = max(3.0 * sqrt(mean_variance), 0.01);
    let history_is_mature =
        previous_sample_count >= maximum_history_samples;
    let lighting_changed = history_is_mature &&
        abs(sample_luminance - history.z) > change_threshold;
    let response_alpha = 1.0 - clamp(
        params.history_hysteresis,
        0.0,
        0.999
    );
    let blend_alpha = max(
        running_alpha,
        select(0.0, response_alpha, lighting_changed)
    );

    var result = sample_sh;
    if (previous_sample_count > 0.0) {
        result = sh_l1_rgb_lerp(
            surface_cache_sh_patch_read(sh_buffer, patch_index),
            sample_sh,
            blend_alpha
        );
    }
    surface_cache_sh_patch_write(sh_buffer, patch_index, result);

    let first_moment = mix(history.z, sample_luminance, blend_alpha);
    let second_moment = mix(
        history.w,
        sample_luminance_squared,
        blend_alpha
    );
    let responsive_sample_count =
        valid_sample_count / max(blend_alpha, 1e-6);
    let stored_sample_count = select(
        next_sample_count,
        min(next_sample_count, responsive_sample_count),
        lighting_changed
    );
    (*cache_buffer)[patch_index].history = vec4<f32>(
        min(stored_sample_count, maximum_history_samples),
        next_sequence,
        first_moment,
        second_moment
    );
    (*cache_buffer)[patch_index].metadata.x = params.frame_index;
    let footprint_history_increment = min(
        valid_sample_count,
        params.rays_per_patch
    );
    (*cache_buffer)[patch_index].metadata.w = 
        (*cache_buffer)[patch_index].metadata.w + footprint_history_increment;
}

// Convert the configured screen-space feature size into a continuous
// world-space exponent. Keeping the fractional component lets adjacent cache
// levels overlap instead of replacing one another at a hard power-of-two line.
fn surface_cache_cell_exponent_value(
    position: vec3<f32>,
    params: SurfaceCacheParams
) -> f32 {
    let view = view_buffer[u32(frame_info.view_index)];
    let projection_y_scale = abs(view.projection_matrix[1][1]);
    let is_perspective = abs(view.projection_matrix[3][3]) < 0.5;
    let view_depth = abs((view.view_matrix * vec4<f32>(position, 1.0)).z);
    // projection_y_scale is cot(vertical_fov / 2), so the perspective branch
    // is the article's d * tan(vertical_fov / 2) half-height calculation.
    let half_view_height = select(
        1.0 / projection_y_scale,
        view_depth / projection_y_scale,
        is_perspective
    );
    let effective_pixel_footprint = max(params.cache_pixel_footprint, 1.0);
    let world_space_footprint = effective_pixel_footprint * (
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
    return SurfaceCacheCellLevels(
        fine_exponent,
        coarse_exponent,
        blend,
        exponent_value
    );
}

fn surface_cache_history_cell_levels_from_base(
    base_exponent_value: f32,
    params: SurfaceCacheParams
) -> SurfaceCacheCellLevels {
    let exponent_value = clamp(
        base_exponent_value,
        f32(SURFACE_CACHE_MIN_CELL_EXPONENT),
        f32(SURFACE_CACHE_MAX_CELL_EXPONENT)
    );
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
    return SurfaceCacheCellLevels(
        fine_exponent,
        coarse_exponent,
        blend,
        exponent_value
    );
}

fn surface_cache_history_cell_levels(
    position: vec3<f32>,
    params: SurfaceCacheParams
) -> SurfaceCacheCellLevels {
    return surface_cache_history_cell_levels_from_base(
        surface_cache_cell_exponent_value(position, params),
        params
    );
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
    let absolute_normal = abs(normal);
    let dominant_axis = select(
        select(2u, 1u, absolute_normal.y >= absolute_normal.z),
        0u,
        absolute_normal.x >= max(absolute_normal.y, absolute_normal.z)
    );
    let bias = max(params.cache_normal_bias, 0.0) * cell_size;

    if (dominant_axis == 0u) {
        return vec3<f32>(select(-bias, bias, normal.x >= 0.0), 0.0, 0.0);
    }
    if (dominant_axis == 1u) {
        return vec3<f32>(0.0, select(-bias, bias, normal.y >= 0.0), 0.0);
    }
    return vec3<f32>(0.0, 0.0, select(-bias, bias, normal.z >= 0.0));
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

fn surface_cache_octahedral_direction(normal: vec3<f32>) -> vec2<f32> {
    let normalized = safe_normalize(normal);
    let projected = normalized / max(
        abs(normalized.x) + abs(normalized.y) + abs(normalized.z),
        1e-6
    );
    var octahedral = projected.xy;
    if (projected.z < 0.0) {
        let signs = vec2<f32>(
            select(-1.0, 1.0, projected.x >= 0.0),
            select(-1.0, 1.0, projected.y >= 0.0)
        );
        octahedral = (vec2<f32>(1.0) - abs(projected.yx)) * signs;
    }
    return clamp(
        octahedral * 0.5 + vec2<f32>(0.5),
        vec2<f32>(0.0),
        vec2<f32>(1.0)
    );
}

fn surface_cache_directional_bin(normal: vec3<f32>) -> u32 {
    let encoded = encode_octahedral(normal);
    let coordinate = min(
        vec2<u32>(
            encoded * f32(SURFACE_CACHE_DIRECTIONAL_BIN_RESOLUTION)
        ),
        vec2<u32>(SURFACE_CACHE_DIRECTIONAL_BIN_RESOLUTION - 1u)
    );
    return coordinate.x +
        coordinate.y * SURFACE_CACHE_DIRECTIONAL_BIN_RESOLUTION;
}

fn surface_cache_hash_key(
    quantized_position: vec3<i32>,
    directional_bin: u32,
    cell_exponent: i32
) -> HashMapKey {
    // The encoded exponent is the exact identity of the automatically selected
    // power-of-two cell size, including sizes smaller than one world unit.
    let encoded_exponent = surface_cache_encode_cell_exponent(cell_exponent);

    var index_hash = hashmap_pcg(directional_bin);
    index_hash = hashmap_pcg_combine(bitcast<u32>(quantized_position.z), index_hash);
    index_hash = hashmap_pcg_combine(bitcast<u32>(quantized_position.y), index_hash);
    index_hash = hashmap_pcg_combine(bitcast<u32>(quantized_position.x), index_hash);
    index_hash = hashmap_pcg_combine(encoded_exponent, index_hash);

    var checksum = hashmap_xxhash32(directional_bin);
    checksum = hashmap_xxhash32_combine(bitcast<u32>(quantized_position.z), checksum);
    checksum = hashmap_xxhash32_combine(bitcast<u32>(quantized_position.y), checksum);
    checksum = hashmap_xxhash32_combine(bitcast<u32>(quantized_position.x), checksum);
    checksum = hashmap_xxhash32_combine(encoded_exponent, checksum);

    return HashMapKey(index_hash, hashmap_sanitize_checksum(checksum));
}

fn surface_cache_make_grid_key(
    quantized_position: vec3<i32>,
    directional_bin: u32,
    cell_exponent: i32
) -> vec4<i32> {
    return vec4<i32>(
        quantized_position,
        i32(
            directional_bin + surface_cache_encode_cell_exponent(cell_exponent) *
            SURFACE_CACHE_DIRECTIONAL_BIN_COUNT
        )
    );
}

fn surface_cache_grid_key_cell_exponent(grid_key: vec4<i32>) -> i32 {
    let encoded_exponent = u32(grid_key.w) / SURFACE_CACHE_DIRECTIONAL_BIN_COUNT;
    return surface_cache_decode_cell_exponent(encoded_exponent);
}

fn surface_cache_patch_descriptor_matches(
    patch_grid_key: vec4<i32>,
    quantized_position: vec3<i32>,
    directional_bin: u32,
    cell_exponent: i32
) -> bool {
    return all(patch_grid_key == surface_cache_make_grid_key(
        quantized_position,
        directional_bin,
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

fn sample_mis_hemisphere_surface_cache(
    normal: vec3<f32>,
    r1: f32,
    r2: f32,
    cosine_probability: f32
) -> SurfaceCacheRaySample {
    let cosine_mix = clamp(cosine_probability, 0.0, 0.999);
    let sample_cosine = r1 < cosine_mix;
    let remapped_r1 = select(
        (r1 - cosine_mix) / max(1.0 - cosine_mix, 1e-6),
        r1 / max(cosine_mix, 1e-6),
        sample_cosine
    );
    let cos_theta = select(r2, sqrt(r2), sample_cosine);
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let phi = 2.0 * PI * remapped_r1;
    let direction = surface_cache_hemisphere_frame(normal) * vec3<f32>(
        sin_theta * cos(phi),
        sin_theta * sin(phi),
        cos_theta
    );
    let uniform_pdf = 0.5 / PI;
    let cosine_pdf = cos_theta / PI;
    let mixture_pdf = mix(uniform_pdf, cosine_pdf, cosine_mix);
    return SurfaceCacheRaySample(direction, 1.0 / max(mixture_pdf, 1e-6));
}

fn generate_ray_sample(
    seed: u32,
    normal: vec3<f32>,
    sample_index: u32,
    cosine_probability: f32
) -> SurfaceCacheRaySample {
    // A Cranley-Patterson rotated R2 sequence drives a uniform/cosine mixture.
    // The uniform component preserves directional SH coverage while the cosine
    // component concentrates work where diffuse irradiance is most
    // sensitive. The inverse mixture PDF keeps the L1 estimator unbiased.
    var rng = random_seed(seed);
    let rotation_u = rand_float(rng);
    rng = random_seed(rng);
    let rotation_v = rand_float(rng);

    let sequence_value = f32(sample_index);
    let r1 = fract(rotation_u + sequence_value * 0.7548776662466927);
    let r2 = fract(rotation_v + sequence_value * 0.5698402909980532);

    return sample_mis_hemisphere_surface_cache(
        normal,
        r1,
        r2,
        cosine_probability
    );
}

fn stabilize_surface_cache_emissive_radiance(
    raw_emissive_radiance: vec3<f32>
) -> vec3<f32> {
    let clamped_radiance = safe_clamp_vec3_max(
        raw_emissive_radiance,
        SURFACE_CACHE_MAX_RADIANCE
    );
    let emissive_luminance = max(luminance(clamped_radiance), 1e-6);
    let compressed_luminance = select(
        emissive_luminance,
        SURFACE_CACHE_EMISSIVE_LUMA_SOFT_CAP +
            (emissive_luminance - SURFACE_CACHE_EMISSIVE_LUMA_SOFT_CAP) *
            SURFACE_CACHE_EMISSIVE_OVERFLOW_SCALE,
        emissive_luminance > SURFACE_CACHE_EMISSIVE_LUMA_SOFT_CAP
    );
    return clamped_radiance * (compressed_luminance / emissive_luminance);
}
