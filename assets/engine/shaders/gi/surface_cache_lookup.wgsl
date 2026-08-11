fn surface_cache_find_patch(
    quantized_position: vec3<i32>,
    directional_bin: u32,
    cell_exponent: i32
) -> i32 {
    let key = surface_cache_hash_key(
        quantized_position,
        directional_bin,
        cell_exponent
    );
    let capacity = max(u32(surface_cache_params.total_patch_count), 1u);
    let patch_index = hashmap_find(
        &surface_cache_hashmap,
        key,
        capacity,
        surface_cache_hash_search_count(surface_cache_params)
    );
    if (
        patch_index != HASHMAP_INVALID_INDEX &&
        surface_cache_patch_descriptor_matches(
            surface_cache[patch_index].grid_key,
            quantized_position,
            directional_bin,
            cell_exponent
        )
    ) {
        return i32(patch_index);
    }
    return -1;
}

fn surface_cache_level_history(
    position: vec3<f32>,
    normal: vec3<f32>,
    cell_exponent: i32
) -> f32 {
    let receiver_normal = safe_normalize(normal);
    let patch_index_i = surface_cache_find_patch(
        surface_cache_quantize_position(
            position,
            receiver_normal,
            cell_exponent,
            surface_cache_params
        ),
        surface_cache_directional_bin(receiver_normal),
        cell_exponent
    );
    if (patch_index_i < 0) {
        return 0.0;
    }
    let surface_patch = surface_cache[u32(patch_index_i)];
    if (dot(
        receiver_normal,
        safe_normalize(surface_patch.normal_cell_exponent.xyz)
    ) < SURFACE_CACHE_LOOKUP_NORMAL_THRESHOLD) {
        return 0.0;
    }
    let position_delta = surface_patch.position_frame.xyz - position;
    let plane_distance = max(
        abs(dot(position_delta, receiver_normal)),
        abs(dot(position_delta, safe_normalize(
            surface_patch.normal_cell_exponent.xyz
        )))
    );
    if (
        plane_distance > max(
            surface_cache_cell_size(cell_exponent) *
                SURFACE_CACHE_LOOKUP_PLANE_LIMIT_SCALE,
            0.002
        )
    ) {
        return 0.0;
    }
    // Feedback snapshots history before this frame's accumulation. Using that
    // value keeps newly allocated cells on the coarse footprint through the
    // resolve that first consumes them, even after their bootstrap rays land.
    return select(
        surface_patch.metadata.w,
        surface_patch.metadata.y,
        u32(surface_patch.metadata.z) == u32(surface_cache_params.frame_index)
    );
}

fn surface_cache_native_history(
    position: vec3<f32>,
    normal: vec3<f32>,
    levels: SurfaceCacheCellLevels
) -> f32 {
    let fine_history = surface_cache_level_history(
        position,
        normal,
        levels.fine_exponent
    );
    if (levels.coarse_exponent == levels.fine_exponent) {
        return fine_history;
    }
    return min(
        fine_history,
        surface_cache_level_history(
            position,
            normal,
            levels.coarse_exponent
        )
    );
}

fn surface_cache_corner_descriptor(
    position: vec3<f32>,
    normal: vec3<f32>,
    tangent_cell: vec2<i32>,
    dominant_axis: u32,
    cell_size: f32,
    params: SurfaceCacheParams
) -> vec3<i32> {
    let descriptor_offset = surface_cache_descriptor_offset(
        normal,
        cell_size,
        params
    );
    let descriptor_tangent_center =
        (vec2<f32>(tangent_cell) + vec2<f32>(0.5)) * cell_size;
    if (dominant_axis == 0u) {
        let tangent_center = descriptor_tangent_center - descriptor_offset.yz;
        let dominant_position = position.x - (
            normal.y * (tangent_center.x - position.y) +
            normal.z * (tangent_center.y - position.z)
        ) / normal.x;
        return vec3<i32>(
            i32(floor((dominant_position + descriptor_offset.x) / cell_size)),
            tangent_cell.x,
            tangent_cell.y
        );
    }
    if (dominant_axis == 1u) {
        let tangent_center = descriptor_tangent_center - descriptor_offset.xz;
        let dominant_position = position.y - (
            normal.x * (tangent_center.x - position.x) +
            normal.z * (tangent_center.y - position.z)
        ) / normal.y;
        return vec3<i32>(
            tangent_cell.x,
            i32(floor((dominant_position + descriptor_offset.y) / cell_size)),
            tangent_cell.y
        );
    }

    let tangent_center = descriptor_tangent_center - descriptor_offset.xy;
    let dominant_position = position.z - (
        normal.x * (tangent_center.x - position.x) +
        normal.y * (tangent_center.y - position.y)
    ) / normal.z;
    return vec3<i32>(
        tangent_cell.x,
        tangent_cell.y,
        i32(floor((dominant_position + descriptor_offset.z) / cell_size))
    );
}

struct SurfaceCacheTapSample {
    irradiance: vec3<f32>,
    sample_count: f32,
    geometry_weight: f32,
};

struct SurfaceCacheLevelSample {
    value: vec4<f32>,
    confidence: f32,
};

const SURFACE_CACHE_LOOKUP_NORMAL_THRESHOLD: f32 = 0.82;
const SURFACE_CACHE_LOOKUP_PLANE_LIMIT_SCALE: f32 = 0.45;
const SURFACE_CACHE_LOOKUP_PLANE_SIGMA_SCALE: f32 = 0.2;

fn surface_cache_dominant_axis(normal: vec3<f32>) -> u32 {
    let absolute_normal = abs(normal);
    return select(
        select(2u, 1u, absolute_normal.y >= absolute_normal.z),
        0u,
        absolute_normal.x >= max(absolute_normal.y, absolute_normal.z)
    );
}

fn surface_cache_tangent_components(value: vec3<f32>, dominant_axis: u32) -> vec2<f32> {
    if (dominant_axis == 0u) {
        return value.yz;
    }
    if (dominant_axis == 1u) {
        return value.xz;
    }
    return value.xy;
}

fn surface_cache_sample_descriptor(
    descriptor: vec3<i32>,
    directional_bin: u32,
    receiver_position: vec3<f32>,
    receiver_normal: vec3<f32>,
    cell_exponent: i32,
    cell_size: f32
) -> SurfaceCacheTapSample {
    let patch_index_i = surface_cache_find_patch(
        descriptor,
        directional_bin,
        cell_exponent
    );
    if (patch_index_i < 0) {
        return SurfaceCacheTapSample(vec3<f32>(0.0), 0.0, 0.0);
    }

    let patch_index = u32(patch_index_i);
    let surface_patch = surface_cache[patch_index];
    let sample_count = surface_patch.history.x;
    let patch_normal = safe_normalize(surface_patch.normal_cell_exponent.xyz);
    let normal_alignment = dot(receiver_normal, patch_normal);
    if (
        sample_count <= 0.0 ||
        normal_alignment < SURFACE_CACHE_LOOKUP_NORMAL_THRESHOLD
    ) {
        return SurfaceCacheTapSample(vec3<f32>(0.0), 0.0, 0.0);
    }

    let position_delta = surface_patch.position_frame.xyz - receiver_position;
    let plane_distance = max(
        abs(dot(position_delta, receiver_normal)),
        abs(dot(position_delta, patch_normal))
    );
    let plane_limit = max(
        cell_size * SURFACE_CACHE_LOOKUP_PLANE_LIMIT_SCALE,
        0.002
    );
    if (plane_distance > plane_limit) {
        return SurfaceCacheTapSample(vec3<f32>(0.0), 0.0, 0.0);
    }
    let plane_sigma = max(
        cell_size * SURFACE_CACHE_LOOKUP_PLANE_SIGMA_SCALE,
        0.001
    );
    let normalized_plane_distance = plane_distance / plane_sigma;
    let plane_weight = exp(
        -0.5 * normalized_plane_distance * normalized_plane_distance
    );
    let normal_weight = smoothstep(
        SURFACE_CACHE_LOOKUP_NORMAL_THRESHOLD,
        0.98,
        normal_alignment
    );
    let history_weight = smoothstep(
        0.0,
        SURFACE_CACHE_MIN_QUERY_SAMPLES,
        sample_count
    );
    let geometry_weight = plane_weight * normal_weight * normal_weight * history_weight;
    if (geometry_weight <= 1e-5) {
        return SurfaceCacheTapSample(vec3<f32>(0.0), 0.0, 0.0);
    }

    let irradiance = max(
        sh_l1_rgb_calculate_irradiance(
            surface_cache_sh_patch_read(&surface_cache_sh, patch_index),
            receiver_normal
        ),
        vec3<f32>(0.0)
    );
    return SurfaceCacheTapSample(irradiance, sample_count, geometry_weight);
}

fn surface_cache_sample_level_nearest(
    position: vec3<f32>,
    normal: vec3<f32>,
    cell_exponent: i32
) -> SurfaceCacheLevelSample {
    let receiver_normal = safe_normalize(normal);
    let tap = surface_cache_sample_descriptor(
        surface_cache_quantize_position(
            position,
            receiver_normal,
            cell_exponent,
            surface_cache_params
        ),
        surface_cache_directional_bin(receiver_normal),
        position,
        receiver_normal,
        cell_exponent,
        surface_cache_cell_size(cell_exponent)
    );
    return SurfaceCacheLevelSample(
        vec4<f32>(tap.irradiance, tap.sample_count),
        clamp(tap.geometry_weight, 0.0, 1.0)
    );
}

// Reconstruct the cache as samples located at surface-constrained cell
// centers. A geometry-aware bilinear footprint removes nearest-cell steps,
// while renormalizing valid taps keeps silhouettes and missing cache entries
// from darkening the result.
fn surface_cache_sample_level(
    position: vec3<f32>,
    normal: vec3<f32>,
    cell_exponent: i32
) -> SurfaceCacheLevelSample {
    let receiver_normal = safe_normalize(normal);
    let directional_bin = surface_cache_directional_bin(receiver_normal);
    let cell_size = surface_cache_cell_size(cell_exponent);
    let dominant_axis = surface_cache_dominant_axis(receiver_normal);
    let descriptor_position = position + surface_cache_descriptor_offset(
        receiver_normal,
        cell_size,
        surface_cache_params
    );
    let tangent_position = surface_cache_tangent_components(
        descriptor_position,
        dominant_axis
    ) / cell_size - vec2<f32>(0.5);
    let tangent_base = vec2<i32>(floor(tangent_position));
    let tangent_fraction = fract(tangent_position);

    var irradiance_sum = vec3<f32>(0.0);
    var sample_count_sum = 0.0;
    var weight_sum = 0.0;
    for (var tap_y = 0i; tap_y <= 1i; tap_y = tap_y + 1i) {
        for (var tap_x = 0i; tap_x <= 1i; tap_x = tap_x + 1i) {
            let tap_offset = vec2<i32>(tap_x, tap_y);
            let descriptor = surface_cache_corner_descriptor(
                position,
                receiver_normal,
                tangent_base + tap_offset,
                dominant_axis,
                cell_size,
                surface_cache_params
            );
            let tap = surface_cache_sample_descriptor(
                descriptor,
                directional_bin,
                position,
                receiver_normal,
                cell_exponent,
                cell_size
            );
            let axis_weight = select(
                vec2<f32>(1.0) - tangent_fraction,
                tangent_fraction,
                tap_offset == vec2<i32>(1)
            );
            let weight = axis_weight.x * axis_weight.y * tap.geometry_weight;
            irradiance_sum += tap.irradiance * weight;
            sample_count_sum += tap.sample_count * weight;
            weight_sum += weight;
        }
    }

    if (weight_sum <= 1e-5) {
        return surface_cache_sample_level_nearest(
            position,
            receiver_normal,
            cell_exponent
        );
    }
    return SurfaceCacheLevelSample(
        vec4<f32>(irradiance_sum, sample_count_sum) / weight_sum,
        clamp(weight_sum, 0.0, 1.0)
    );
}

fn surface_cache_finalize_irradiance_sample(sample: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        sample.xyz * surface_cache_params.indirect_boost,
        sample.w
    );
}

fn surface_cache_finalize_sample(sample: vec4<f32>) -> vec4<f32> {
    return surface_cache_finalize_irradiance_sample(sample);
}

fn surface_cache_blend_level_samples(
    fine_sample: SurfaceCacheLevelSample,
    coarse_sample: SurfaceCacheLevelSample,
    blend: f32
) -> vec4<f32> {
    let lod_blend = clamp(blend, 0.0, 1.0);
    let fine_weight = (1.0 - lod_blend) * fine_sample.confidence;
    let coarse_weight = lod_blend * coarse_sample.confidence;
    let weight_sum = fine_weight + coarse_weight;
    if (weight_sum > 1e-5) {
        return (
            fine_sample.value * fine_weight +
            coarse_sample.value * coarse_weight
        ) / weight_sum;
    }

    // At a sparse edge the nominal LOD can have no geometrically compatible
    // taps. Select the better-supported neighbor without treating a missing
    // level as black irradiance.
    return select(
        fine_sample.value,
        coarse_sample.value,
        coarse_sample.confidence > fine_sample.confidence
    );
}

fn surface_cache_sample(
    position: vec3<f32>,
    normal: vec3<f32>
) -> vec4<f32> {
    let base_levels = surface_cache_cell_levels(position, surface_cache_params);
    let levels = surface_cache_history_cell_levels(
        position,
        surface_cache_native_history(position, normal, base_levels),
        surface_cache_params
    );
    let fine_sample = surface_cache_sample_level(
        position,
        normal,
        levels.fine_exponent
    );
    if (levels.coarse_exponent == levels.fine_exponent) {
        return surface_cache_finalize_sample(fine_sample.value);
    }

    let coarse_sample = surface_cache_sample_level(
        position,
        normal,
        levels.coarse_exponent
    );
    return surface_cache_finalize_sample(
        surface_cache_blend_level_samples(
            fine_sample,
            coarse_sample,
            levels.blend
        )
    );
}

// Cache rays and the visible deferred resolve both consume incident irradiance.
// The traced hit applies DDGI's diffuse response during recurrence, while the
// deferred lighting pass applies the visible surface's material response.
fn surface_cache_sample_nearest_irradiance(
    position: vec3<f32>,
    normal: vec3<f32>
) -> vec4<f32> {
    let base_levels = surface_cache_cell_levels(position, surface_cache_params);
    let levels = surface_cache_history_cell_levels(
        position,
        surface_cache_native_history(position, normal, base_levels),
        surface_cache_params
    );
    let fine_sample = surface_cache_sample_level_nearest(
        position,
        normal,
        levels.fine_exponent
    );
    if (levels.coarse_exponent == levels.fine_exponent) {
        return surface_cache_finalize_irradiance_sample(fine_sample.value);
    }
    let coarse_sample = surface_cache_sample_level_nearest(
        position,
        normal,
        levels.coarse_exponent
    );
    return surface_cache_finalize_irradiance_sample(
        surface_cache_blend_level_samples(
            fine_sample,
            coarse_sample,
            levels.blend
        )
    );
}

fn surface_cache_sample_nearest(
    position: vec3<f32>,
    normal: vec3<f32>
) -> vec4<f32> {
    return surface_cache_sample_nearest_irradiance(
        position,
        normal
    );
}
