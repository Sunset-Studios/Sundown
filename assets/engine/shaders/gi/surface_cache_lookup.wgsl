fn surface_cache_dominant_axis(normal: vec3<f32>) -> u32 {
    let absolute_normal = abs(normal);
    return select(
        select(2u, 1u, absolute_normal.y >= absolute_normal.z),
        0u,
        absolute_normal.x >= max(absolute_normal.y, absolute_normal.z)
    );
}

struct SurfaceCacheLookupContext {
    receiver_position: vec3<f32>,
    receiver_normal: vec3<f32>,
    descriptor_normal: vec3<f32>,
    directional_bin: u32,
    dominant_axis: u32,
    descriptor_dominant_axis: u32,
    hash_capacity: u32,
    hash_search_count: u32,
};

fn surface_cache_lookup_context(
    position: vec3<f32>,
    normal: vec3<f32>
) -> SurfaceCacheLookupContext {
    // All lookup entry points validate or construct a nonzero surface normal.
    // normalize therefore matches safe_normalize's selected result without its
    // additional length test at each normalization stage.
    let receiver_normal = normalize(normal);
    let descriptor_normal = normalize(receiver_normal);
    return SurfaceCacheLookupContext(
        position,
        receiver_normal,
        descriptor_normal,
        // directional_bin normalized the same receiver again. Reuse the exact
        // normalized value already needed by descriptor quantization.
        surface_cache_directional_bin(descriptor_normal),
        surface_cache_dominant_axis(receiver_normal),
        surface_cache_dominant_axis(descriptor_normal),
        max(u32(surface_cache_params.total_patch_count), 1u),
        surface_cache_hash_search_count(surface_cache_params)
    );
}

fn surface_cache_lookup_context_normalized(
    position: vec3<f32>,
    normal: vec3<f32>
) -> SurfaceCacheLookupContext {
    let dominant_axis = surface_cache_dominant_axis(normal);
    // Resolve already normalizes its G-buffer normal. Keeping that invariant
    // explicit avoids two redundant reciprocal-square-root sequences per pixel.
    return SurfaceCacheLookupContext(
        position,
        normal,
        normal,
        surface_cache_directional_bin(normal),
        dominant_axis,
        dominant_axis,
        max(u32(surface_cache_params.total_patch_count), 1u),
        surface_cache_hash_search_count(surface_cache_params)
    );
}

fn surface_cache_find_patch(
    quantized_position: vec3<i32>,
    directional_bin: u32,
    cell_exponent: i32,
    capacity: u32,
    search_count: u32
) -> i32 {
    let key = surface_cache_hash_key(
        quantized_position,
        directional_bin,
        cell_exponent
    );
    let patch_index = hashmap_find(
        &surface_cache_hashmap,
        key,
        capacity,
        search_count
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
    context: SurfaceCacheLookupContext,
    cell_exponent: i32
) -> f32 {
    let patch_index_i = surface_cache_find_patch(
        surface_cache_quantize_position(context.receiver_position, context.receiver_normal, cell_exponent, surface_cache_params),
        context.directional_bin,
        cell_exponent,
        context.hash_capacity,
        context.hash_search_count
    );
    if (patch_index_i < 0) {
        return 0.0;
    }
    let patch_index = u32(patch_index_i);
    // Exact descriptor matches refer only to initialized patches, whose stored
    // normals were validated at feedback time.
    // Feedback is the sole writer and stores an already normalized G-buffer
    // normal. Avoid renormalizing it for every history query.
    let patch_normal = surface_cache[patch_index].normal_cell_exponent.xyz;
    if (dot(
        context.receiver_normal,
        patch_normal
    ) < SURFACE_CACHE_LOOKUP_NORMAL_THRESHOLD) {
        return 0.0;
    }
    let position_delta = surface_cache[patch_index].position_frame.xyz -
        context.receiver_position;
    let plane_distance = max(
        abs(dot(position_delta, context.receiver_normal)),
        abs(dot(position_delta, patch_normal))
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
    let metadata = surface_cache[patch_index].metadata;
    return select(
        metadata.w,
        metadata.y,
        u32(metadata.z) == u32(surface_cache_params.frame_index)
    );
}

fn surface_cache_native_history_exponents(
    context: SurfaceCacheLookupContext,
    fine_exponent: i32,
    coarse_exponent: i32
) -> f32 {
    let fine_history = surface_cache_level_history(
        context,
        fine_exponent
    );
    if (coarse_exponent == fine_exponent) {
        return fine_history;
    }
    return min(
        fine_history,
        surface_cache_level_history(
            context,
            coarse_exponent
        )
    );
}

fn surface_cache_native_history(
    context: SurfaceCacheLookupContext,
    levels: SurfaceCacheCellLevels
) -> f32 {
    return surface_cache_native_history_exponents(
        context,
        levels.fine_exponent,
        levels.coarse_exponent
    );
}

fn surface_cache_corner_descriptor(
    position: vec3<f32>,
    normal: vec3<f32>,
    tangent_cell: vec2<i32>,
    dominant_axis: u32,
    cell_size: f32,
    descriptor_offset: vec3<f32>
) -> vec3<i32> {
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
    error_variance: f32,
};

struct SurfaceCacheLevelSample {
    value: vec4<f32>,
    confidence: f32,
    error_variance: f32,
};

struct SurfaceCachePresentationSample {
    value: vec4<f32>,
    confidence: f32,
    standard_error: f32,
    fallback_weight: f32,
};

const SURFACE_CACHE_LOOKUP_NORMAL_THRESHOLD: f32 = 0.82;
const SURFACE_CACHE_LOOKUP_PLANE_LIMIT_SCALE: f32 = 0.45;
const SURFACE_CACHE_LOOKUP_PLANE_SIGMA_SCALE: f32 = 0.2;

fn surface_cache_evaluate_patch_irradiance(
    patch_index: u32,
    normal: vec3<f32>
) -> vec3<f32> {
    let base_offset = patch_index * SURFACE_CACHE_SH_PATCH_SIZE_U32;
    let packed_0 = unpack2x16float(surface_cache_sh[base_offset]);
    let packed_1 = unpack2x16float(surface_cache_sh[base_offset + 1u]);
    let packed_2 = unpack2x16float(surface_cache_sh[base_offset + 2u]);
    let packed_3 = unpack2x16float(surface_cache_sh[base_offset + 3u]);
    let packed_4 = unpack2x16float(surface_cache_sh[base_offset + 4u]);
    let packed_5 = unpack2x16float(surface_cache_sh[base_offset + 5u]);
    let coefficient_0 = vec3<f32>(packed_0.x, packed_0.y, packed_1.x);
    let coefficient_1 = vec3<f32>(packed_1.y, packed_2.x, packed_2.y);
    let coefficient_2 = vec3<f32>(packed_3.x, packed_3.y, packed_4.x);
    let coefficient_3 = vec3<f32>(packed_4.y, packed_5.x, packed_5.y);

    // Evaluate the same convolved L1 dot product directly. This avoids
    // materializing three temporary SH structs and indexing their arrays for
    // every accepted reconstruction tap.
    var irradiance = vec3<f32>(SH_BASIS_L0) *
        (coefficient_0 * SH_COSINE_A0);
    irradiance += vec3<f32>(SH_BASIS_L1 * normal.y) *
        (coefficient_1 * SH_COSINE_A1);
    irradiance += vec3<f32>(SH_BASIS_L1 * normal.z) *
        (coefficient_2 * SH_COSINE_A1);
    irradiance += vec3<f32>(SH_BASIS_L1 * normal.x) *
        (coefficient_3 * SH_COSINE_A1);
    return irradiance;
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
    context: SurfaceCacheLookupContext,
    cell_exponent: i32,
    plane_limit: f32,
    inverse_plane_sigma: f32
) -> SurfaceCacheTapSample {
    let patch_index_i = surface_cache_find_patch(
        descriptor,
        context.directional_bin,
        cell_exponent,
        context.hash_capacity,
        context.hash_search_count
    );
    if (patch_index_i < 0) {
        return SurfaceCacheTapSample(vec3<f32>(0.0), 0.0, 0.0, 0.0);
    }

    let patch_index = u32(patch_index_i);
    let sample_count = surface_cache[patch_index].history.x;
    // Newly allocated patches are common during camera movement. Reject them
    // before normalizing geometry or evaluating any spatial weights.
    if (sample_count <= 0.0) {
        return SurfaceCacheTapSample(vec3<f32>(0.0), 0.0, 0.0, 0.0);
    }
    // Patch normals have already been normalized by the feedback pass. This
    // removes the most expensive repeated arithmetic in the accepted-tap path.
    let patch_normal = surface_cache[patch_index].normal_cell_exponent.xyz;
    let normal_alignment = dot(context.receiver_normal, patch_normal);
    if (normal_alignment < SURFACE_CACHE_LOOKUP_NORMAL_THRESHOLD) {
        return SurfaceCacheTapSample(vec3<f32>(0.0), 0.0, 0.0, 0.0);
    }

    let position_delta = surface_cache[patch_index].position_frame.xyz -
        context.receiver_position;
    let plane_distance = max(
        abs(dot(position_delta, context.receiver_normal)),
        abs(dot(position_delta, patch_normal))
    );
    if (plane_distance > plane_limit) {
        return SurfaceCacheTapSample(vec3<f32>(0.0), 0.0, 0.0, 0.0);
    }
    let normalized_plane_distance = plane_distance * inverse_plane_sigma;
    let plane_weight = exp(
        -0.5 * normalized_plane_distance * normalized_plane_distance
    );
    // Most accepted taps are well-aligned and converged. Preserve smoothstep
    // inside its transition while bypassing it at the exact saturated endpoint.
    var normal_weight = 1.0;
    if (normal_alignment < 0.98) {
        normal_weight = smoothstep(
            SURFACE_CACHE_LOOKUP_NORMAL_THRESHOLD,
            0.98,
            normal_alignment
        );
    }
    var history_weight = 1.0;
    if (sample_count < SURFACE_CACHE_MIN_QUERY_SAMPLES) {
        history_weight = smoothstep(
            0.0,
            SURFACE_CACHE_MIN_QUERY_SAMPLES,
            sample_count
        );
    }
    let geometry_weight = plane_weight * normal_weight * normal_weight * history_weight;
    if (geometry_weight <= 1e-5) {
        return SurfaceCacheTapSample(vec3<f32>(0.0), 0.0, 0.0, 0.0);
    }

    let irradiance = max(
        surface_cache_evaluate_patch_irradiance(
            patch_index,
            context.receiver_normal
        ),
        vec3<f32>(0.0)
    );
    let history = surface_cache[patch_index].history;
    let sample_variance = max(history.w - history.z * history.z, 0.0);
    let error_variance = sample_variance / max(sample_count, 1.0);
    return SurfaceCacheTapSample(
        irradiance,
        sample_count,
        geometry_weight,
        error_variance
    );
}

fn surface_cache_sample_level_nearest(
    context: SurfaceCacheLookupContext,
    cell_exponent: i32
) -> SurfaceCacheLevelSample {
    let cell_size = surface_cache_cell_size(cell_exponent);
    let tap = surface_cache_sample_descriptor(
        surface_cache_quantize_position(context.receiver_position, context.receiver_normal, cell_exponent, surface_cache_params),
        context,
        cell_exponent,
        max(cell_size * SURFACE_CACHE_LOOKUP_PLANE_LIMIT_SCALE, 0.002),
        1.0 / max(cell_size * SURFACE_CACHE_LOOKUP_PLANE_SIGMA_SCALE, 0.001)
    );
    return SurfaceCacheLevelSample(
        vec4<f32>(tap.irradiance, tap.sample_count),
        clamp(tap.geometry_weight, 0.0, 1.0),
        tap.error_variance
    );
}

// Reconstruct the cache as samples located at surface-constrained cell
// centers. A geometry-aware bilinear footprint removes nearest-cell steps,
// while renormalizing valid taps keeps silhouettes and missing cache entries
// from darkening the result.
fn surface_cache_sample_level(
    context: SurfaceCacheLookupContext,
    cell_exponent: i32
) -> SurfaceCacheLevelSample {
    let cell_size = surface_cache_cell_size(cell_exponent);
    let dominant_axis = context.dominant_axis;
    let descriptor_offset = surface_cache_descriptor_offset(
        context.descriptor_normal,
        cell_size,
        surface_cache_params
    );
    let descriptor_position = context.receiver_position + descriptor_offset;
    let tangent_position = surface_cache_tangent_components(
        descriptor_position,
        dominant_axis
    ) / cell_size - vec2<f32>(0.5);
    let tangent_base = vec2<i32>(floor(tangent_position));
    let tangent_fraction = fract(tangent_position);
    let plane_limit = max(
        cell_size * SURFACE_CACHE_LOOKUP_PLANE_LIMIT_SCALE,
        0.002
    );
    let inverse_plane_sigma = 1.0 / max(
        cell_size * SURFACE_CACHE_LOOKUP_PLANE_SIGMA_SCALE,
        0.001
    );

    var irradiance_sum = vec3<f32>(0.0);
    var sample_count_sum = 0.0;
    var weight_sum = 0.0;
    var error_variance_sum = 0.0;
    for (var tap_y = 0i; tap_y <= 1i; tap_y = tap_y + 1i) {
        for (var tap_x = 0i; tap_x <= 1i; tap_x = tap_x + 1i) {
            let tap_offset = vec2<i32>(tap_x, tap_y);
            let axis_weight = select(
                vec2<f32>(1.0) - tangent_fraction,
                tangent_fraction,
                tap_offset == vec2<i32>(1)
            );
            let bilinear_weight = axis_weight.x * axis_weight.y;
            // Exact zero-weight corners cannot affect the renormalized result.
            // Skip their hash probes, patch reads, and potential SH evaluation.
            if (bilinear_weight <= 0.0) {
                continue;
            }
            let descriptor = surface_cache_corner_descriptor(
                context.receiver_position,
                context.receiver_normal,
                tangent_base + tap_offset,
                dominant_axis,
                cell_size,
                descriptor_offset
            );
            let tap = surface_cache_sample_descriptor(
                descriptor,
                context,
                cell_exponent,
                plane_limit,
                inverse_plane_sigma
            );
            let weight = bilinear_weight * tap.geometry_weight;
            irradiance_sum += tap.irradiance * weight;
            sample_count_sum += tap.sample_count * weight;
            error_variance_sum += tap.error_variance * weight * weight;
            weight_sum += weight;
        }
    }

    if (weight_sum <= 1e-5) {
        return surface_cache_sample_level_nearest(
            context,
            cell_exponent
        );
    }
    return SurfaceCacheLevelSample(
        vec4<f32>(irradiance_sum, sample_count_sum) / weight_sum,
        clamp(weight_sum, 0.0, 1.0),
        error_variance_sum / (weight_sum * weight_sum)
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

fn surface_cache_blend_level_estimates(
    fine_sample: SurfaceCacheLevelSample,
    coarse_sample: SurfaceCacheLevelSample,
    blend: f32
) -> SurfaceCacheLevelSample {
    let lod_blend = clamp(blend, 0.0, 1.0);
    let fine_weight = (1.0 - lod_blend) * fine_sample.confidence;
    let coarse_weight = lod_blend * coarse_sample.confidence;
    let weight_sum = fine_weight + coarse_weight;
    if (weight_sum > 1e-5) {
        return SurfaceCacheLevelSample(
            (
                fine_sample.value * fine_weight +
                coarse_sample.value * coarse_weight
            ) / weight_sum,
            clamp(weight_sum, 0.0, 1.0),
            (
                fine_sample.error_variance * fine_weight * fine_weight +
                coarse_sample.error_variance * coarse_weight * coarse_weight
            ) / (weight_sum * weight_sum)
        );
    }

    // At a sparse edge the nominal LOD can have no geometrically compatible
    // taps. Select the better-supported neighbor without treating a missing
    // level as black irradiance.
    if (coarse_sample.confidence > fine_sample.confidence) {
        return coarse_sample;
    }
    return fine_sample;
}

fn surface_cache_blend_level_samples(
    fine_sample: SurfaceCacheLevelSample,
    coarse_sample: SurfaceCacheLevelSample,
    blend: f32
) -> vec4<f32> {
    return surface_cache_blend_level_estimates(
        fine_sample,
        coarse_sample,
        blend
    ).value;
}

fn surface_cache_level_estimate_confidence(
    sample: SurfaceCacheLevelSample
) -> f32 {
    if (sample.value.w <= 0.0 || sample.confidence <= 1e-5) {
        return 0.0;
    }

    // Sample count prevents a coincidentally zero first-frame variance from
    // promoting a patch. Standard error then lets smooth, low-variance patches
    // mature sooner than difficult emissive or high-contrast neighborhoods.
    let sample_readiness = smoothstep(
        SURFACE_CACHE_MIN_QUERY_SAMPLES,
        64.0,
        sample.value.w
    );
    let sample_luminance = max(luminance(sample.value.xyz), 0.05);
    let relative_error = sqrt(max(sample.error_variance, 0.0)) /
        sample_luminance;
    let uncertainty_readiness = 1.0 - smoothstep(0.15, 0.75, relative_error);
    return clamp(
        sample.confidence * sample_readiness * uncertainty_readiness,
        0.0,
        1.0
    );
}

// Resolve native cache levels separately from their one-level-coarser parent.
// A young native estimate remains hidden behind a compatible mature parent
// until its measured confidence crosses the configured promotion window.
fn surface_cache_presentation_sample_context(
    context: SurfaceCacheLookupContext
) -> SurfaceCachePresentationSample {
    let exponent_value = surface_cache_cell_exponent_value(
        context.receiver_position,
        surface_cache_params
    );
    let native_fine_exponent = i32(floor(exponent_value));
    let native_coarse_exponent = min(
        native_fine_exponent + 1,
        SURFACE_CACHE_MAX_CELL_EXPONENT
    );
    let level_blend = select(
        smoothstep(
            SURFACE_CACHE_LEVEL_BLEND_START,
            SURFACE_CACHE_LEVEL_BLEND_END,
            fract(exponent_value)
        ),
        0.0,
        native_fine_exponent == native_coarse_exponent
    );

    let native_fine = surface_cache_sample_level(
        context,
        native_fine_exponent
    );
    var native = native_fine;
    var native_coarse = native_fine;
    if (native_coarse_exponent != native_fine_exponent) {
        native_coarse = surface_cache_sample_level(
            context,
            native_coarse_exponent
        );
        native = surface_cache_blend_level_estimates(
            native_fine,
            native_coarse,
            level_blend
        );
    }

    let parent_fine_exponent = native_coarse_exponent;
    let parent_coarse_exponent = min(
        parent_fine_exponent + 1,
        SURFACE_CACHE_MAX_CELL_EXPONENT
    );
    let parent_fine = native_coarse;
    var parent = parent_fine;
    if (parent_coarse_exponent != parent_fine_exponent) {
        parent = surface_cache_blend_level_estimates(
            parent_fine,
            surface_cache_sample_level(context, parent_coarse_exponent),
            level_blend
        );
    }

    let native_confidence = surface_cache_level_estimate_confidence(native);
    let parent_confidence = surface_cache_level_estimate_confidence(parent);
    var presented = native;
    var presented_confidence = native_confidence;
    var fallback_weight = 0.0;

    if (parent.value.w > 0.0 && parent.confidence > 1e-5) {
        if (native.value.w <= 0.0 || native.confidence <= 1e-5) {
            presented = parent;
            presented_confidence = parent_confidence;
            fallback_weight = 1.0;
        } else {
            let native_promotion = smoothstep(
                surface_cache_params.native_promotion_start_confidence,
                surface_cache_params.native_promotion_end_confidence,
                native_confidence
            );
            // Never hold on to a parent which is less trustworthy than the
            // native level. This also keeps a newly allocated parent from
            // becoming another noisy fallback during a large reveal.
            let weak_parent_release = 1.0 - smoothstep(
                0.0,
                0.5,
                parent_confidence
            );
            let inferior_parent_release = smoothstep(
                0.0,
                0.2,
                native_confidence - parent_confidence
            );
            let parent_release = max(
                weak_parent_release,
                inferior_parent_release
            );
            let promotion = max(native_promotion, parent_release);
            fallback_weight = 1.0 - promotion;
            presented = SurfaceCacheLevelSample(
                mix(parent.value, native.value, promotion),
                mix(parent.confidence, native.confidence, promotion),
                mix(
                    parent.error_variance,
                    native.error_variance,
                    promotion * promotion
                )
            );
            presented_confidence = mix(
                parent_confidence,
                native_confidence,
                promotion
            );
        }
    }

    let boost = surface_cache_params.indirect_boost;
    return SurfaceCachePresentationSample(
        vec4<f32>(presented.value.xyz * boost, presented.value.w),
        presented_confidence,
        sqrt(max(presented.error_variance, 0.0)) * boost,
        fallback_weight
    );
}

fn surface_cache_presentation_sample_normalized(
    position: vec3<f32>,
    normal: vec3<f32>
) -> SurfaceCachePresentationSample {
    return surface_cache_presentation_sample_context(
        surface_cache_lookup_context_normalized(position, normal)
    );
}

fn surface_cache_sample_context(
    context: SurfaceCacheLookupContext
) -> vec4<f32> {
    // Native blending is never consumed here; only its exponents seed history
    // selection. Avoid the otherwise unused fract and smoothstep calculation.
    let base_exponent_value = surface_cache_cell_exponent_value(
        context.receiver_position,
        surface_cache_params
    );
    let levels = surface_cache_history_cell_levels_from_base(
        base_exponent_value,
        surface_cache_params
    );
    let fine_sample = surface_cache_sample_level(
        context,
        levels.fine_exponent
    );
    if (levels.coarse_exponent == levels.fine_exponent) {
        return surface_cache_finalize_sample(fine_sample.value);
    }

    let coarse_sample = surface_cache_sample_level(
        context,
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

fn surface_cache_sample(
    position: vec3<f32>,
    normal: vec3<f32>
) -> vec4<f32> {
    return surface_cache_sample_context(
        surface_cache_lookup_context(position, normal)
    );
}

fn surface_cache_sample_normalized(
    position: vec3<f32>,
    normal: vec3<f32>
) -> vec4<f32> {
    return surface_cache_sample_context(
        surface_cache_lookup_context_normalized(position, normal)
    );
}

// Cache rays and the visible deferred resolve both consume incident irradiance.
// The traced hit applies DDGI's diffuse response during recurrence, while the
// deferred lighting pass applies the visible surface's material response.
fn surface_cache_sample_nearest_irradiance(
    position: vec3<f32>,
    normal: vec3<f32>
) -> vec4<f32> {
    let base_exponent_value = surface_cache_cell_exponent_value(
        position,
        surface_cache_params
    );
    let context = surface_cache_lookup_context(position, normal);
    let levels = surface_cache_history_cell_levels_from_base(
        base_exponent_value,
        surface_cache_params
    );
    let fine_sample = surface_cache_sample_level_nearest(
        context,
        levels.fine_exponent
    );
    if (levels.coarse_exponent == levels.fine_exponent) {
        return surface_cache_finalize_irradiance_sample(fine_sample.value);
    }
    let coarse_sample = surface_cache_sample_level_nearest(
        context,
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
