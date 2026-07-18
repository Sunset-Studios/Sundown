fn surface_cache_find_patch(
    quantized_position: vec3<i32>,
    quantized_normal: vec2<i32>,
    lod: u32
) -> i32 {
    let bucket_start = surface_cache_bucket_start(
        quantized_position,
        quantized_normal,
        lod,
        surface_cache_params
    );
    let fingerprint = surface_cache_hash_fingerprint(
        quantized_position,
        quantized_normal,
        lod
    );
    for (var probe = 0u; probe < SURFACE_CACHE_BUCKET_SIZE; probe = probe + 1u) {
        let patch_index = bucket_start + probe;
        if (
            surface_cache[patch_index].fingerprint == fingerprint &&
            surface_cache_patch_descriptor_matches(
                surface_cache[patch_index].grid_key,
                quantized_position,
                quantized_normal,
                lod
            )
        ) {
            return i32(patch_index);
        }
    }
    return -1;
}

fn surface_cache_corner_descriptor(
    position: vec3<f32>,
    normal: vec3<f32>,
    tangent_cell: vec2<i32>,
    dominant_axis: u32,
    cell_size: f32
) -> vec3<i32> {
    let tangent_center = (vec2<f32>(tangent_cell) + vec2<f32>(0.5)) * cell_size;
    if (dominant_axis == 0u) {
        let dominant_position = position.x - (
            normal.y * (tangent_center.x - position.y) +
            normal.z * (tangent_center.y - position.z)
        ) / normal.x;
        return vec3<i32>(
            i32(floor(dominant_position / cell_size + 0.0001)),
            tangent_cell.x,
            tangent_cell.y
        );
    }
    if (dominant_axis == 1u) {
        let dominant_position = position.y - (
            normal.x * (tangent_center.x - position.x) +
            normal.z * (tangent_center.y - position.z)
        ) / normal.y;
        return vec3<i32>(
            tangent_cell.x,
            i32(floor(dominant_position / cell_size + 0.0001)),
            tangent_cell.y
        );
    }

    let dominant_position = position.z - (
        normal.x * (tangent_center.x - position.x) +
        normal.y * (tangent_center.y - position.y)
    ) / normal.z;
    return vec3<i32>(
        tangent_cell.x,
        tangent_cell.y,
        i32(floor(dominant_position / cell_size + 0.0001))
    );
}

// Resolve one explicit logarithmic level. w is the weighted history count and
// doubles as a validity/confidence channel for adjacent-level selection.
fn surface_cache_sample_lod(
    position: vec3<f32>,
    normal_input: vec3<f32>,
    lod: u32
) -> vec4<f32> {
    let normal = safe_normalize(normal_input);
    let cell_size = surface_cache_lod_cell_size(lod, surface_cache_params);
    let quantized_normal = surface_cache_quantize_normal(normal);
    let absolute_normal = abs(normal);
    let dominant_axis = select(
        select(2u, 1u, absolute_normal.y >= absolute_normal.z),
        0u,
        absolute_normal.x >= max(absolute_normal.y, absolute_normal.z)
    );
    var tangent_grid = position.xy / cell_size - vec2<f32>(0.5);
    if (dominant_axis == 0u) {
        tangent_grid = position.yz / cell_size - vec2<f32>(0.5);
    } else if (dominant_axis == 1u) {
        tangent_grid = position.xz / cell_size - vec2<f32>(0.5);
    }
    let base_tangent_cell = vec2<i32>(floor(tangent_grid));
    let tangent_fraction = fract(tangent_grid);

    var corner_offsets = array<vec2<i32>, 3>(
        vec2<i32>(0, 0),
        vec2<i32>(1, 0),
        vec2<i32>(0, 1)
    );
    var corner_weights = array<f32, 3>(
        1.0 - tangent_fraction.x - tangent_fraction.y,
        tangent_fraction.x,
        tangent_fraction.y
    );
    if (tangent_fraction.x + tangent_fraction.y > 1.0) {
        corner_offsets = array<vec2<i32>, 3>(
            vec2<i32>(1, 1),
            vec2<i32>(0, 1),
            vec2<i32>(1, 0)
        );
        corner_weights = array<f32, 3>(
            tangent_fraction.x + tangent_fraction.y - 1.0,
            1.0 - tangent_fraction.x,
            1.0 - tangent_fraction.y
        );
    }

    var irradiance_sum = vec3<f32>(0.0);
    var sample_sum = 0.0;
    var weight_sum = 0.0;
    for (var corner = 0u; corner < 3u; corner = corner + 1u) {
        let descriptor_position = surface_cache_corner_descriptor(
            position,
            normal,
            base_tangent_cell + corner_offsets[corner],
            dominant_axis,
            cell_size
        );
        let patch_index_i = surface_cache_find_patch(
            descriptor_position,
            quantized_normal,
            lod
        );
        if (patch_index_i < 0) {
            continue;
        }

        let patch_index = u32(patch_index_i);
        let surface_patch = surface_cache[patch_index];
        let sample_count = surface_patch.history.x;
        let patch_normal = safe_normalize(surface_patch.normal_lod.xyz);
        if (
            sample_count < SURFACE_CACHE_MIN_QUERY_SAMPLES ||
            dot(normal, patch_normal) < 0.75
        ) {
            continue;
        }

        let receiver_direction = safe_normalize(
            surface_cache_world_to_hemisphere(normal, patch_normal)
        );
        let patch_irradiance = max(
            sh_l1_rgb_calculate_irradiance(
                surface_cache_sh_patch_read(&surface_cache_sh, patch_index),
                receiver_direction
            ),
            vec3<f32>(0.0)
        );
        let confidence = clamp(sample_count / 8.0, 0.25, 1.0);
        let weight = corner_weights[corner] * confidence;
        irradiance_sum += patch_irradiance * weight;
        sample_sum += sample_count * weight;
        weight_sum += weight;
    }

    if (weight_sum > 1e-6) {
        return vec4<f32>(irradiance_sum / weight_sum, sample_sum / weight_sum);
    }

    // Sparse regions still get an exact-cell fallback without extending the
    // footprint across unrelated geometry.
    let nearest_position = surface_cache_quantize_position(
        position,
        lod,
        surface_cache_params
    );
    let nearest_index_i = surface_cache_find_patch(
        nearest_position,
        quantized_normal,
        lod
    );
    if (nearest_index_i >= 0) {
        let nearest_index = u32(nearest_index_i);
        let sample_count = surface_cache[nearest_index].history.x;
        let patch_normal = safe_normalize(surface_cache[nearest_index].normal_lod.xyz);
        if (
            sample_count >= SURFACE_CACHE_MIN_QUERY_SAMPLES &&
            dot(normal, patch_normal) >= 0.75
        ) {
            let receiver_direction = safe_normalize(
                surface_cache_world_to_hemisphere(normal, patch_normal)
            );
            let irradiance = max(
                sh_l1_rgb_calculate_irradiance(
                    surface_cache_sh_patch_read(&surface_cache_sh, nearest_index),
                    receiver_direction
                ),
                vec3<f32>(0.0)
            );
            return vec4<f32>(irradiance, sample_count);
        }
    }
    return vec4<f32>(0.0);
}

fn surface_cache_finalize_sample(sample: vec4<f32>) -> vec4<f32> {
    // The downstream indirect BRDF multiplies by diffuse albedo, so expose
    // Lambertian diffuse radiance (irradiance / PI), not raw irradiance.
    return vec4<f32>(
        sample.xyz * (surface_cache_params.indirect_boost / PI),
        sample.w
    );
}

fn surface_cache_sample(
    position: vec3<f32>,
    normal: vec3<f32>,
    camera_position: vec3<f32>
) -> vec4<f32> {
    let maximum_lod = u32(surface_cache_params.surface_cache_lod_count) - 1u;
    let lod_value = surface_cache_lod_value(
        position,
        camera_position,
        surface_cache_params
    );
    let lower_lod = min(u32(floor(lod_value)), maximum_lod);
    let upper_lod = min(lower_lod + 1u, maximum_lod);
    let lower_sample = surface_cache_sample_lod(position, normal, lower_lod);

    if (upper_lod != lower_lod) {
        let upper_sample = surface_cache_sample_lod(position, normal, upper_lod);
        if (lower_sample.w > 0.0 && upper_sample.w > 0.0) {
            let level_blend = smoothstep(0.2, 0.8, fract(lod_value));
            return surface_cache_finalize_sample(mix(
                lower_sample,
                upper_sample,
                level_blend
            ));
        }
        if (upper_sample.w > 0.0) {
            return surface_cache_finalize_sample(upper_sample);
        }
    }

    if (lower_sample.w > 0.0) {
        return surface_cache_finalize_sample(lower_sample);
    }

    // Missing fine cells fall back to successively coarser parents. Levels
    // are alternatives, never additive contributors, so radiance is not
    // double-counted when multiple LODs overlap in the table.
    for (var fallback_lod = upper_lod + 1u; fallback_lod <= maximum_lod; fallback_lod = fallback_lod + 1u) {
        let fallback_sample = surface_cache_sample_lod(
            position,
            normal,
            fallback_lod
        );
        if (fallback_sample.w > 0.0) {
            return surface_cache_finalize_sample(fallback_sample);
        }
    }
    return vec4<f32>(0.0);
}
