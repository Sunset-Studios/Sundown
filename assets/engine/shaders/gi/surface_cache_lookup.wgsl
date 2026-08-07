fn surface_cache_find_patch(
    quantized_position: vec3<i32>,
    quantized_normal: vec3<i32>,
    cell_exponent: i32
) -> i32 {
    let key = surface_cache_hash_key(
        quantized_position,
        quantized_normal,
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
            quantized_normal,
            cell_exponent
        )
    ) {
        return i32(patch_index);
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
            i32(floor(dominant_position / cell_size)),
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
            i32(floor(dominant_position / cell_size)),
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
        i32(floor(dominant_position / cell_size))
    );
}

fn surface_cache_sample(
    position: vec3<f32>,
    normal: vec3<f32>
) -> vec4<f32> {
    let receiver_normal = safe_normalize(normal);
    let cell_exponent = surface_cache_cell_exponent(position, surface_cache_params);
    let patch_index_i = surface_cache_find_patch(
        surface_cache_quantize_position(position, cell_exponent),
        surface_cache_quantize_normal(receiver_normal),
        cell_exponent
    );
    if (patch_index_i < 0) {
        return vec4<f32>(0.0);
    }

    let patch_index = u32(patch_index_i);
    let surface_patch = surface_cache[patch_index];
    let sample_count = surface_patch.history.x;
    let patch_normal = safe_normalize(surface_patch.normal_cell_exponent.xyz);
    if (
        sample_count < SURFACE_CACHE_MIN_QUERY_SAMPLES ||
        dot(receiver_normal, patch_normal) < 0.75
    ) {
        return vec4<f32>(0.0);
    }

    let receiver_direction = safe_normalize(
        surface_cache_world_to_hemisphere(receiver_normal, patch_normal)
    );
    let irradiance = max(
        sh_l1_rgb_calculate_irradiance(
            surface_cache_sh_patch_read(&surface_cache_sh, patch_index),
            receiver_direction
        ),
        vec3<f32>(0.0)
    );
    return vec4<f32>(
        irradiance * (surface_cache_params.indirect_boost / PI),
        sample_count
    );
}
