fn scgi_find_patch(
    quantized_position: vec3<i32>,
    quantized_normal: vec2<i32>,
    lod: u32
) -> i32 {
    let bucket_start = scgi_bucket_start(quantized_position, quantized_normal, lod, scgi_params);
    let fingerprint = scgi_hash_fingerprint(quantized_position, quantized_normal, lod);
    for (var probe = 0u; probe < SCGI_BUCKET_SIZE; probe = probe + 1u) {
        let patch_index = bucket_start + probe;
        if (
            surface_cache[patch_index].fingerprint == fingerprint &&
            scgi_patch_descriptor_matches(
                surface_cache[patch_index].position_frame.xyz,
                surface_cache[patch_index].normal_unused.xyz,
                quantized_position,
                quantized_normal,
                lod,
                scgi_params
            )
        ) {
            return i32(patch_index);
        }
    }
    return -1;
}

fn scgi_surface_corner_descriptor(
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

fn scgi_sample_surface_cache(
    position: vec3<f32>,
    normal: vec3<f32>,
    camera_position: vec3<f32>
) -> vec4<f32> {
    let lod = scgi_select_lod(position, camera_position, scgi_params);
    let cell_size = scgi_lod_cell_size(lod, scgi_params);
    let quantized_normal = scgi_quantize_normal(normal);

    // A surface cache is sparse in 3D but dense along the local surface. Pick
    // the dominant normal axis, bilinearly sample the other two axes, and use
    // the receiver plane to predict the cell on the dominant axis. This spends
    // all four taps on patches the surface can actually cross.
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

    // Split the tangent-plane quad along a fixed diagonal. Three barycentric
    // taps reconstruct a continuous field with one fewer hash lookup and SH
    // evaluation than bilinear reconstruction. The final resolve supplies the
    // already-filtered SH buffer; this function adds no denoising of its own.
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
        let offset = corner_offsets[corner];
        let tangent_cell = base_tangent_cell + offset;
        let descriptor_position = scgi_surface_corner_descriptor(
            position,
            normal,
            tangent_cell,
            dominant_axis,
            cell_size
        );
        let patch_index = scgi_find_patch(descriptor_position, quantized_normal, lod);
        if (patch_index < 0) {
            continue;
        }

        let weight = corner_weights[corner];
        let surface_patch = surface_cache[u32(patch_index)];
        let patch_normal = safe_normalize(surface_patch.normal_unused.xyz);
        let sample_count = surface_patch.history.x;
        let patch_sh = scgi_sh_patch_read(&surface_cache_sh, u32(patch_index));
        let receiver_direction = safe_normalize(scgi_world_to_hemisphere(normal, patch_normal));
        let patch_irradiance = max(
            sh_l1_rgb_calculate_irradiance(patch_sh, receiver_direction),
            vec3<f32>(0.0)
        );
        irradiance_sum += patch_irradiance * weight;
        sample_sum += sample_count * weight;
        weight_sum += weight;
    }

    if (weight_sum > 1e-6) {
        let irradiance = irradiance_sum * (scgi_params.indirect_boost / weight_sum);
        return vec4<f32>(irradiance, sample_sum / weight_sum);
    }

    let nearest_position = scgi_quantize_position(position, lod, scgi_params);
    let nearest_index = scgi_find_patch(nearest_position, quantized_normal, lod);
    if (nearest_index >= 0) {
        let nearest_patch = u32(nearest_index);
        let nearest_sh = scgi_sh_patch_read(&surface_cache_sh, nearest_patch);
        let patch_normal = safe_normalize(surface_cache[nearest_patch].normal_unused.xyz);
        let receiver_direction = safe_normalize(scgi_world_to_hemisphere(normal, patch_normal));
        let irradiance = max(
            sh_l1_rgb_calculate_irradiance(nearest_sh, receiver_direction),
            vec3<f32>(0.0)
        );
        return vec4<f32>(
            irradiance * scgi_params.indirect_boost,
            surface_cache[nearest_patch].history.x
        );
    }
    return vec4<f32>(0.0);
}
