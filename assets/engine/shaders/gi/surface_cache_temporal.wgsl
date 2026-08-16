#include "common.wgsl"

struct SurfaceCacheTemporalParams {
    response: f32,
    max_history_frames: f32,
    depth_threshold: f32,
    normal_threshold: f32,
    spatial_filter_radius: f32,
    recurrent_blur_max_radius: f32,
    recurrent_blur_history_frames: f32,
    recurrent_blur_min_strength: f32,
    recurrent_blur_max_strength: f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
};

struct SurfaceCacheHistoryTap {
    value: vec4<f32>,
    weight: f32,
};

struct SurfaceCacheHistoryRepair {
    value: vec4<f32>,
    score: f32,
};

struct SurfaceCacheCurrentEstimate {
    value: vec4<f32>,
    luminance_minimum: f32,
    luminance_maximum: f32,
    luminance_sum: f32,
    luminance_squared_sum: f32,
    valid_tap_count: f32,
};

@group(1) @binding(0) var<uniform> temporal_params: SurfaceCacheTemporalParams;
@group(1) @binding(1) var current_diffuse: texture_2d<f32>;
@group(1) @binding(2) var history_diffuse: texture_2d<f32>;
@group(1) @binding(3) var depth_texture: texture_2d<f32>;
@group(1) @binding(4) var prev_depth_texture: texture_2d<f32>;
@group(1) @binding(5) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(6) var prev_gbuffer_normal: texture_2d<f32>;
@group(1) @binding(7) var motion_texture: texture_2d<f32>;
@group(1) @binding(8) var output_diffuse: texture_storage_2d<rgba16float, write>;

fn surface_cache_history_tap(
    tap_coord: vec2<i32>,
    tap_weight: f32,
    resolution: vec2<u32>,
    current_position: vec3<f32>,
    current_normal: vec3<f32>,
    current_linear_depth: f32,
    compare_surface_plane: bool,
    view_index: u32
) -> SurfaceCacheHistoryTap {
    // Static reprojection commonly produces one unit bilinear weight and three
    // exact zeros. Reject zero-weight taps before any history or geometry fetches.
    if (tap_weight <= 0.0) {
        return SurfaceCacheHistoryTap(vec4<f32>(0.0), 0.0);
    }
    if (
        tap_coord.x < 0 || tap_coord.y < 0 ||
        tap_coord.x >= i32(resolution.x) ||
        tap_coord.y >= i32(resolution.y)
    ) {
        return SurfaceCacheHistoryTap(vec4<f32>(0.0), 0.0);
    }

    let history = textureLoad(history_diffuse, tap_coord, 0);
    if (history.w <= 0.0) {
        return SurfaceCacheHistoryTap(vec4<f32>(0.0), 0.0);
    }
    let previous_normal_data = textureLoad(
        prev_gbuffer_normal,
        tap_coord,
        0
    ).xyz;
    if (dot(previous_normal_data, previous_normal_data) <= 1e-8) {
        return SurfaceCacheHistoryTap(vec4<f32>(0.0), 0.0);
    }
    // Reject normal discontinuities before fetching depth and reconstructing
    // a world position for a tap that cannot contribute.
    let previous_normal = normalize(previous_normal_data);
    if (
        dot(current_normal, previous_normal) <
        temporal_params.normal_threshold
    ) {
        return SurfaceCacheHistoryTap(vec4<f32>(0.0), 0.0);
    }
    let previous_depth = textureLoad(prev_depth_texture, tap_coord, 0).r;
    if (previous_depth >= 1.0) {
        return SurfaceCacheHistoryTap(vec4<f32>(0.0), 0.0);
    }

    let previous_position = reconstruct_prev_world_position(
        coord_to_uv(tap_coord, resolution),
        previous_depth,
        view_index
    );
    let previous_linear_depth = abs(
        (view_buffer[view_index].view_matrix * vec4<f32>(previous_position, 1.0)).z
    );
    let relative_depth_delta = abs(
        previous_linear_depth - current_linear_depth
    ) / max(current_linear_depth, 1e-3);
    let relative_plane_delta = abs(dot(
        previous_position - current_position,
        current_normal
    )) / max(current_linear_depth, 1e-3);
    // Wide blur taps should follow the receiving plane. Raw view-depth deltas
    // reject one screen axis on sloped surfaces and turn a circular kernel into
    // an iso-depth streak; ordinary bilinear reprojection keeps its strict
    // depth comparison.
    let geometry_delta = select(
        relative_depth_delta,
        relative_plane_delta,
        compare_surface_plane
    );
    let geometry_threshold = select(
        temporal_params.depth_threshold,
        max(temporal_params.depth_threshold * 0.5, 0.005),
        compare_surface_plane
    );
    if (geometry_delta > geometry_threshold) {
        return SurfaceCacheHistoryTap(vec4<f32>(0.0), 0.0);
    }

    return SurfaceCacheHistoryTap(history * tap_weight, tap_weight);
}

fn surface_cache_history_repair_candidate(
    tap_coord: vec2<i32>,
    resolution: vec2<u32>,
    current_position: vec3<f32>,
    current_normal: vec3<f32>,
    current_linear_depth: f32,
    tap_offset: vec2<i32>,
    inverse_depth: f32,
    repair_normal_threshold: f32,
    repair_depth_threshold: f32,
    repair_plane_threshold: f32,
    search_radius_denominator: f32,
    view_index: u32
) -> SurfaceCacheHistoryRepair {
    if (
        tap_coord.x < 0 || tap_coord.y < 0 ||
        tap_coord.x >= i32(resolution.x) ||
        tap_coord.y >= i32(resolution.y)
    ) {
        return SurfaceCacheHistoryRepair(vec4<f32>(0.0), 1e20);
    }

    // Repair runs on misses, where most candidates are empty. Keep dependent
    // texture reads behind progressively cheaper rejection tests.
    let history = textureLoad(history_diffuse, tap_coord, 0);
    if (history.w <= 0.0) {
        return SurfaceCacheHistoryRepair(vec4<f32>(0.0), 1e20);
    }
    let previous_normal_data = textureLoad(
        prev_gbuffer_normal,
        tap_coord,
        0
    ).xyz;
    if (dot(previous_normal_data, previous_normal_data) <= 1e-8) {
        return SurfaceCacheHistoryRepair(vec4<f32>(0.0), 1e20);
    }
    // The length check above guarantees normalize cannot take the zero path.
    let previous_normal = normalize(previous_normal_data);
    let normal_alignment = dot(current_normal, previous_normal);
    if (normal_alignment < repair_normal_threshold) {
        return SurfaceCacheHistoryRepair(vec4<f32>(0.0), 1e20);
    }
    let previous_depth = textureLoad(prev_depth_texture, tap_coord, 0).r;
    if (previous_depth >= 1.0) {
        return SurfaceCacheHistoryRepair(vec4<f32>(0.0), 1e20);
    }

    let previous_position = reconstruct_prev_world_position(
        coord_to_uv(tap_coord, resolution),
        previous_depth,
        view_index
    );
    let previous_linear_depth = abs(
        (view_buffer[view_index].view_matrix * vec4<f32>(previous_position, 1.0)).z
    );
    let relative_depth_delta = abs(
        previous_linear_depth - current_linear_depth
    ) * inverse_depth;
    let position_delta = previous_position - current_position;
    let relative_plane_delta = abs(dot(position_delta, current_normal)) *
        inverse_depth;
    if (
        relative_depth_delta > repair_depth_threshold ||
        relative_plane_delta > repair_plane_threshold
    ) {
        return SurfaceCacheHistoryRepair(vec4<f32>(0.0), 1e20);
    }
    let relative_position_delta = length(position_delta) * inverse_depth;
    if (relative_position_delta > repair_depth_threshold) {
        return SurfaceCacheHistoryRepair(vec4<f32>(0.0), 1e20);
    }

    let depth_score = relative_depth_delta / repair_depth_threshold;
    let position_score = relative_position_delta / repair_depth_threshold;
    let plane_score = relative_plane_delta / repair_plane_threshold;
    let normal_score = (1.0 - normal_alignment) /
        max(1.0 - repair_normal_threshold, 1e-3);
    // Distance is needed only for candidates that survived every geometry test.
    let spatial_score = length(vec2<f32>(tap_offset)) /
        search_radius_denominator;
    return SurfaceCacheHistoryRepair(
        history,
        depth_score + position_score + plane_score +
            normal_score + spatial_score * 0.25
    );
}

// Reprojection lands on the previous occluder for a true disocclusion. Search
// a sparse 5x5 footprint around that location for nearby history belonging to
// the newly exposed surface. This runs only when the normal four taps fail.
fn surface_cache_repair_history(
    previous_base: vec2<i32>,
    resolution: vec2<u32>,
    current_position: vec3<f32>,
    current_normal: vec3<f32>,
    current_linear_depth: f32,
    search_radius: i32,
    view_index: u32
) -> SurfaceCacheHistoryRepair {
    var best = SurfaceCacheHistoryRepair(vec4<f32>(0.0), 1e20);
    let inverse_depth = 1.0 / max(current_linear_depth, 1e-3);
    let repair_normal_threshold = max(
        temporal_params.normal_threshold,
        0.95
    );
    let repair_depth_threshold = max(
        temporal_params.depth_threshold * 2.0,
        0.01
    );
    let repair_plane_threshold = max(
        temporal_params.depth_threshold * 0.5,
        0.005
    );
    let search_radius_denominator = max(f32(search_radius), 1.0);
    for (var tap_y = -2; tap_y <= 2; tap_y = tap_y + 1) {
        for (var tap_x = -2; tap_x <= 2; tap_x = tap_x + 1) {
            if (tap_x == 0 && tap_y == 0) {
                continue;
            }
            let tap_offset = vec2<i32>(
                (tap_x * search_radius) / 2,
                (tap_y * search_radius) / 2
            );
            let candidate = surface_cache_history_repair_candidate(
                previous_base + tap_offset,
                resolution,
                current_position,
                current_normal,
                current_linear_depth,
                tap_offset,
                inverse_depth,
                repair_normal_threshold,
                repair_depth_threshold,
                repair_plane_threshold,
                search_radius_denominator,
                view_index
            );
            if (candidate.score < best.score) {
                best = candidate;
            }
        }
    }
    return best;
}

// Surface-cache discontinuities are several pixels wide, so adjacent-pixel
// filtering barely touches them. A sparse kernel reaches across roughly one
// cache cell while the depth and normal tests keep unrelated surfaces apart.
fn surface_cache_reconstruct_current(
    coord: vec2<i32>,
    resolution: vec2<u32>,
    center: vec4<f32>,
    current_normal: vec3<f32>,
    current_depth: f32,
    current_linear_depth: f32,
    view_index: u32,
    collect_luminance: bool
) -> SurfaceCacheCurrentEstimate {
    let tap_stride = max(i32(temporal_params.spatial_filter_radius + 0.5), 1);
    let kernel = vec3<f32>(0.27901, 0.44198, 0.27901);
    let depth_sigma = max(temporal_params.depth_threshold * 0.5, 1e-4);
    let maximum_coord = vec2<i32>(resolution) - vec2<i32>(1);

    var color_sum = vec3<f32>(0.0);
    var color_weight_sum = 0.0;
    var confidence_sum = 0.0;
    var geometry_weight_sum = 0.0;
    var luminance_sum = 0.0;
    var luminance_squared_sum = 0.0;
    var luminance_minimum = 1e20;
    var luminance_maximum = 0.0;
    var valid_tap_count = 0.0;

    for (var tap_y = -1; tap_y <= 1; tap_y = tap_y + 1) {
        for (var tap_x = -1; tap_x <= 1; tap_x = tap_x + 1) {
            let tap_coord = clamp(
                coord + vec2<i32>(tap_x, tap_y) * tap_stride,
                vec2<i32>(0),
                maximum_coord
            );
            let center_tap = tap_x == 0 && tap_y == 0;
            var tap = center;
            if (!center_tap) {
                tap = textureLoad(current_diffuse, tap_coord, 0);
            }
            if (tap.w <= 0.0) {
                continue;
            }

            var tap_normal = current_normal;
            if (!center_tap) {
                let tap_normal_data = textureLoad(gbuffer_normal, tap_coord, 0).xyz;
                if (dot(tap_normal_data, tap_normal_data) <= 1e-8) {
                    continue;
                }
                tap_normal = normalize(tap_normal_data);
            }
            let normal_alignment = dot(current_normal, tap_normal);
            if (normal_alignment < temporal_params.normal_threshold) {
                continue;
            }

            var tap_depth = current_depth;
            if (!center_tap) {
                tap_depth = textureLoad(depth_texture, tap_coord, 0).r;
            }
            if (tap_depth >= 1.0) {
                continue;
            }
            var tap_linear_depth = current_linear_depth;
            if (!center_tap) {
                let tap_position = reconstruct_world_position(
                    coord_to_uv(tap_coord, resolution),
                    tap_depth,
                    view_index
                );
                tap_linear_depth = abs(
                    (view_buffer[view_index].view_matrix * vec4<f32>(tap_position, 1.0)).z
                );
            }
            let relative_depth_delta = abs(
                tap_linear_depth - current_linear_depth
            ) / max(current_linear_depth, 1e-3);
            if (relative_depth_delta > temporal_params.depth_threshold) {
                continue;
            }

            let kernel_weight = kernel[u32(tap_x + 1)] * kernel[u32(tap_y + 1)];
            let normalized_depth_delta = relative_depth_delta / depth_sigma;
            let depth_weight = exp(
                -0.5 * normalized_depth_delta * normalized_depth_delta
            );
            let normal_weight = smoothstep(
                temporal_params.normal_threshold,
                1.0,
                normal_alignment
            );
            let geometry_weight = kernel_weight * depth_weight * normal_weight;
            let confidence_weight = clamp(tap.w / 8.0, 0.05, 1.0);
            let color_weight = geometry_weight * confidence_weight;
            color_sum += tap.xyz * color_weight;
            color_weight_sum += color_weight;
            confidence_sum += tap.w * geometry_weight;
            geometry_weight_sum += geometry_weight;

            if (collect_luminance && tap.w >= 2.0) {
                let tap_luminance = luminance(tap.xyz);
                luminance_sum += tap_luminance;
                luminance_squared_sum += tap_luminance * tap_luminance;
                luminance_minimum = min(luminance_minimum, tap_luminance);
                luminance_maximum = max(luminance_maximum, tap_luminance);
                valid_tap_count += 1.0;
            }
        }
    }

    var reconstructed = center;
    if (color_weight_sum > 1e-5 && geometry_weight_sum > 1e-5) {
        reconstructed = vec4<f32>(
            color_sum / color_weight_sum,
            confidence_sum / geometry_weight_sum
        );
    }
    return SurfaceCacheCurrentEstimate(
        reconstructed,
        luminance_minimum,
        luminance_maximum,
        luminance_sum,
        luminance_squared_sum,
        valid_tap_count
    );
}

fn surface_cache_reliability(sample_count: f32) -> f32 {
    // Converged cache taps dominate this path. Avoid a square root whenever
    // clamp would select either endpoint exactly.
    if (sample_count >= 32.0) {
        return 1.0;
    }
    if (sample_count <= 0.5) {
        return 0.125;
    }
    return sqrt(sample_count / 32.0);
}

// Recurrently filter the complete valid history over a sparse, geometry-aware
// footprint. Low-age history receives the configured maximum blur while mature
// history retains the minimum, so converged GI still sheds residual noise.
// Keeping the center age prevents neighboring mature pixels from prematurely
// weakening the filter on a newly exposed surface.
fn surface_cache_recurrent_blur_history(
    previous_pixel: vec2<f32>,
    resolution: vec2<u32>,
    current_position: vec3<f32>,
    current_normal: vec3<f32>,
    current_linear_depth: f32,
    view_index: u32,
    history: vec4<f32>
) -> vec4<f32> {
    let blur_history_frames = temporal_params.recurrent_blur_history_frames;
    let maximum_radius = temporal_params.recurrent_blur_max_radius;
    let minimum_strength = temporal_params.recurrent_blur_min_strength;
    let maximum_strength = temporal_params.recurrent_blur_max_strength;
    if (maximum_radius < 0.5 || maximum_strength <= 0.0) {
        return history;
    }

    let history_progress = clamp(
        (history.w - 1.0) / max(blur_history_frames - 1.0, 1.0),
        0.0,
        1.0
    );
    let convergence = smoothstep(0.0, 1.0, history_progress);
    let blur_strength = mix(maximum_strength, minimum_strength, convergence);
    let tap_stride = max(i32(ceil(maximum_radius * blur_strength)), 1);
    let previous_center = vec2<i32>(floor(previous_pixel + vec2<f32>(0.5)));
    let kernel = vec3<f32>(0.27901, 0.44198, 0.27901);
    let center_weight = kernel.y * kernel.y;
    let center_reliability = surface_cache_reliability(history.w);
    var color_sum = history.xyz * center_weight * center_reliability;
    var color_weight_sum = center_weight * center_reliability;

    for (var tap_y = -1; tap_y <= 1; tap_y = tap_y + 1) {
        for (var tap_x = -1; tap_x <= 1; tap_x = tap_x + 1) {
            if (tap_x == 0 && tap_y == 0) {
                continue;
            }
            let kernel_weight = kernel[u32(tap_x + 1)] *
                kernel[u32(tap_y + 1)];
            let tap = surface_cache_history_tap(
                previous_center + vec2<i32>(tap_x, tap_y) * tap_stride,
                kernel_weight,
                resolution,
                current_position,
                current_normal,
                current_linear_depth,
                true,
                view_index
            );
            if (tap.weight <= 1e-5) {
                continue;
            }
            let tap_history_frames = tap.value.w / tap.weight;
            let tap_reliability = surface_cache_reliability(tap_history_frames);
            color_sum += tap.value.xyz * tap_reliability;
            color_weight_sum += tap.weight * tap_reliability;
        }
    }

    if (color_weight_sum <= 1e-5) {
        return history;
    }
    let blurred_history = color_sum / color_weight_sum;
    return vec4<f32>(
        mix(history.xyz, blurred_history, blur_strength),
        history.w
    );
}

// When no temporal source exists, widen the current-frame reconstruction over
// four à-trous scales. The sparse footprint combines many independently
// gathered cache cells without paying this cost on temporally stable pixels.
fn surface_cache_reconstruct_disocclusion(
    coord: vec2<i32>,
    resolution: vec2<u32>,
    current_position: vec3<f32>,
    current_normal: vec3<f32>,
    current_linear_depth: f32,
    view_index: u32,
    center: vec4<f32>
) -> vec4<f32> {
    let radii = array<i32, 4>(2, 4, 8, 16);
    var color_sum = vec3<f32>(0.0);
    var color_weight_sum = 0.0;
    var sample_count_sum = 0.0;
    var geometry_weight_sum = 0.0;
    let inverse_depth = 1.0 / max(current_linear_depth, 1e-3);
    let plane_threshold = temporal_params.depth_threshold * 0.5;

    for (var scale_index = 0u; scale_index < 4u; scale_index = scale_index + 1u) {
        let radius = radii[scale_index];
        let scale_weight = 1.0 / (1.0 + f32(scale_index));
        for (var tap_y = -1; tap_y <= 1; tap_y = tap_y + 1) {
            for (var tap_x = -1; tap_x <= 1; tap_x = tap_x + 1) {
                if (tap_x == 0 && tap_y == 0) {
                    continue;
                }
                let tap_coord = coord + vec2<i32>(tap_x, tap_y) * radius;
                if (
                    tap_coord.x < 0 || tap_coord.y < 0 ||
                    tap_coord.x >= i32(resolution.x) ||
                    tap_coord.y >= i32(resolution.y)
                ) {
                    continue;
                }

                let tap = textureLoad(current_diffuse, tap_coord, 0);
                if (tap.w <= 0.0) {
                    continue;
                }
                let tap_normal_data = textureLoad(
                    gbuffer_normal,
                    tap_coord,
                    0
                ).xyz;
                if (dot(tap_normal_data, tap_normal_data) <= 1e-8) {
                    continue;
                }
                let tap_normal = normalize(tap_normal_data);
                let normal_alignment = dot(current_normal, tap_normal);
                if (normal_alignment < temporal_params.normal_threshold) {
                    continue;
                }

                let tap_depth = textureLoad(depth_texture, tap_coord, 0).r;
                if (tap_depth >= 1.0) {
                    continue;
                }
                let tap_position = reconstruct_world_position(
                    coord_to_uv(tap_coord, resolution),
                    tap_depth,
                    view_index
                );
                let tap_linear_depth = abs(
                    (view_buffer[view_index].view_matrix * vec4<f32>(tap_position, 1.0)).z
                );
                let relative_depth_delta = abs(
                    tap_linear_depth - current_linear_depth
                ) * inverse_depth;
                let relative_plane_delta = abs(dot(
                    tap_position - current_position,
                    current_normal
                )) * inverse_depth;
                if (
                    relative_depth_delta > temporal_params.depth_threshold ||
                    relative_plane_delta > plane_threshold
                ) {
                    continue;
                }

                let depth_amount = relative_depth_delta /
                    max(temporal_params.depth_threshold, 1e-4);
                let depth_weight = exp(-2.0 * depth_amount * depth_amount);
                let normal_weight = smoothstep(
                    temporal_params.normal_threshold,
                    1.0,
                    normal_alignment
                );
                let geometry_weight = scale_weight * depth_weight * normal_weight;
                // Sample count is used only as a denoiser reliability signal;
                // it does not alter cache LOD selection or fallback behavior.
                let reliability = surface_cache_reliability(tap.w);
                let color_weight = geometry_weight * reliability;
                color_sum += tap.xyz * color_weight;
                color_weight_sum += color_weight;
                sample_count_sum += tap.w * geometry_weight;
                geometry_weight_sum += geometry_weight;
            }
        }
    }

    if (center.w > 0.0) {
        let center_reliability = surface_cache_reliability(center.w);
        color_sum += center.xyz * center_reliability;
        color_weight_sum += center_reliability;
        sample_count_sum += center.w;
        geometry_weight_sum += 1.0;
    }
    if (color_weight_sum <= 1e-5 || geometry_weight_sum <= 1e-5) {
        return center;
    }
    return vec4<f32>(
        color_sum / color_weight_sum,
        sample_count_sum / geometry_weight_sum
    );
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let resolution = textureDimensions(current_diffuse);
    if (gid.x >= resolution.x || gid.y >= resolution.y) {
        return;
    }

    let coord = vec2<i32>(gid.xy);
    let normal_data = textureLoad(gbuffer_normal, coord, 0).xyz;
    if (dot(normal_data, normal_data) <= 1e-8) {
        textureStore(output_diffuse, coord, vec4<f32>(0.0));
        return;
    }

    let view_index = u32(frame_info.view_index);
    let uv = coord_to_uv(coord, resolution);
    let current_depth = textureLoad(depth_texture, coord, 0).r;
    let current_position = reconstruct_world_position(
        uv,
        current_depth,
        view_index
    );
    let current_linear_depth = abs(
        (view_buffer[view_index].view_matrix * vec4<f32>(current_position, 1.0)).z
    );
    let current_normal = normalize(normal_data);
    let center = textureLoad(current_diffuse, coord, 0);
    let motion = textureLoad(motion_texture, coord, 0).xy;
    let previous_uv = uv + vec2<f32>(-0.5 * motion.x, 0.5 * motion.y);
    let previous_pixel = previous_uv * vec2<f32>(resolution) - vec2<f32>(0.5);
    let previous_base = vec2<i32>(floor(previous_pixel));
    let previous_fraction = fract(previous_pixel);
    let bilinear_weights = vec4<f32>(
        (1.0 - previous_fraction.x) * (1.0 - previous_fraction.y),
        previous_fraction.x * (1.0 - previous_fraction.y),
        (1.0 - previous_fraction.x) * previous_fraction.y,
        previous_fraction.x * previous_fraction.y
    );

    let tap_00 = surface_cache_history_tap(
        previous_base,
        bilinear_weights.x,
        resolution,
        current_position,
        current_normal,
        current_linear_depth,
        false,
        view_index
    );
    var history_sum = tap_00.value;
    var history_weight = tap_00.weight;
    let tap_10 = surface_cache_history_tap(
        previous_base + vec2<i32>(1, 0),
        bilinear_weights.y,
        resolution,
        current_position,
        current_normal,
        current_linear_depth,
        false,
        view_index
    );
    history_sum += tap_10.value;
    history_weight += tap_10.weight;
    let tap_01 = surface_cache_history_tap(
        previous_base + vec2<i32>(0, 1),
        bilinear_weights.z,
        resolution,
        current_position,
        current_normal,
        current_linear_depth,
        false,
        view_index
    );
    history_sum += tap_01.value;
    history_weight += tap_01.weight;
    let tap_11 = surface_cache_history_tap(
        previous_base + vec2<i32>(1, 1),
        bilinear_weights.w,
        resolution,
        current_position,
        current_normal,
        current_linear_depth,
        false,
        view_index
    );
    history_sum += tap_11.value;
    history_weight += tap_11.weight;
    var history = vec4<f32>(0.0);
    if (history_weight > 1e-5) {
        history = history_sum / history_weight;
    } else {
        let motion_distance_pixels = length(
            previous_pixel - vec2<f32>(coord)
        );
        let repair_radius = i32(clamp(
            ceil(motion_distance_pixels) + 2.0,
            4.0,
            16.0
        ));
        let repair = surface_cache_repair_history(
            previous_base,
            resolution,
            current_position,
            current_normal,
            current_linear_depth,
            repair_radius,
            view_index
        );
        if (repair.score < 1e19) {
            // Repaired history is a temporary bridge, not an authoritative
            // reprojection. A short age hands control back to the current cache
            // quickly and limits trails when the selected neighbor was imperfect.
            history = vec4<f32>(repair.value.xyz, min(repair.value.w, 4.0));
            history_weight = 1.0;
        }
    }

    let history_valid = history_weight > 1e-5;
    if (history_valid) {
        history = surface_cache_recurrent_blur_history(
            previous_pixel,
            resolution,
            current_position,
            current_normal,
            current_linear_depth,
            view_index,
            history
        );
    }
    // Luminance moments are consumed only by the history-clipping path. True
    // disocclusions skip that arithmetic while retaining the same reconstruction.
    let current_estimate = surface_cache_reconstruct_current(
        coord,
        resolution,
        center,
        current_normal,
        current_depth,
        current_linear_depth,
        view_index,
        history_valid
    );
    var current = current_estimate.value;
    if (!history_valid) {
        current = surface_cache_reconstruct_disocclusion(
            coord,
            resolution,
            current_position,
            current_normal,
            current_linear_depth,
            view_index,
            center
        );
    }

    let current_valid = current.w >= 2.0;
    if (!current_valid) {
        // Preserve valid reprojection while the current cache is still
        // underconverged, but do not turn a true disocclusion black when the
        // current cache is the only estimate.
        let retained_history = select(
            select(
                vec4<f32>(0.0),
                vec4<f32>(current.xyz, 1.0),
                current.w > 0.0
            ),
            vec4<f32>(history.xyz, max(history.w - 1.0, 0.0)),
            history_valid && history.w > 1.0
        );
        textureStore(output_diffuse, coord, retained_history);
        return;
    }
    if (!history_valid) {
        textureStore(output_diffuse, coord, vec4<f32>(current.xyz, 1.0));
        return;
    }

    // A relaxed luminance window rejects disocclusion outliers without
    // pulling converged history toward the noisy current cache every frame.
    let mean_luminance = current_estimate.luminance_sum /
        max(current_estimate.valid_tap_count, 1.0);
    let luminance_variance = max(
        current_estimate.luminance_squared_sum /
            max(current_estimate.valid_tap_count, 1.0) -
            mean_luminance * mean_luminance,
        0.0
    );
    let clip_margin = max(
        0.03,
        max(sqrt(luminance_variance) * 2.5, mean_luminance * 0.2)
    );
    let history_luminance = luminance(history.xyz);
    let clipped_luminance = clamp(
        history_luminance,
        max(0.0, current_estimate.luminance_minimum - clip_margin),
        current_estimate.luminance_maximum + clip_margin
    );
    let clipped_history = history.xyz *
        (clipped_luminance / max(history_luminance, 1e-4));
    let response = max(
        temporal_params.response,
        1.0 / (min(history.w, temporal_params.max_history_frames) + 1.0)
    );
    let result = mix(clipped_history, current.xyz, response);
    let history_frames = min(history.w + 1.0, temporal_params.max_history_frames);
    textureStore(output_diffuse, coord, vec4<f32>(result, history_frames));
}
