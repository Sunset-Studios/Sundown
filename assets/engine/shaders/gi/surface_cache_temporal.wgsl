#include "common.wgsl"

struct SurfaceCacheTemporalParams {
    response: f32,
    max_history_frames: f32,
    depth_threshold: f32,
    normal_threshold: f32,
};

struct SurfaceCacheHistoryTap {
    value: vec4<f32>,
    weight: f32,
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
    current_normal: vec3<f32>,
    current_linear_depth: f32
) -> SurfaceCacheHistoryTap {
    if (
        tap_coord.x < 0 || tap_coord.y < 0 ||
        tap_coord.x >= i32(resolution.x) ||
        tap_coord.y >= i32(resolution.y)
    ) {
        return SurfaceCacheHistoryTap(vec4<f32>(0.0), 0.0);
    }

    let previous_normal_data = textureLoad(prev_gbuffer_normal, tap_coord, 0);
    let previous_depth = textureLoad(prev_depth_texture, tap_coord, 0).r;
    if (
        dot(previous_normal_data.xyz, previous_normal_data.xyz) <= 1e-8 ||
        previous_depth >= 1.0
    ) {
        return SurfaceCacheHistoryTap(vec4<f32>(0.0), 0.0);
    }

    let view_index = u32(frame_info.view_index);
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
    let previous_normal = safe_normalize(previous_normal_data.xyz);
    if (
        relative_depth_delta > temporal_params.depth_threshold ||
        dot(current_normal, previous_normal) < temporal_params.normal_threshold
    ) {
        return SurfaceCacheHistoryTap(vec4<f32>(0.0), 0.0);
    }

    let history = textureLoad(history_diffuse, tap_coord, 0);
    if (history.w <= 0.0) {
        return SurfaceCacheHistoryTap(vec4<f32>(0.0), 0.0);
    }
    return SurfaceCacheHistoryTap(history * tap_weight, tap_weight);
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let resolution = textureDimensions(current_diffuse);
    if (gid.x >= resolution.x || gid.y >= resolution.y) {
        return;
    }

    let coord = vec2<i32>(gid.xy);
    let current = textureLoad(current_diffuse, coord, 0);
    let normal_data = textureLoad(gbuffer_normal, coord, 0);
    if (dot(normal_data.xyz, normal_data.xyz) <= 1e-8) {
        textureStore(output_diffuse, coord, vec4<f32>(0.0));
        return;
    }

    let view_index = u32(frame_info.view_index);
    let uv = coord_to_uv(coord, resolution);
    let current_position = reconstruct_world_position(
        uv,
        textureLoad(depth_texture, coord, 0).r,
        view_index
    );
    let current_linear_depth = abs(
        (view_buffer[view_index].view_matrix * vec4<f32>(current_position, 1.0)).z
    );
    let current_normal = safe_normalize(normal_data.xyz);
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
        current_normal,
        current_linear_depth
    );
    let tap_10 = surface_cache_history_tap(
        previous_base + vec2<i32>(1, 0),
        bilinear_weights.y,
        resolution,
        current_normal,
        current_linear_depth
    );
    let tap_01 = surface_cache_history_tap(
        previous_base + vec2<i32>(0, 1),
        bilinear_weights.z,
        resolution,
        current_normal,
        current_linear_depth
    );
    let tap_11 = surface_cache_history_tap(
        previous_base + vec2<i32>(1, 1),
        bilinear_weights.w,
        resolution,
        current_normal,
        current_linear_depth
    );
    let history_weight = tap_00.weight + tap_10.weight +
        tap_01.weight + tap_11.weight;
    var history = vec4<f32>(0.0);
    if (history_weight > 1e-5) {
        history = (tap_00.value + tap_10.value + tap_01.value + tap_11.value) /
            history_weight;
    }

    let current_valid = current.w >= 2.0;
    let history_valid = history_weight > 1e-5;
    if (!current_valid) {
        let retained_history = select(
            vec4<f32>(0.0),
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
    var luminance_sum = 0.0;
    var luminance_squared_sum = 0.0;
    var luminance_minimum = 1e20;
    var luminance_maximum = 0.0;
    var valid_tap_count = 0.0;
    for (var tap_y = -1; tap_y <= 1; tap_y = tap_y + 1) {
        for (var tap_x = -1; tap_x <= 1; tap_x = tap_x + 1) {
            let tap_coord = clamp(
                coord + vec2<i32>(tap_x, tap_y),
                vec2<i32>(0),
                vec2<i32>(resolution) - vec2<i32>(1)
            );
            let tap = textureLoad(current_diffuse, tap_coord, 0);
            let tap_normal = textureLoad(gbuffer_normal, tap_coord, 0).xyz;
            if (
                tap.w < 2.0 ||
                dot(tap_normal, tap_normal) <= 1e-8 ||
                dot(current_normal, safe_normalize(tap_normal)) <
                    temporal_params.normal_threshold
            ) {
                continue;
            }
            let tap_luminance = luminance(tap.xyz);
            luminance_sum += tap_luminance;
            luminance_squared_sum += tap_luminance * tap_luminance;
            luminance_minimum = min(luminance_minimum, tap_luminance);
            luminance_maximum = max(luminance_maximum, tap_luminance);
            valid_tap_count += 1.0;
        }
    }

    let mean_luminance = luminance_sum / max(valid_tap_count, 1.0);
    let luminance_variance = max(
        luminance_squared_sum / max(valid_tap_count, 1.0) -
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
        max(0.0, luminance_minimum - clip_margin),
        luminance_maximum + clip_margin
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
