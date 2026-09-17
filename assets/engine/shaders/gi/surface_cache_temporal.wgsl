#include "common.wgsl"

struct SurfaceCacheTemporalParams {
    response: f32,
    max_history_frames: f32,
    depth_threshold: f32,
    normal_threshold: f32,
    spatial_filter_radius: f32,
    _padding0: f32,
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
@group(1) @binding(2) var reprojected_history: texture_2d<f32>;
@group(1) @binding(3) var depth_texture: texture_2d<f32>;
@group(1) @binding(4) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(5) var output_diffuse: texture_storage_2d<rgba16float, write>;

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

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let resolution = textureDimensions(current_diffuse);
    if (gid.x >= resolution.x || gid.y >= resolution.y) {
        return;
    }

    let coord = vec2<i32>(gid.xy);
    let normal = textureLoad(gbuffer_normal, coord, 0).xyz;
    if (dot(normal, normal) <= 1e-8) {
        textureStore(output_diffuse, coord, vec4<f32>(0.0));
        return;
    }

    let view_index = u32(frame_info.view_index);
    let current_depth = textureLoad(depth_texture, coord, 0).r;
    let current_position = reconstruct_world_position(
        coord_to_uv(coord, resolution),
        current_depth,
        view_index
    );
    let current_linear_depth = abs(
        (view_buffer[view_index].view_matrix * vec4<f32>(current_position, 1.0)).z
    );
    let history = textureLoad(reprojected_history, coord, 0);
    let history_valid = history.w > 0.0;
    let current_estimate = surface_cache_reconstruct_current(
        coord,
        resolution,
        textureLoad(current_diffuse, coord, 0),
        normal,
        current_depth,
        current_linear_depth,
        view_index,
        history_valid
    );
    let current = current_estimate.value;

    let current_valid = current.w >= 2.0;
    if (!current_valid) {
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
