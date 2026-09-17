#include "common.wgsl"

struct SurfaceCacheTemporalParams {
    response: f32,
    max_history_frames: f32,
    depth_threshold: f32,
    normal_threshold: f32,
    spatial_filter_radius: f32,
    _padding0: f32,
};

struct SurfaceCacheHistoryTap {
    value: vec4<f32>,
    weight: f32,
};

@group(1) @binding(0) var<uniform> temporal_params: SurfaceCacheTemporalParams;
@group(1) @binding(1) var history_diffuse: texture_2d<f32>;
@group(1) @binding(2) var depth_texture: texture_2d<f32>;
@group(1) @binding(3) var prev_depth_texture: texture_2d<f32>;
@group(1) @binding(4) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(5) var prev_gbuffer_normal: texture_2d<f32>;
@group(1) @binding(6) var motion_texture: texture_2d<f32>;
@group(1) @binding(7) var output_history: texture_storage_2d<rgba16float, write>;

fn surface_cache_history_tap(
    tap_coord: vec2<i32>,
    tap_weight: f32,
    resolution: vec2<u32>,
    current_position: vec3<f32>,
    current_normal: vec3<f32>,
    current_linear_depth: f32,
    view_index: u32
) -> SurfaceCacheHistoryTap {
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
    if (relative_depth_delta > temporal_params.depth_threshold) {
        return SurfaceCacheHistoryTap(vec4<f32>(0.0), 0.0);
    }

    return SurfaceCacheHistoryTap(history * tap_weight, tap_weight);
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let resolution = textureDimensions(depth_texture);
    if (gid.x >= resolution.x || gid.y >= resolution.y) {
        return;
    }

    let coord = vec2<i32>(gid.xy);
    let current_normal = textureLoad(gbuffer_normal, coord, 0).xyz;
    if (dot(current_normal, current_normal) <= 1e-8) {
        textureStore(output_history, coord, vec4<f32>(0.0));
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
        view_index
    );
    history_sum += tap_11.value;
    history_weight += tap_11.weight;

    var history = vec4<f32>(0.0);
    if (history_weight > 1e-5) {
        history = history_sum / history_weight;
    }
    textureStore(output_history, coord, history);
}
