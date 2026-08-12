#include "common.wgsl"

struct VBAOSettings {
    radius: f32,
    strength: f32,
    bias: f32,
    slice_count: f32,
    sample_count: f32,
    thickness: f32,
    temporal_response: f32,
    history_valid: f32,
};

@group(1) @binding(0) var ao_src: texture_2d<f32>;
@group(1) @binding(1) var history_ao_tex: texture_2d<f32>;
@group(1) @binding(2) var motion_tex: texture_2d<f32>;
@group(1) @binding(3) var depth_tex: texture_2d<f32>;
@group(1) @binding(4) var prev_depth_tex: texture_2d<f32>;
@group(1) @binding(5) var normal_tex: texture_2d<f32>;
@group(1) @binding(6) var prev_normal_tex: texture_2d<f32>;
@group(1) @binding(7) var ao_output: texture_storage_2d<r32float, write>;
@group(1) @binding(8) var<uniform> settings: VBAOSettings;

const NORMAL_HISTORY_THRESHOLD = 0.8;
const DEPTH_SIGMA_SCALE = 0.015;
const DEPTH_SIGMA_MIN = 0.01;
const POSITION_REJECT_MIN = 0.03;
const POSITION_REJECT_DISTANCE_SCALE = 0.003;
const HISTORY_SIGMA_SCALE = 1.5;
const HISTORY_CLAMP_MIN = 0.015;
const VELOCITY_RESPONSE_SCALE = 8.0;

fn view_z_from_depth(depth: f32, view_index: u32) -> f32 {
    let projection = view_buffer[view_index].projection_matrix;
    let denominator = depth * projection[2][3] - projection[2][2];
    let safe_denominator = select(
        -max(abs(denominator), 1e-6),
        max(abs(denominator), 1e-6),
        denominator >= 0.0
    );
    return (projection[3][2] - depth * projection[3][3]) / safe_denominator;
}

fn sample_history_bilinear(prev_uv: vec2<f32>, dims: vec2<u32>) -> f32 {
    let history_pos = prev_uv * vec2<f32>(dims) - vec2<f32>(0.5);
    let base_coord = vec2<i32>(floor(history_pos));
    let fraction = fract(history_pos);
    let max_coord = vec2<i32>(dims) - vec2<i32>(1);

    let p00 = clamp(base_coord, vec2<i32>(0), max_coord);
    let p10 = clamp(base_coord + vec2<i32>(1, 0), vec2<i32>(0), max_coord);
    let p01 = clamp(base_coord + vec2<i32>(0, 1), vec2<i32>(0), max_coord);
    let p11 = clamp(base_coord + vec2<i32>(1, 1), vec2<i32>(0), max_coord);

    let h00 = textureLoad(history_ao_tex, p00, 0).r;
    let h10 = textureLoad(history_ao_tex, p10, 0).r;
    let h01 = textureLoad(history_ao_tex, p01, 0).r;
    let h11 = textureLoad(history_ao_tex, p11, 0).r;

    return mix(mix(h00, h10, fraction.x), mix(h01, h11, fraction.x), fraction.y);
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let full_resolution = textureDimensions(normal_tex);
    if (gid.x >= full_resolution.x || gid.y >= full_resolution.y) {
        return;
    }

    let coord = vec2<i32>(gid.xy);
    let uv = coord_to_uv(coord, full_resolution);
    let center_normal_raw = textureLoad(normal_tex, coord, 0).xyz;
    let center_normal_len = length(center_normal_raw);
    let center_depth = textureLoad(depth_tex, coord, 0).r;
    if (center_depth >= 0.9999 || center_normal_len < 1e-6) {
        textureStore(ao_output, coord, vec4<f32>(1.0, 0.0, 0.0, 1.0));
        return;
    }

    let view_index = u32(frame_info.view_index);
    let trace_resolution = textureDimensions(ao_src);
    let center_normal = center_normal_raw / center_normal_len;
    let center_view_z = view_z_from_depth(center_depth, view_index);
    let depth_sigma = max(DEPTH_SIGMA_MIN, abs(center_view_z) * DEPTH_SIGMA_SCALE);
    let trace_position = uv * vec2<f32>(trace_resolution) - vec2<f32>(0.5);
    let trace_center = vec2<i32>(floor(trace_position + vec2<f32>(0.5)));
    let max_trace_coord = vec2<i32>(trace_resolution) - vec2<i32>(1);

    // Spatial reconstruction and temporal accumulation share one pass. Besides
    // avoiding a full-resolution intermediate, this reuses the same neighborhood
    // for edge-aware denoising, variance estimation, and history clipping.
    var ao_sum = 0.0;
    var ao_sum_sq = 0.0;
    var weight_sum = 0.0;
    var ao_min = 1.0;
    var ao_max = 0.0;

    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let tap_coord = clamp(
                trace_center + vec2<i32>(x, y),
                vec2<i32>(0),
                max_trace_coord
            );
            let tap_uv = coord_to_uv(tap_coord, trace_resolution);
            let tap_full_coord = uv_to_coord(tap_uv, full_resolution);
            let tap_normal_raw = textureLoad(normal_tex, tap_full_coord, 0).xyz;
            let tap_normal_len = length(tap_normal_raw);
            let tap_depth = textureLoad(depth_tex, tap_full_coord, 0).r;
            if (tap_depth >= 0.9999 || tap_normal_len < 1e-6) {
                continue;
            }

            let tap_ao = textureLoad(ao_src, tap_coord, 0).r;
            let tap_normal = tap_normal_raw / tap_normal_len;
            let tap_view_z = view_z_from_depth(tap_depth, view_index);
            let trace_delta = vec2<f32>(tap_coord) - trace_position;
            let spatial_weight = 1.0 / (1.0 + dot(trace_delta, trace_delta));
            let depth_ratio = abs(tap_view_z - center_view_z) / depth_sigma;
            let depth_weight = 1.0 / (1.0 + depth_ratio * depth_ratio);
            let normal_dot = max(dot(center_normal, tap_normal), 0.0);
            let normal_dot2 = normal_dot * normal_dot;
            let normal_dot4 = normal_dot2 * normal_dot2;
            let normal_dot8 = normal_dot4 * normal_dot4;
            let normal_weight = normal_dot8 * normal_dot8;
            let weight = spatial_weight * depth_weight * normal_weight;

            ao_sum += tap_ao * weight;
            ao_sum_sq += tap_ao * tap_ao * weight;
            weight_sum += weight;
            if (weight > 1e-6) {
                ao_min = min(ao_min, tap_ao);
                ao_max = max(ao_max, tap_ao);
            }
        }
    }

    if (weight_sum <= 1e-6) {
        textureStore(ao_output, coord, vec4<f32>(1.0, 0.0, 0.0, 1.0));
        return;
    }

    let current_ao = clamp(ao_sum / weight_sum, 0.0, 1.0);
    let ao_variance = max(ao_sum_sq / weight_sum - current_ao * current_ao, 0.0);
    let ao_stddev = sqrt(ao_variance);
    let clamp_extent = max(HISTORY_CLAMP_MIN, ao_stddev * HISTORY_SIGMA_SCALE);
    let history_min = max(0.0, min(ao_min, current_ao - clamp_extent));
    let history_max = min(1.0, max(ao_max, current_ao + clamp_extent));

    let motion = textureLoad(motion_tex, coord, 0).xy;
    let prev_uv = uv + vec2<f32>(-0.5 * motion.x, 0.5 * motion.y);
    let prev_in_bounds = all(prev_uv >= vec2<f32>(0.0)) && all(prev_uv <= vec2<f32>(1.0));
    let prev_coord = uv_to_coord(prev_uv, full_resolution);
    let prev_depth = textureLoad(prev_depth_tex, prev_coord, 0).r;
    let prev_normal_raw = textureLoad(prev_normal_tex, prev_coord, 0).xyz;
    let prev_normal_len = length(prev_normal_raw);

    var surface_history_valid = false;
    if (prev_in_bounds && prev_depth < 0.9999 && prev_normal_len > 1e-6) {
        let current_position = reconstruct_world_position(uv, center_depth, view_index);
        let previous_position = reconstruct_prev_world_position(prev_uv, prev_depth, view_index);
        let view_distance = distance(view_buffer[view_index].view_position.xyz, current_position);
        let position_threshold = max(
            POSITION_REJECT_MIN,
            view_distance * POSITION_REJECT_DISTANCE_SCALE
        );
        let position_valid = distance(current_position, previous_position) <= position_threshold;
        let normal_valid = dot(center_normal, prev_normal_raw / prev_normal_len) >= NORMAL_HISTORY_THRESHOLD;
        surface_history_valid = position_valid && normal_valid;
    }

    let history_available = settings.history_valid > 0.5 && surface_history_valid;
    let history_raw = sample_history_bilinear(prev_uv, full_resolution);
    let history_clamped = clamp(history_raw, history_min, history_max);
    let velocity_response = clamp(length(motion) * VELOCITY_RESPONSE_SCALE, 0.0, 1.0);
    let response = max(clamp(settings.temporal_response, 0.01, 1.0), velocity_response);
    let temporal_ao = mix(history_clamped, current_ao, response);
    let resolved_ao = select(current_ao, temporal_ao, history_available);

    textureStore(ao_output, coord, vec4<f32>(resolved_ao, 0.0, 0.0, 1.0));
}
