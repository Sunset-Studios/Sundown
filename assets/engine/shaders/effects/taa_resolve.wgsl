#include "common.wgsl"

struct TAAParams {
    current_jitter_ndc: vec2<f32>,
    previous_jitter_ndc: vec2<f32>,
    feedback_history_valid: vec2<f32>,
};

@group(1) @binding(0) var current_color_tex: texture_2d<f32>;
@group(1) @binding(1) var history_color_tex: texture_2d<f32>;
@group(1) @binding(2) var motion_tex: texture_2d<f32>;
@group(1) @binding(3) var depth_tex: texture_2d<f32>;
@group(1) @binding(4) var prev_depth_tex: texture_2d<f32>;
@group(1) @binding(5) var normal_tex: texture_2d<f32>;
@group(1) @binding(6) var output_tex: texture_storage_2d<rgba16float, write>;
@group(1) @binding(7) var<uniform> params: TAAParams;

const FLT_EPSILON = 1e-6;
const AABB_EXPAND_FACTOR = 0.12;
const AABB_EXPAND_MIN = 0.01;
const POSITION_REJECT_MIN = 0.025;
const POSITION_REJECT_DISTANCE_SCALE = 0.002;
const VELOCITY_REJECT_SCALE = 10.0;
const FXAA_EDGE_THRESHOLD_MIN = 0.0312;
const FXAA_EDGE_THRESHOLD = 0.125;
const FXAA_REDUCE_MIN = 0.0078125;
const FXAA_REDUCE_MUL = 0.125;
const FXAA_SPAN_MAX = 8.0;
const SHARPEN_STRENGTH = 0.03;

fn clip_aabb(aabb_min: vec3<f32>, aabb_max: vec3<f32>, q: vec4<f32>) -> vec4<f32> {
    let center = 0.5 * (aabb_max + aabb_min);
    let extents = 0.5 * (aabb_max - aabb_min) + vec3<f32>(FLT_EPSILON);
    let offset = q.xyz - center;
    let unit = abs(offset / extents);
    let max_unit = max(unit.x, max(unit.y, unit.z));
    let clipped = center + offset / max_unit;
    return select(q, vec4<f32>(clipped, q.a), max_unit > 1.0);
}

fn ndc_jitter_to_uv(jitter_ndc: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(0.5 * jitter_ndc.x, -0.5 * jitter_ndc.y);
}

fn sample_current(uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(current_color_tex, clamped_sampler, uv, 0.0);
}

fn sample_current_rgb(uv: vec2<f32>) -> vec3<f32> {
    return sample_current(uv).rgb;
}

fn sample_motion(uv: vec2<f32>) -> vec2<f32> {
    return textureSampleLevel(motion_tex, clamped_sampler, uv, 0.0).xy;
}

fn sample_depth(uv: vec2<f32>) -> f32 {
    return textureSampleLevel(depth_tex, non_filtering_sampler, uv, 0.0).r;
}

fn sample_normal(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(normal_tex, clamped_sampler, uv, 0.0).xyz;
}

fn fxaa_current(uv: vec2<f32>, resolution: vec2<u32>) -> vec3<f32> {
    let inv_resolution = 1.0 / vec2<f32>(f32(resolution.x), f32(resolution.y));

    let rgb_m = sample_current_rgb(uv);
    let rgb_nw = sample_current_rgb(uv + vec2<f32>(-1.0, -1.0) * inv_resolution);
    let rgb_ne = sample_current_rgb(uv + vec2<f32>(1.0, -1.0) * inv_resolution);
    let rgb_sw = sample_current_rgb(uv + vec2<f32>(-1.0, 1.0) * inv_resolution);
    let rgb_se = sample_current_rgb(uv + vec2<f32>(1.0, 1.0) * inv_resolution);

    let luma_m = luminance(rgb_m);
    let luma_nw = luminance(rgb_nw);
    let luma_ne = luminance(rgb_ne);
    let luma_sw = luminance(rgb_sw);
    let luma_se = luminance(rgb_se);

    let luma_min = min(luma_m, min(min(luma_nw, luma_ne), min(luma_sw, luma_se)));
    let luma_max = max(luma_m, max(max(luma_nw, luma_ne), max(luma_sw, luma_se)));
    let luma_range = luma_max - luma_min;

    if (luma_range < max(FXAA_EDGE_THRESHOLD_MIN, luma_max * FXAA_EDGE_THRESHOLD)) {
        return rgb_m;
    }

    var dir = vec2<f32>(
        -((luma_nw + luma_ne) - (luma_sw + luma_se)),
        ((luma_nw + luma_sw) - (luma_ne + luma_se))
    );
    let dir_reduce = max(
        (luma_nw + luma_ne + luma_sw + luma_se) * (0.25 * FXAA_REDUCE_MUL),
        FXAA_REDUCE_MIN
    );
    let rcp_dir_min = 1.0 / (min(abs(dir.x), abs(dir.y)) + dir_reduce);
    dir = clamp(
        dir * rcp_dir_min,
        vec2<f32>(-FXAA_SPAN_MAX),
        vec2<f32>(FXAA_SPAN_MAX)
    ) * inv_resolution;

    let rgb_a = 0.5 * (
        sample_current_rgb(uv + dir * (1.0 / 3.0 - 0.5)) +
        sample_current_rgb(uv + dir * (2.0 / 3.0 - 0.5))
    );
    let rgb_b = 0.5 * rgb_a + 0.25 * (
        sample_current_rgb(uv + dir * -0.5) +
        sample_current_rgb(uv + dir * 0.5)
    );
    let luma_b = luminance(rgb_b);

    return select(rgb_b, rgb_a, luma_b < luma_min || luma_b > luma_max);
}

fn sharpen_resolve(color: vec3<f32>, uv: vec2<f32>, resolution: vec2<u32>) -> vec3<f32> {
    let inv_resolution = 1.0 / vec2<f32>(f32(resolution.x), f32(resolution.y));
    let center = sample_current_rgb(uv);
    let blur =
        center * 4.0 +
        sample_current_rgb(uv + vec2<f32>(1.0, 0.0) * inv_resolution) +
        sample_current_rgb(uv + vec2<f32>(-1.0, 0.0) * inv_resolution) +
        sample_current_rgb(uv + vec2<f32>(0.0, 1.0) * inv_resolution) +
        sample_current_rgb(uv + vec2<f32>(0.0, -1.0) * inv_resolution);
    let detail = center - blur * 0.125;
    return max(vec3<f32>(0.0), color + detail * SHARPEN_STRENGTH);
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let resolution = textureDimensions(current_color_tex);
    if (gid.x >= resolution.x || gid.y >= resolution.y) {
        return;
    }

    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let uv = coord_to_uv(coord, resolution);
    let current_jitter_uv = ndc_jitter_to_uv(params.current_jitter_ndc);
    let previous_jitter_uv = ndc_jitter_to_uv(params.previous_jitter_ndc);
    let current_sample_uv = uv + current_jitter_uv;
    let current_raw = sample_current(current_sample_uv);
    let current = vec4<f32>(fxaa_current(current_sample_uv, resolution), current_raw.a);

    var neighborhood_min = current.rgb;
    var neighborhood_max = current.rgb;

    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let tap_uv =
                current_sample_uv +
                vec2<f32>(f32(x), f32(y)) / vec2<f32>(f32(resolution.x), f32(resolution.y));
            let tap = sample_current(tap_uv).rgb;
            neighborhood_min = min(neighborhood_min, tap);
            neighborhood_max = max(neighborhood_max, tap);
        }
    }

    let motion = sample_motion(current_sample_uv);
    let prev_gbuffer_uv =
        current_sample_uv +
        vec2<f32>(-0.5 * motion.x, 0.5 * motion.y);
    let prev_history_uv = prev_gbuffer_uv - previous_jitter_uv;
    let prev_in_bounds =
        all(prev_history_uv >= vec2<f32>(0.0)) &&
        all(prev_history_uv <= vec2<f32>(1.0)) &&
        all(prev_gbuffer_uv >= vec2<f32>(0.0)) &&
        all(prev_gbuffer_uv <= vec2<f32>(1.0));

    let view_index = u32(frame_info.view_index);
    let depth = sample_depth(current_sample_uv);
    let prev_coord = uv_to_coord(prev_gbuffer_uv, resolution);
    let prev_depth = textureLoad(prev_depth_tex, prev_coord, 0).r;
    let finite_depth = depth < 0.9999 && prev_depth < 0.9999;
    let current_world_position = reconstruct_world_position(current_sample_uv, depth, view_index);
    let previous_world_position = reconstruct_prev_world_position(prev_gbuffer_uv, prev_depth, view_index);
    let view_distance = distance(view_buffer[view_index].view_position.xyz, current_world_position);
    let position_threshold = max(POSITION_REJECT_MIN, view_distance * POSITION_REJECT_DISTANCE_SCALE);
    let position_valid =
        !finite_depth ||
        distance(current_world_position, previous_world_position) <= position_threshold;

    let normal = sample_normal(current_sample_uv);
    let has_surface = length(normal) > 0.0;
    let history_available =
        params.feedback_history_valid.y > 0.5 &&
        prev_in_bounds &&
        position_valid &&
        (has_surface || depth < 0.9999);

    let history = textureSampleLevel(history_color_tex, clamped_sampler, prev_history_uv, 0.0);
    let extent = neighborhood_max - neighborhood_min;
    let margin = max(extent * AABB_EXPAND_FACTOR, vec3<f32>(AABB_EXPAND_MIN));
    let clipped_history = clip_aabb(
        neighborhood_min - margin,
        neighborhood_max + margin,
        history
    );

    let lum_current = luminance(current.rgb);
    let lum_history = luminance(clipped_history.rgb);
    let lum_delta = abs(lum_current - lum_history) / max(max(lum_current, lum_history), 0.00001);
    let velocity_alpha = clamp(length(motion) * VELOCITY_REJECT_SCALE, 0.0, 1.0);
    let luminance_alpha = clamp(lum_delta * 0.5, 0.0, 1.0);
    let base_feedback = clamp(params.feedback_history_valid.x, 0.5, 1.0);
    let feedback = clamp(max(base_feedback, max(velocity_alpha, luminance_alpha)), 0.5, 1.0);

    let resolved = mix(clipped_history.rgb, current.rgb, feedback);
    let temporal_color = select(current.rgb, resolved, history_available);
    let output_color = sharpen_resolve(temporal_color, current_sample_uv, resolution);

    textureStore(output_tex, coord, vec4<f32>(output_color, current_raw.a));
}
