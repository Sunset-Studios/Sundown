#include "common.wgsl"

@group(1) @binding(0) var ao_src: texture_2d<f32>;
@group(1) @binding(1) var depth_tex: texture_2d<f32>;
@group(1) @binding(2) var normal_tex: texture_2d<f32>;
@group(1) @binding(3) var ao_output: texture_storage_2d<r32float, write>;

const NORMAL_WEIGHT_POWER = 16.0;
const POSITION_SIGMA_SCALE = 0.01;
const POSITION_SIGMA_MIN = 0.0025;

fn gaussian(distance_sq: f32, sigma: f32) -> f32 {
    if (sigma <= 0.0) {
        return 1.0;
    }
    let denom = 2.0 * sigma * sigma;
    return exp(-distance_sq / denom);
}

fn bilinear_weight(offset: vec2<i32>, frac: vec2<f32>) -> f32 {
    let wx = select(1.0 - frac.x, frac.x, offset.x == 1);
    let wy = select(1.0 - frac.y, frac.y, offset.y == 1);
    return wx * wy;
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let full_resolution = textureDimensions(normal_tex);
    if (gid.x >= full_resolution.x || gid.y >= full_resolution.y) {
        return;
    }

    let trace_resolution = textureDimensions(ao_src);
    let coord = vec2<i32>(gid.xy);
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) /
        vec2<f32>(f32(full_resolution.x), f32(full_resolution.y));

    let normal_raw = textureLoad(normal_tex, coord, 0).xyz;
    let normal_len = length(normal_raw);
    if (normal_len < 1e-6) {
        textureStore(ao_output, coord, vec4<f32>(1.0, 1.0, 1.0, 1.0));
        return;
    }

    let view_index = u32(frame_info.view_index);
    let center_normal = normal_raw / normal_len;
    let center_depth = textureLoad(depth_tex, coord, 0).r;
    let center_position = reconstruct_world_position(uv, center_depth, view_index);
    let trace_position = uv * vec2<f32>(f32(trace_resolution.x), f32(trace_resolution.y)) - vec2<f32>(0.5);
    let base = vec2<i32>(floor(trace_position));
    let frac = fract(trace_position);

    let camera_position = view_buffer[u32(frame_info.view_index)].view_position.xyz;
    let view_distance = distance(camera_position, center_position);
    let position_sigma = max(POSITION_SIGMA_MIN, view_distance * POSITION_SIGMA_SCALE);

    var ao_sum = 0.0;
    var weight_sum = 0.0;

    for (var oy = 0; oy <= 1; oy = oy + 1) {
        for (var ox = 0; ox <= 1; ox = ox + 1) {
            let offset = vec2<i32>(ox, oy);
            let tap = vec2<i32>(
                clamp(base.x + offset.x, 0, i32(trace_resolution.x) - 1),
                clamp(base.y + offset.y, 0, i32(trace_resolution.y) - 1)
            );
            let tap_ao = textureLoad(ao_src, tap, 0).r;
            let tap_uv = (vec2<f32>(f32(tap.x), f32(tap.y)) + 0.5) /
                vec2<f32>(f32(trace_resolution.x), f32(trace_resolution.y));
            let tap_full_coord = uv_to_coord(tap_uv, full_resolution);
            let tap_normal_raw = textureLoad(normal_tex, tap_full_coord, 0).xyz;
            let tap_normal_len = length(tap_normal_raw);
            if (tap_normal_len < 1e-6) {
                continue;
            }

            let tap_normal = tap_normal_raw / tap_normal_len;
            let tap_depth = textureLoad(depth_tex, tap_full_coord, 0).r;
            let tap_position = reconstruct_world_position(tap_uv, tap_depth, view_index);
            let spatial_weight = bilinear_weight(offset, frac);
            let plane_distance = abs(dot(tap_position - center_position, center_normal));
            let position_weight = gaussian(plane_distance * plane_distance, position_sigma);
            let normal_weight = pow(max(dot(center_normal, tap_normal), 0.0), NORMAL_WEIGHT_POWER);
            let weight = spatial_weight * position_weight * max(normal_weight, 1e-4);

            ao_sum += tap_ao * weight;
            weight_sum += weight;
        }
    }

    let resolved_ao = clamp(ao_sum / weight_sum, 0.0, 1.0);

    textureStore(ao_output, coord, vec4<f32>(resolved_ao, 0.0, 0.0, 1.0));
}

