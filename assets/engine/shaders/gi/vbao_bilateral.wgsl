#include "common.wgsl"

struct VBAOSettings {
    radius: f32,
    bias: f32,
    slice_count: f32,
    sample_count: f32,
    max_radius_px: f32,
    thickness: f32,
    temporal_response: f32,
    denoise_radius: f32,
    denoise_position_sigma: f32,
    denoise_normal_power: f32,
    denoise_ao_sigma: f32,
    denoise_direction: vec2<f32>,
    denoise_radius_px: f32,
};

@group(1) @binding(0) var depth_tex: texture_2d<f32>;
@group(1) @binding(1) var normal_tex: texture_2d<f32>;
@group(1) @binding(2) var ao_src: texture_2d<f32>;
@group(1) @binding(3) var ao_dst: texture_storage_2d<r32float, write>;
@group(1) @binding(4) var<uniform> settings: VBAOSettings;

fn gaussian(distance_sq: f32, sigma: f32) -> f32 {
    if (sigma <= 0.0) {
        return 1.0;
    }
    let denom = 2.0 * sigma * sigma;
    return exp(-distance_sq / denom);
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(ao_src);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let view_index = u32(frame_info.view_index);
    let full_dims = textureDimensions(normal_tex);
    let coord = vec2<i32>(gid.xy);
    let radius = max(0, i32(settings.denoise_radius_px + 0.5));
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) / vec2<f32>(f32(dims.x), f32(dims.y));
    let full_coord = uv_to_coord(uv, full_dims);

    let center_normal_raw = textureLoad(normal_tex, full_coord, 0).xyz;
    let center_normal_len = length(center_normal_raw);
    let center_ao = textureLoad(ao_src, coord, 0).r;

    if (center_normal_len < 1e-6) {
        textureStore(ao_dst, coord, vec4<f32>(center_ao, 0.0, 0.0, 1.0));
        return;
    }

    let center_normal = center_normal_raw / center_normal_len;
    let center_depth = textureLoad(depth_tex, full_coord, 0).r;
    let center_position = reconstruct_world_position(uv, center_depth, view_index);

    var weight_sum = 1.0;
    var ao_sum = center_ao;

    for (var step = -radius; step <= radius; step = step + 1) {
        let tap_offset = vec2<i32>(
            i32(round(f32(step) * settings.denoise_direction.x)),
            i32(round(f32(step) * settings.denoise_direction.y))
        );
        let tap_coord = vec2<i32>(
            clamp(coord.x + tap_offset.x, 0, i32(dims.x) - 1),
            clamp(coord.y + tap_offset.y, 0, i32(dims.y) - 1)
        );
        let tap_uv = (vec2<f32>(f32(tap_coord.x), f32(tap_coord.y)) + 0.5) /
            vec2<f32>(f32(dims.x), f32(dims.y));
        let tap_full_coord = uv_to_coord(tap_uv, full_dims);

        let tap_normal_raw = textureLoad(normal_tex, tap_full_coord, 0).xyz;
        let tap_normal_len = length(tap_normal_raw);
        if (tap_normal_len < 1e-6) {
            continue;
        }

        let tap_normal = tap_normal_raw / tap_normal_len;
        let tap_depth = textureLoad(depth_tex, tap_full_coord, 0).r;
        let tap_position = reconstruct_world_position(tap_uv, tap_depth, view_index);
        let tap_ao = textureLoad(ao_src, tap_coord, 0).r;

        let spatial_weight = gaussian(f32(step * step), max(1.0, settings.denoise_radius_px * 0.5));
        let position_delta = tap_position - center_position;
        let plane_distance = abs(dot(position_delta, center_normal));
        let position_weight = gaussian(plane_distance * plane_distance, settings.denoise_position_sigma);
        let normal_weight = pow(max(dot(center_normal, tap_normal), 0.0), max(settings.denoise_normal_power, 0.0));
        let ao_delta = tap_ao - center_ao;
        let ao_weight = gaussian(ao_delta * ao_delta, settings.denoise_ao_sigma);

        let weight = spatial_weight * position_weight * normal_weight * ao_weight;
        ao_sum += tap_ao * weight;
        weight_sum += weight;
    }

    let filtered_ao = clamp(ao_sum / max(weight_sum, 1e-5), 0.0, 1.0);

    textureStore(ao_dst, coord, vec4<f32>(filtered_ao, 0.0, 0.0, 1.0));
}


