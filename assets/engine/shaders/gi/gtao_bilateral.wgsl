#include "common.wgsl"

struct GTAODenoiseSettings {
    direction: vec2f,
    radius_px: f32,
    position_sigma: f32,
    normal_power: f32,
    ao_sigma: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(1) @binding(0) var position_tex: texture_2d<f32>;
@group(1) @binding(1) var normal_tex: texture_2d<f32>;
@group(1) @binding(2) var ao_src: texture_2d<f32>;
@group(1) @binding(3) var bent_src: texture_2d<f32>;
@group(1) @binding(4) var ao_dst: texture_storage_2d<r32float, write>;
@group(1) @binding(5) var bent_dst: texture_storage_2d<rgba16float, write>;
@group(1) @binding(6) var<uniform> settings: GTAODenoiseSettings;

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

    let coord = vec2<i32>(gid.xy);
    let radius = max(0, i32(settings.radius_px + 0.5));
    let center_normal_raw = textureLoad(normal_tex, coord, 0).xyz;
    let center_normal_len = length(center_normal_raw);
    let center_ao = textureLoad(ao_src, coord, 0).r;
    let center_bent = safe_normalize(textureLoad(bent_src, coord, 0).xyz);

    if (center_normal_len < 1e-6) {
        textureStore(ao_dst, coord, vec4f(center_ao, center_ao, center_ao, 1.0));
        textureStore(bent_dst, coord, vec4f(center_bent, 1.0));
        return;
    }

    let center_normal = center_normal_raw / center_normal_len;
    let center_position = textureLoad(position_tex, coord, 0).xyz;

    var weight_sum = 1.0;
    var ao_sum = center_ao;
    var bent_sum = center_bent;

    for (var step = -radius; step <= radius; step = step + 1) {
        let tap_offset = vec2<i32>(
            i32(round(f32(step) * settings.direction.x)),
            i32(round(f32(step) * settings.direction.y))
        );
        let tap_coord = vec2<i32>(
            clamp(coord.x + tap_offset.x, 0, i32(dims.x) - 1),
            clamp(coord.y + tap_offset.y, 0, i32(dims.y) - 1)
        );

        let tap_normal_raw = textureLoad(normal_tex, tap_coord, 0).xyz;
        let tap_normal_len = length(tap_normal_raw);
        if (tap_normal_len < 1e-6) {
            continue;
        }

        let tap_normal = tap_normal_raw / tap_normal_len;
        let tap_position = textureLoad(position_tex, tap_coord, 0).xyz;
        let tap_ao = textureLoad(ao_src, tap_coord, 0).r;
        let tap_bent = safe_normalize(textureLoad(bent_src, tap_coord, 0).xyz);

        let spatial_weight = gaussian(f32(step * step), max(1.0, settings.radius_px * 0.5));
        let position_delta = tap_position - center_position;
        let plane_distance = abs(dot(position_delta, center_normal));
        let position_weight = gaussian(plane_distance * plane_distance, settings.position_sigma);
        let normal_weight = pow(max(dot(center_normal, tap_normal), 0.0), max(settings.normal_power, 0.0));
        let ao_delta = tap_ao - center_ao;
        let ao_weight = gaussian(ao_delta * ao_delta, settings.ao_sigma);

        let weight = spatial_weight * position_weight * normal_weight * ao_weight;
        ao_sum += tap_ao * weight;
        bent_sum += tap_bent * weight;
        weight_sum += weight;
    }

    let filtered_ao = clamp(ao_sum / max(weight_sum, 1e-5), 0.0, 1.0);
    var filtered_bent = safe_normalize(bent_sum / max(weight_sum, 1e-5));
    if (dot(filtered_bent, center_normal) < 0.0) {
        filtered_bent = center_normal;
    }

    textureStore(ao_dst, coord, vec4f(filtered_ao, filtered_ao, filtered_ao, 1.0));
    textureStore(bent_dst, coord, vec4f(filtered_bent, 1.0));
}
