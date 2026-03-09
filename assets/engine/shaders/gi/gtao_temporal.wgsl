#include "common.wgsl"

struct GTAOTemporalSettings {
    temporal_response: f32,
    radius: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(1) @binding(0) var current_ao_tex: texture_2d<f32>;
@group(1) @binding(1) var current_bent_tex: texture_2d<f32>;
@group(1) @binding(2) var history_ao_tex: texture_2d<f32>;
@group(1) @binding(3) var history_bent_tex: texture_2d<f32>;
@group(1) @binding(4) var position_tex: texture_2d<f32>;
@group(1) @binding(5) var prev_position_tex: texture_2d<f32>;
@group(1) @binding(6) var normal_tex: texture_2d<f32>;
@group(1) @binding(7) var prev_normal_tex: texture_2d<f32>;
@group(1) @binding(8) var motion_tex: texture_2d<f32>;
@group(1) @binding(9) var ao_output: texture_storage_2d<r32float, write>;
@group(1) @binding(10) var bent_output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(11) var<uniform> settings: GTAOTemporalSettings;

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(current_ao_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let coord = vec2<i32>(gid.xy);
    let resolution = vec2f(dims);
    let uv = (vec2f(gid.xy) + 0.5) / resolution;

    let current_ao = textureLoad(current_ao_tex, coord, 0).r;
    let current_bent = safe_normalize(textureLoad(current_bent_tex, coord, 0).xyz);
    let current_normal_raw = textureLoad(normal_tex, coord, 0).xyz;
    let current_normal_len = length(current_normal_raw);
    if (current_normal_len < 1e-6) {
        textureStore(ao_output, coord, vec4f(current_ao, current_ao, current_ao, 1.0));
        textureStore(bent_output, coord, vec4f(current_bent, 1.0));
        return;
    }

    let current_normal = current_normal_raw / current_normal_len;
    let current_position = textureLoad(position_tex, coord, 0).xyz;
    let motion = textureLoad(motion_tex, coord, 0).xy;
    let prev_uv = uv + vec2f(-0.5 * motion.x, 0.5 * motion.y);
    let prev_in_bounds = all(prev_uv >= vec2f(0.0)) && all(prev_uv <= vec2f(1.0));

    var neighborhood_min = current_ao;
    var neighborhood_max = current_ao;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let tap = vec2<i32>(
                clamp(coord.x + dx, 0, i32(dims.x) - 1),
                clamp(coord.y + dy, 0, i32(dims.y) - 1)
            );
            let tap_ao = textureLoad(current_ao_tex, tap, 0).r;
            neighborhood_min = min(neighborhood_min, tap_ao);
            neighborhood_max = max(neighborhood_max, tap_ao);
        }
    }

    var history_valid = false;
    var history_ao = current_ao;
    var history_bent = current_bent;

    if (prev_in_bounds) {
        let prev_coord = uv_to_coord(prev_uv, dims);
        let prev_normal_raw = textureLoad(prev_normal_tex, prev_coord, 0).xyz;
        let prev_normal_len = length(prev_normal_raw);
        if (prev_normal_len > 1e-6) {
            let prev_normal = prev_normal_raw / prev_normal_len;
            let prev_position = textureLoad(prev_position_tex, prev_coord, 0).xyz;
            let camera_position = view_buffer[u32(frame_info.view_index)].view_position.xyz;
            let view_distance = distance(camera_position, current_position);
            let position_threshold = max(settings.radius * 0.35, 0.0025 * view_distance);
            let normal_match = dot(current_normal, prev_normal);
            let position_error = distance(current_position, prev_position);
            history_valid = normal_match > 0.85 && position_error <= position_threshold;

            if (history_valid) {
                history_ao = textureLoad(history_ao_tex, prev_coord, 0).r;
                history_bent = safe_normalize(textureLoad(history_bent_tex, prev_coord, 0).xyz);
                history_ao = clamp(history_ao, neighborhood_min, neighborhood_max);
                if (dot(history_bent, current_normal) < 0.0) {
                    history_bent = current_bent;
                }
            }
        }
    }

    let motion_pixels = length(vec2f(-0.5 * motion.x, 0.5 * motion.y) * resolution);
    let discrepancy = abs(history_ao - current_ao);
    let current_weight = clamp(
        max(settings.temporal_response, discrepancy * 0.85 + motion_pixels * 0.03),
        0.0,
        1.0
    );
    let blend = select(1.0, current_weight, history_valid);

    let ao_value = mix(history_ao, current_ao, blend);
    var bent_value = safe_normalize(mix(history_bent, current_bent, blend));
    if (dot(bent_value, current_normal) < 0.0) {
        bent_value = current_normal;
    }

    textureStore(ao_output, coord, vec4f(ao_value, ao_value, ao_value, 1.0));
    textureStore(bent_output, coord, vec4f(bent_value, 1.0));
}
