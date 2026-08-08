#include "common.wgsl"

struct SurfaceCacheSpatialParams {
    depth_sigma: f32,
    normal_threshold: f32,
    luminance_sigma: f32,
    strength: f32,
    direction_x: f32,
    direction_y: f32,
    padding0: f32,
    padding1: f32,
};

@group(1) @binding(0) var<uniform> spatial_params: SurfaceCacheSpatialParams;
@group(1) @binding(1) var input_diffuse: texture_2d<f32>;
@group(1) @binding(2) var depth_texture: texture_2d<f32>;
@group(1) @binding(3) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(4) var output_diffuse: texture_storage_2d<rgba16float, write>;

const SURFACE_CACHE_SPATIAL_WEIGHTS = array<f32, 5>(
    0.0625,
    0.25,
    0.375,
    0.25,
    0.0625
);

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let resolution = textureDimensions(input_diffuse);
    if (gid.x >= resolution.x || gid.y >= resolution.y) {
        return;
    }

    let coord = vec2<i32>(gid.xy);
    let center = textureLoad(input_diffuse, coord, 0);
    let center_normal_data = textureLoad(gbuffer_normal, coord, 0);
    if (dot(center_normal_data.xyz, center_normal_data.xyz) <= 1e-8) {
        textureStore(output_diffuse, coord, center);
        return;
    }

    let view_index = u32(frame_info.view_index);
    let view = view_buffer[view_index];
    let center_position = reconstruct_world_position(
        coord_to_uv(coord, resolution),
        textureLoad(depth_texture, coord, 0).r,
        view_index
    );
    let center_depth = abs((view.view_matrix * vec4<f32>(center_position, 1.0)).z);
    let center_normal = safe_normalize(center_normal_data.xyz);
    let center_luminance = luminance(center.xyz);
    let direction = vec2<i32>(
        i32(spatial_params.direction_x),
        i32(spatial_params.direction_y)
    );

    var diffuse_sum = vec3<f32>(0.0);
    var history_sum = 0.0;
    var weight_sum = 0.0;
    for (var tap_index = 0; tap_index < 5; tap_index = tap_index + 1) {
        let tap_offset = tap_index - 2;
        let tap_coord = clamp(
            coord + direction * tap_offset,
            vec2<i32>(0),
            vec2<i32>(resolution) - vec2<i32>(1)
        );
        let tap_normal_data = textureLoad(gbuffer_normal, tap_coord, 0);
        if (dot(tap_normal_data.xyz, tap_normal_data.xyz) <= 1e-8) {
            continue;
        }

        let tap = textureLoad(input_diffuse, tap_coord, 0);
        if (tap.w <= 0.0) {
            continue;
        }
        let tap_position = reconstruct_world_position(
            coord_to_uv(tap_coord, resolution),
            textureLoad(depth_texture, tap_coord, 0).r,
            view_index
        );
        let tap_depth = abs((view.view_matrix * vec4<f32>(tap_position, 1.0)).z);
        let normal_weight = smoothstep(
            spatial_params.normal_threshold,
            1.0,
            dot(center_normal, safe_normalize(tap_normal_data.xyz))
        );
        let relative_depth_delta = abs(tap_depth - center_depth) /
            max(center_depth, 1e-3);
        let depth_weight = exp(
            -relative_depth_delta / max(spatial_params.depth_sigma, 1e-4)
        );
        let tap_luminance = luminance(tap.xyz);
        let luminance_scale = max(max(center_luminance, tap_luminance), 0.25);
        let luminance_weight = 1.0 / (
            1.0 + abs(tap_luminance - center_luminance) /
                max(spatial_params.luminance_sigma * luminance_scale, 1e-4)
        );
        let weight = SURFACE_CACHE_SPATIAL_WEIGHTS[tap_index]
            * depth_weight
            * normal_weight
            * luminance_weight;
        diffuse_sum += tap.xyz * weight;
        history_sum += tap.w * weight;
        weight_sum += weight;
    }

    let filtered = select(
        center.xyz,
        diffuse_sum / max(weight_sum, 1e-5),
        weight_sum > 1e-5
    );
    let filtered_history = select(
        center.w,
        history_sum / max(weight_sum, 1e-5),
        weight_sum > 1e-5
    );
    textureStore(
        output_diffuse,
        coord,
        vec4<f32>(
            mix(center.xyz, filtered, clamp(spatial_params.strength, 0.0, 1.0)),
            filtered_history
        )
    );
}
