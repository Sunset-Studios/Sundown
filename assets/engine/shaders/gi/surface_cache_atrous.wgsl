#include "common.wgsl"

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  SCGI DISOCCLUSION RECONSTRUCTION                                       ║
// ║                                                                          ║
// ║  Young or uncertain presentation pixels cross a geometry-aware à-trous  ║
// ║  pyramid. Mature history takes the copy path, keeping the steady-state   ║
// ║  cost small and preserving converged surface-cache detail.               ║
// ╚══════════════════════════════════════════════════════════════════════════╝

struct SurfaceCacheAtrousParams {
    step_width: f32,
    phi_plane: f32,
    phi_normal: f32,
    luma_sigma: f32,
    confidence_threshold: f32,
    disocclusion_history_frames: f32,
    _padding0: f32,
    _padding1: f32,
};

@group(1) @binding(0) var<uniform> atrous_params: SurfaceCacheAtrousParams;
@group(1) @binding(1) var input_diffuse: texture_2d<f32>;
@group(1) @binding(2) var resolve_aux: texture_2d<f32>;
@group(1) @binding(3) var depth_texture: texture_2d<f32>;
@group(1) @binding(4) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(5) var output_diffuse: texture_storage_2d<rgba16float, write>;

const SURFACE_CACHE_ATROUS_KERNEL: array<f32, 3> = array<f32, 3>(
    1.0,
    2.0 / 3.0,
    1.0 / 6.0
);

fn surface_cache_atrous_kernel_weight(offset: vec2<i32>) -> f32 {
    return SURFACE_CACHE_ATROUS_KERNEL[u32(abs(offset.x))] *
        SURFACE_CACHE_ATROUS_KERNEL[u32(abs(offset.y))];
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let resolution = textureDimensions(input_diffuse);
    if (gid.x >= resolution.x || gid.y >= resolution.y) {
        return;
    }

    let coord = vec2<i32>(gid.xy);
    let center = textureLoad(input_diffuse, coord, 0);
    let center_aux = textureLoad(resolve_aux, coord, 0);
    if (center.w <= 0.0) {
        textureStore(output_diffuse, coord, center);
        return;
    }

    let history_progress = smoothstep(
        1.0,
        max(atrous_params.disocclusion_history_frames, 2.0),
        center.w
    );
    let history_filter_amount = 1.0 - history_progress;
    var confidence_filter_amount = 0.0;
    if (center_aux.w > 0.5) {
        confidence_filter_amount = 1.0 - smoothstep(
            atrous_params.confidence_threshold,
            1.0,
            center_aux.x
        );
    }
    let filter_amount = max(
        history_filter_amount,
        confidence_filter_amount
    );
    if (filter_amount <= 1e-3) {
        textureStore(output_diffuse, coord, center);
        return;
    }

    let center_depth = textureLoad(depth_texture, coord, 0).r;
    let center_normal_data = textureLoad(gbuffer_normal, coord, 0).xyz;
    if (
        center_depth >= 1.0 ||
        dot(center_normal_data, center_normal_data) <= 1e-8
    ) {
        textureStore(output_diffuse, coord, center);
        return;
    }

    let view_index = u32(frame_info.view_index);
    let center_normal = normalize(center_normal_data);
    let center_position = reconstruct_world_position(
        coord_to_uv(coord, resolution),
        center_depth,
        view_index
    );
    let center_linear_depth = abs(
        (view_buffer[view_index].view_matrix * vec4<f32>(center_position, 1.0)).z
    );
    let inverse_center_depth = 1.0 / max(center_linear_depth, 1e-3);
    let center_luminance = luminance(center.xyz);
    let step_width = max(i32(atrous_params.step_width + 0.5), 1);
    let maximum_coord = vec2<i32>(resolution) - vec2<i32>(1);

    var radiance_sum = vec3<f32>(0.0);
    var weight_sum = 0.0;
    for (var tap_y = -2; tap_y <= 2; tap_y = tap_y + 1) {
        for (var tap_x = -2; tap_x <= 2; tap_x = tap_x + 1) {
            let tap_offset = vec2<i32>(tap_x, tap_y);
            let tap_coord = clamp(
                coord + tap_offset * step_width,
                vec2<i32>(0),
                maximum_coord
            );
            let tap_depth = textureLoad(depth_texture, tap_coord, 0).r;
            let tap_normal_data = textureLoad(
                gbuffer_normal,
                tap_coord,
                0
            ).xyz;
            if (
                tap_depth >= 1.0 ||
                dot(tap_normal_data, tap_normal_data) <= 1e-8
            ) {
                continue;
            }

            let tap_normal = normalize(tap_normal_data);
            let normal_alignment = max(dot(center_normal, tap_normal), 0.0);
            if (normal_alignment < 0.8) {
                continue;
            }
            let tap_position = reconstruct_world_position(
                coord_to_uv(tap_coord, resolution),
                tap_depth,
                view_index
            );
            let position_delta = tap_position - center_position;
            let plane_delta = max(
                abs(dot(position_delta, center_normal)),
                abs(dot(position_delta, tap_normal))
            ) * inverse_center_depth;

            let tap_radiance = textureLoad(input_diffuse, tap_coord, 0);
            if (tap_radiance.w <= 0.0) {
                continue;
            }
            let tap_aux = textureLoad(resolve_aux, tap_coord, 0);
            let tap_luminance = luminance(tap_radiance.xyz);
            let uncertainty_sigma = max(
                (center_aux.y + tap_aux.y) * atrous_params.luma_sigma,
                max(center_luminance * 0.15, 0.03)
            );

            let kernel_weight = surface_cache_atrous_kernel_weight(tap_offset);
            let plane_weight = exp(
                -plane_delta / max(atrous_params.phi_plane, 1e-4)
            );
            let normal_weight = pow(
                normal_alignment,
                max(atrous_params.phi_normal, 1.0)
            );
            let luminance_weight = exp(
                -abs(tap_luminance - center_luminance) / uncertainty_sigma
            );
            let tap_reliability = select(
                0.5,
                mix(0.25, 1.0, clamp(tap_aux.x, 0.0, 1.0)),
                tap_aux.w > 0.5
            );
            let weight = kernel_weight * plane_weight * normal_weight *
                luminance_weight * tap_reliability;
            radiance_sum += tap_radiance.xyz * weight;
            weight_sum += weight;
        }
    }

    if (weight_sum <= 1e-5) {
        textureStore(output_diffuse, coord, center);
        return;
    }
    let filtered = radiance_sum / weight_sum;
    textureStore(
        output_diffuse,
        coord,
        vec4<f32>(mix(center.xyz, filtered, filter_amount), center.w)
    );
}
