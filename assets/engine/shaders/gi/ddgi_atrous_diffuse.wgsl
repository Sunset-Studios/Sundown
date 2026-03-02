#include "common.wgsl"

struct DdgiAtrousParams {
    step_width: f32,
    phi_depth: f32,
    phi_normal: f32,
    luma_sigma: f32,
};

@group(1) @binding(0) var<uniform> atrous_params: DdgiAtrousParams;
@group(1) @binding(1) var input_diffuse: texture_2d<f32>;
@group(1) @binding(2) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(3) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(4) var output_diffuse: texture_storage_2d<rgba16float, write>;

const KERNEL_WEIGHTS: array<f32, 3> = array<f32, 3>(
    1.0,
    2.0 / 3.0,
    1.0 / 6.0
);

fn get_kernel_weight(offset: vec2<i32>) -> f32 {
    let x = abs(offset.x);
    let y = abs(offset.y);
    return KERNEL_WEIGHTS[x] * KERNEL_WEIGHTS[y];
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let resolution = textureDimensions(input_diffuse);
    if (gid.x >= resolution.x || gid.y >= resolution.y) {
        return;
    }

    let pixel_coord = vec2<i32>(i32(gid.x), i32(gid.y));

    let center_position_data = textureLoad(gbuffer_position, pixel_coord, 0);
    let center_normal_data = textureLoad(gbuffer_normal, pixel_coord, 0);
    let center_diffuse_data = textureLoad(input_diffuse, pixel_coord, 0);

    if (center_position_data.w <= 0.0 || length(center_normal_data.xyz) <= 0.0) {
        textureStore(output_diffuse, pixel_coord, center_diffuse_data);
        return;
    }

    let center_position = center_position_data.xyz;
    let center_normal = safe_normalize(center_normal_data.xyz);
    let center_view = view_buffer[u32(frame_info.view_index)].view_position.xyz;
    let center_depth = length(center_position - center_view);
    let center_luma = luminance(center_diffuse_data.xyz);

    let step_width_i32 = max(1, i32(atrous_params.step_width));
    let max_coord = vec2<i32>(i32(resolution.x) - 1, i32(resolution.y) - 1);

    var weighted_sum = vec3<f32>(0.0);
    var weight_sum = 0.0;
    var alpha_sum = 0.0;

    for (var y = -2; y <= 2; y = y + 1) {
        for (var x = -2; x <= 2; x = x + 1) {
            let offset = vec2<i32>(x, y);
            let sample_coord = clamp(
                pixel_coord + offset * step_width_i32,
                vec2<i32>(0, 0),
                max_coord
            );

            let sample_position_data = textureLoad(gbuffer_position, sample_coord, 0);
            let sample_normal_data = textureLoad(gbuffer_normal, sample_coord, 0);
            let sample_diffuse_data = textureLoad(input_diffuse, sample_coord, 0);

            if (sample_position_data.w <= 0.0 || length(sample_normal_data.xyz) <= 0.0) {
                continue;
            }

            let sample_position = sample_position_data.xyz;
            let sample_normal = safe_normalize(sample_normal_data.xyz);
            let sample_depth = length(sample_position - center_view);
            let sample_luma = luminance(sample_diffuse_data.xyz);

            let kernel_weight = get_kernel_weight(offset);
            let depth_rel = abs(sample_depth - center_depth) / max(center_depth, 1e-4);
            let depth_weight = exp(-depth_rel / max(atrous_params.phi_depth, 1e-4));
            let normal_weight = pow(max(dot(center_normal, sample_normal), 0.0), atrous_params.phi_normal);
            let luma_weight = exp(-abs(sample_luma - center_luma) / max(atrous_params.luma_sigma, 1e-4));

            let final_weight = kernel_weight * depth_weight * normal_weight * luma_weight;
            weighted_sum += sample_diffuse_data.xyz * final_weight;
            alpha_sum += sample_diffuse_data.w * final_weight;
            weight_sum += final_weight;
        }
    }

    let normalized_diffuse = select(center_diffuse_data.xyz, weighted_sum / weight_sum, weight_sum > 1e-6);
    let normalized_alpha = select(center_diffuse_data.w, alpha_sum / weight_sum, weight_sum > 1e-6);
    textureStore(output_diffuse, pixel_coord, vec4<f32>(normalized_diffuse, normalized_alpha));
}
