#include "common.wgsl"

struct RtaoAtrousParams {
    step_width: f32,
    phi_depth: f32,
    phi_normal: f32,
    ao_sigma: f32,
};

@group(1) @binding(0) var<uniform> atrous_params: RtaoAtrousParams;
@group(1) @binding(1) var input_ao: texture_2d<f32>;
@group(1) @binding(2) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(3) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(4) var output_ao: texture_storage_2d<r32float, write>;

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
    let resolution = textureDimensions(input_ao);
    if (gid.x >= resolution.x || gid.y >= resolution.y) {
        return;
    }

    let pixel_coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let center_position_data = textureLoad(gbuffer_position, pixel_coord, 0);
    let center_normal_data = textureLoad(gbuffer_normal, pixel_coord, 0);
    let center_ao = textureLoad(input_ao, pixel_coord, 0).r;

    if (center_position_data.w <= 0.0 || length(center_normal_data.xyz) <= 0.0) {
        textureStore(output_ao, pixel_coord, vec4<f32>(center_ao, 0.0, 0.0, 1.0));
        return;
    }

    let center_position = center_position_data.xyz;
    let center_normal = safe_normalize(center_normal_data.xyz);
    let center_view = view_buffer[u32(frame_info.view_index)].view_position.xyz;
    let center_depth = length(center_position - center_view);

    let step_width_i32 = max(1, i32(atrous_params.step_width));
    let max_coord = vec2<i32>(i32(resolution.x) - 1, i32(resolution.y) - 1);

    var ao_sum = 0.0;
    var weight_sum = 0.0;

    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let offset = vec2<i32>(x, y);
            let sample_coord = clamp(
                pixel_coord + offset * step_width_i32,
                vec2<i32>(0, 0),
                max_coord
            );

            let sample_position_data = textureLoad(gbuffer_position, sample_coord, 0);
            let sample_normal_data = textureLoad(gbuffer_normal, sample_coord, 0);
            if (sample_position_data.w <= 0.0 || length(sample_normal_data.xyz) <= 0.0) {
                continue;
            }

            let sample_position = sample_position_data.xyz;
            let sample_normal = safe_normalize(sample_normal_data.xyz);
            let sample_depth = length(sample_position - center_view);
            let sample_ao = textureLoad(input_ao, sample_coord, 0).r;

            let kernel_weight = get_kernel_weight(offset);
            let depth_rel = abs(sample_depth - center_depth) / max(center_depth, 1e-4);
            let depth_weight = exp(-depth_rel / max(atrous_params.phi_depth, 1e-4));
            let normal_weight = pow(max(dot(center_normal, sample_normal), 0.0), atrous_params.phi_normal);
            let ao_weight = exp(-abs(sample_ao - center_ao) / max(atrous_params.ao_sigma, 1e-4));

            let final_weight = kernel_weight * depth_weight * normal_weight * ao_weight;
            ao_sum += sample_ao * final_weight;
            weight_sum += final_weight;
        }
    }

    let filtered_ao = select(center_ao, ao_sum / weight_sum, weight_sum > 1e-6);
    textureStore(output_ao, pixel_coord, vec4<f32>(filtered_ao, 0.0, 0.0, 1.0));
}
