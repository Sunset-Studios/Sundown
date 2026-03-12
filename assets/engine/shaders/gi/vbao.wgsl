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

@group(1) @binding(0) var normal_tex: texture_2d<f32>;
@group(1) @binding(1) var depth_tex: texture_2d<f32>;
@group(1) @binding(2) var ao_output: texture_storage_2d<r32float, write>;
@group(1) @binding(3) var<uniform> settings: VBAOSettings;

const HALF_PI = 1.5707963267948966;
const INV_PI = 0.3183098861837907;
const OCCLUSION_BIN_COUNT = 32.0;
const OCCLUSION_BIN_COUNT_U = 32u;
const OCCLUSION_BIN_RCP = 1.0 / OCCLUSION_BIN_COUNT;

fn is_orthographic_projection(view_index: u32) -> bool {
    return view_buffer[view_index].projection_matrix[3][3] > 0.5;
}

fn project_view_to_uv(position_vs: vec3<f32>, view_index: u32) -> vec2<f32> {
    let clip = view_buffer[view_index].projection_matrix * vec4<f32>(position_vs, 1.0);
    let ndc = clip.xy / max(clip.w, epsilon);
    return vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
}

fn acos_poly(x: f32) -> f32 {
    return HALF_PI + (-0.20491203466059038 + 0.04832927023878897 * x) * x;
}

fn fast_acos(x: f32) -> f32 {
    let ax = clamp(abs(x), 0.0, 1.0);
    let value = acos_poly(ax) * sqrt(max(0.0, 1.0 - ax));
    return select(PI - value, value, x >= 0.0);
}

fn slice_rel_cdf_cos(x: f32, normal_angle: f32, normal_cos: f32, positive_side: bool) -> f32 {
    if (x <= 0.0 || x >= 1.0) {
        return clamp(x, 0.0, 1.0);
    }

    let phi = x * PI - HALF_PI;
    let n0 = select(1.0, 3.0, positive_side);
    let n1 = select(1.0, -1.0, positive_side);
    let n2 = select(0.0, 4.0, positive_side);

    let numerator =
        n0 * normal_cos +
        n1 * cos(normal_angle - 2.0 * phi) +
        (n2 * normal_angle + (n1 * 2.0) * phi + PI) * sin(normal_angle);
    let denominator = 4.0 * (normal_cos + normal_angle * sin(normal_angle));

    return numerator / max(denominator, 1e-5);
}

fn sample_slice_dir(normal_vvs: vec3<f32>, rnd01: f32) -> vec2<f32> {
    let angle = rnd01 * PI;
    var dir0 = vec2<f32>(cos(angle), sin(angle));
    let xy_length = length(normal_vvs.xy);
    if (xy_length <= 1e-6) {
        return dir0;
    }

    if (dot(dir0, normal_vvs.xy) < 0.0) {
        dir0 = -dir0;
    }

    let n = normal_vvs.xy / xy_length;
    var x = dir0.x * n.y - dir0.y * n.x;

    var stretch = xy_length;
    stretch += (stretch - stretch * stretch) * 0.15;

    let k = 0.21545;
    let a = 0.5 + 0.5 / k;
    let b = 0.5 - 0.5 / k;
    let d = b * b;
    let c = 4.0 / k;
    let stretch_x = a - sqrt(max(d + c * pow(0.5 - 0.5 * stretch, 2.0), 0.0));
    x *= stretch_x;

    let v = select(0.0, 2.0, x > 0.0);
    let g = -k - 1.0;
    let y_curve = abs(v - sqrt(clamp((abs(x) * k + g) * abs(x) + 1.0, 0.0, 1.0)));
    let stretch_y = 1.0 / max(stretch, 1e-5);

    var dir = vec2<f32>(0.0);
    dir.y = stretch_y - stretch_y * y_curve;
    dir.x = sqrt(clamp(1.0 - dir.y * dir.y, 0.0, 1.0));

    return vec2<f32>(
        dir.x * n.x - dir.y * n.y,
        dir.y * n.x + dir.x * n.y
    );
}

fn make_rng(pixel_coord: vec2<u32>, frame_index: u32) -> u32 {
    return hash(
        (pixel_coord.x * 0x9E3779B9u) ^
        (pixel_coord.y * 0x85EBCA6Bu) ^
        (frame_index * 0xC2B2AE35u)
    );
}

fn rnd4(pixel_coord: vec2<u32>, sample_index: u32, frame_index: u32) -> vec4<f32> {
    let rng = make_rng(pixel_coord, frame_index);
    return vec4<f32>(
        rand_sobol(rng, sample_index, 0u),
        rand_sobol(rng, sample_index, 1u),
        rand_sobol(rng, sample_index, 2u),
        rand_sobol(rng, sample_index, 3u)
    );
}

fn horizon_interval_mask(interval01: vec2<f32>) -> u32 {
    let quantized = vec2<u32>(floor(clamp(interval01, vec2<f32>(0.0), vec2<f32>(1.0)) * OCCLUSION_BIN_COUNT));

    var left_mask = 0u;
    if (quantized.x < OCCLUSION_BIN_COUNT_U) {
        left_mask = 0xFFFFFFFFu << quantized.x;
    }

    var right_mask = 0u;
    if (quantized.y > 0u) {
        right_mask = 0xFFFFFFFFu >> (OCCLUSION_BIN_COUNT_U - min(quantized.y, OCCLUSION_BIN_COUNT_U));
    }

    return left_mask & right_mask;
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let trace_resolution = textureDimensions(ao_output);
    if (gid.x >= trace_resolution.x || gid.y >= trace_resolution.y) {
        return;
    }

    let full_resolution = textureDimensions(normal_tex);
    let coord = vec2<i32>(gid.xy);
    let trace_resolution_f = vec2<f32>(f32(trace_resolution.x), f32(trace_resolution.y));
    let full_resolution_f = vec2<f32>(f32(full_resolution.x), f32(full_resolution.y));
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) / trace_resolution_f;
    let full_coord = uv_to_coord(uv, full_resolution);

    let normal_raw = textureLoad(normal_tex, full_coord, 0).xyz;
    let normal_len = length(normal_raw);
    if (normal_len < 1e-6) {
        textureStore(ao_output, coord, vec4<f32>(1.0, 1.0, 1.0, 1.0));
        return;
    }

    let view_index = u32(frame_info.view_index);
    let view = view_buffer[view_index];
    let orthographic = is_orthographic_projection(view_index);
    let frame_index = u32(frame_info.frame_index);

    let current_uv = (vec2<f32>(f32(full_coord.x), f32(full_coord.y)) + 0.5) / full_resolution_f;
    let current_depth = textureLoad(depth_tex, full_coord, 0).r;
    if (current_depth >= 1.0) {
        textureStore(ao_output, coord, vec4<f32>(1.0, 1.0, 1.0, 1.0));
        return;
    }

    let normal_ws = normal_raw / normal_len;
    let position_ws = reconstruct_world_position(current_uv, current_depth, view_index) + normal_ws * settings.bias;

    let position_vs = (view.view_matrix * vec4<f32>(position_ws, 1.0)).xyz;
    let normal_vs = safe_normalize((view.view_matrix * vec4<f32>(normal_ws, 0.0)).xyz);
    let view_vec_vs = select(safe_normalize(-position_vs), vec3<f32>(0.0, 0.0, 1.0), orthographic);

    let basis_to_vs = orthonormalize(-view_vec_vs);
    let normal_vvs = transpose(basis_to_vs) * normal_vs;
    let ray_start_px = vec2<f32>(f32(full_coord.x), f32(full_coord.y)) + 0.5;

    let slice_count = clamp(u32(settings.slice_count + 0.5), 1u, 4u);
    let sample_count = clamp(u32(settings.sample_count + 0.5), 4u, 32u);
    let projection_scale = 0.5 * max(
        abs(view.projection_matrix[0][0]) * full_resolution_f.x,
        abs(view.projection_matrix[1][1]) * full_resolution_f.y
    );
    let view_depth = select(max(-position_vs.z, 1e-3), 1.0, orthographic);
    let radius_px = clamp(settings.radius * projection_scale / view_depth, 1.0, settings.max_radius_px);
    let step_scale = pow(max(radius_px, 1.0), 1.0 / f32(sample_count));

    var ao_accum = 0.0;
    var valid_slice_count = 0u;

    for (var slice_index = 0u; slice_index < slice_count; slice_index = slice_index + 1u) {
        let sample_index = frame_index * slice_count + slice_index;
        let rnd = rnd4(vec2<u32>(u32(full_coord.x), u32(full_coord.y)), sample_index, frame_index);

        let local_dir = sample_slice_dir(normal_vvs, rnd.x);
        let sample_dir_vs = safe_normalize(basis_to_vs * vec3<f32>(local_dir, 0.0));
        if (length(sample_dir_vs) < 1e-6) {
            continue;
        }

        let probe_uv = project_view_to_uv(
            position_vs + sample_dir_vs * max(view.near * 0.5, 0.05),
            view_index
        );
        let screen_dir = (probe_uv - current_uv) * full_resolution_f;
        let screen_dir_len = length(screen_dir);
        if (screen_dir_len < 1e-5) {
            continue;
        }
        let screen_step_dir = screen_dir / screen_dir_len;

        let slice_normal = safe_normalize(cross(view_vec_vs, sample_dir_vs));
        let projected_normal = normal_vs - slice_normal * dot(normal_vs, slice_normal);
        let projected_normal_len = length(projected_normal);
        if (projected_normal_len < 1e-5) {
            continue;
        }

        let projected_normal_rcp_len = 1.0 / projected_normal_len;
        let normal_cos = clamp(dot(projected_normal, view_vec_vs) * projected_normal_rcp_len, -1.0, 1.0);
        let tangent = cross(slice_normal, projected_normal);
        let angle_sign = select(1.0, -1.0, dot(view_vec_vs, tangent) < 0.0);
        let normal_angle = angle_sign * fast_acos(normal_cos);
        let angle_offset = normal_angle * INV_PI + 0.5;
        let point_jitter = rnd.w * OCCLUSION_BIN_RCP;

        var occ_bits = 0u;
        var side_jitter = rnd.z;

        for (var side_index = 0u; side_index < 2u; side_index = side_index + 1u) {
            let side_sign = select(-1.0, 1.0, side_index == 1u);
            var t = pow(step_scale, side_jitter);
            side_jitter = 1.0 - side_jitter;

            for (var step_index = 0u; step_index < sample_count; step_index = step_index + 1u) {
                let sample_px = ray_start_px + screen_step_dir * (side_sign * t);
                t *= step_scale;

                if (any(sample_px < vec2<f32>(0.0)) || any(sample_px >= full_resolution_f)) {
                    break;
                }

                let sample_uv = sample_px / full_resolution_f;
                let sample_coord = uv_to_coord(sample_uv, full_resolution);
                let sample_depth = textureLoad(depth_tex, sample_coord, 0).r;
                if (sample_depth >= 1.0) {
                    continue;
                }

                let sample_uv_center =
                    (vec2<f32>(f32(sample_coord.x), f32(sample_coord.y)) + 0.5) / full_resolution_f;
                let sample_position_ws = reconstruct_world_position(sample_uv_center, sample_depth, view_index);
                let sample_position_vs = (view.view_matrix * vec4<f32>(sample_position_ws, 1.0)).xyz;
                let delta_front = sample_position_vs - position_vs;
                let delta_front_len_sq = dot(delta_front, delta_front);
                if (delta_front_len_sq <= 1e-8) {
                    continue;
                }

                let sample_view_ray = select(safe_normalize(sample_position_vs), -view_vec_vs, orthographic);
                let sample_back_position_vs = sample_position_vs + sample_view_ray * settings.thickness;
                let delta_back = sample_back_position_vs - position_vs;
                let delta_back_len_sq = dot(delta_back, delta_back);
                if (delta_back_len_sq <= 1e-8) {
                    continue;
                }

                let horizon_cos = vec2<f32>(
                    dot(delta_front * inverseSqrt(delta_front_len_sq), view_vec_vs),
                    dot(delta_back * inverseSqrt(delta_back_len_sq), view_vec_vs)
                );
                let horizon_angles = vec2<f32>(
                    fast_acos(clamp(horizon_cos.x, -1.0, 1.0)),
                    fast_acos(clamp(horizon_cos.y, -1.0, 1.0))
                ) * side_sign;

                var horizon01 = clamp(horizon_angles * INV_PI + angle_offset, vec2<f32>(0.0), vec2<f32>(1.0));
                if (side_sign < 0.0) {
                    horizon01 = horizon01.yx;
                }

                horizon01 = vec2<f32>(
                    slice_rel_cdf_cos(horizon01.x, normal_angle, normal_cos, side_sign > 0.0),
                    slice_rel_cdf_cos(horizon01.y, normal_angle, normal_cos, side_sign > 0.0)
                );
                horizon01 = clamp(horizon01 + vec2<f32>(point_jitter), vec2<f32>(0.0), vec2<f32>(1.0));

                occ_bits |= horizon_interval_mask(horizon01);
            }
        }

        let occlusion = f32(countOneBits(occ_bits)) * OCCLUSION_BIN_RCP;
        ao_accum += 1.0 - occlusion;
        valid_slice_count += 1u;
    }

    let ao_visibility = clamp(
        select(1.0, ao_accum / f32(valid_slice_count), valid_slice_count > 0u),
        0.0,
        1.0
    );

    textureStore(ao_output, coord, vec4<f32>(ao_visibility, 0.0, 0.0, 1.0));
}


