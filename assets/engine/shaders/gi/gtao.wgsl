#include "common.wgsl"

struct GTAOSettings {
    radius: f32,
    bias: f32,
    sample_count: f32,
    max_radius_px: f32,
    thickness: f32,
    temporal_response: f32,
    denoise_radius: f32,
    denoise_position_sigma: f32,
    denoise_normal_power: f32,
    denoise_ao_sigma: f32,
    denoise_direction: vec2f,
    denoise_radius_px: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(1) @binding(0) var normal_tex: texture_2d<f32>;
@group(1) @binding(1) var hzb_tex: texture_2d<f32>;
@group(1) @binding(2) var ao_output: texture_storage_2d<r32float, write>;
@group(1) @binding(3) var bent_output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(4) var<uniform> settings: GTAOSettings;

const HALF_PI = 1.5707963267948966;
const TWO_PI = 6.283185307179586;
const NUM_SLICES: u32 = 3u;
const MIN_STEPS_PER_SIDE: u32 = 3u;
const MAX_STEPS_PER_SIDE: u32 = 8u;

fn hash2(ixy: vec2<u32>, seed: u32) -> vec2<f32> {
    let h0 = hash(ixy.x ^ (seed * 374761393u) ^ (ixy.y * 668265263u));
    let h1 = hash(ixy.y ^ (seed * 2246822519u) ^ (ixy.x * 3266489917u));
    return vec2f(
        f32(h0 & 0x00ffffffu) * (1.0 / 16777215.0),
        f32(h1 & 0x00ffffffu) * (1.0 / 16777215.0)
    );
}

fn fast_acos(x: f32) -> f32 {
    let ax = abs(x);
    var res = -0.156583 * ax + HALF_PI;
    res *= sqrt(max(0.0, 1.0 - ax));
    return select(PI - res, res, x >= 0.0);
}

fn project_view_to_uv(position_vs: vec3f, view_index: u32) -> vec2f {
    let clip = view_buffer[view_index].projection_matrix * vec4f(position_vs, 1.0);
    let ndc = clip.xy / max(clip.w, epsilon);
    return vec2f(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
}

fn reconstruct_view_position(uv: vec2f, depth: f32, view_index: u32) -> vec3f {
    let view = view_buffer[view_index];
    let z_eye = linearize_depth(depth * 2.0 - 1.0, view.near, view.far, view_index);
    let view_z = -z_eye;
    let ndc = vec2f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
    let inv_proj_x = 1.0 / max(abs(view.projection_matrix[0][0]), epsilon);
    let inv_proj_y = 1.0 / max(abs(view.projection_matrix[1][1]), epsilon);
    let view_xy = ndc * z_eye * vec2f(inv_proj_x, inv_proj_y);
    return vec3f(view_xy, view_z);
}

fn sample_hzb_depth(uv: vec2f, step_px: f32) -> f32 {
    let mip_count = textureNumLevels(hzb_tex);
    let mip = clamp(floor(log2(max(step_px, 1.0))) - 1.0, 0.0, f32(max(1u, mip_count) - 1u));
    return textureSampleLevel(hzb_tex, non_filtering_sampler, uv, mip).r;
}

fn integrate_slice(
    n_angle: f32,
    cos_norm: f32,
    h0: f32,
    h1: f32,
    proj_len: f32,
    slice_dir_perp_vs: vec3f,
    view_vec: vec3f,
) -> vec4f {
    let iarc0 = (cos_norm + 2.0 * h0 * sin(n_angle) - cos(2.0 * h0 - n_angle)) * 0.25;
    let iarc1 = (cos_norm + 2.0 * h1 * sin(n_angle) - cos(2.0 * h1 - n_angle)) * 0.25;
    let visibility = proj_len * (iarc0 + iarc1);

    let t0 = (
        6.0 * sin(h0 - n_angle) - sin(3.0 * h0 - n_angle) +
        6.0 * sin(h1 - n_angle) - sin(3.0 * h1 - n_angle) +
        16.0 * sin(n_angle) -
        3.0 * (sin(h0 + n_angle) + sin(h1 + n_angle))
    ) / 12.0;
    let t1 = (
        -cos(3.0 * h0 - n_angle) - cos(3.0 * h1 - n_angle) +
        8.0 * cos(n_angle) -
        3.0 * (cos(h0 + n_angle) + cos(h1 + n_angle))
    ) / 12.0;

    let bent_vs = (slice_dir_perp_vs * t0 + view_vec * t1) * proj_len;
    return vec4f(bent_vs, visibility);
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let trace_resolution = textureDimensions(ao_output);
    if (gid.x >= trace_resolution.x || gid.y >= trace_resolution.y) {
        return;
    }

    let full_resolution = textureDimensions(normal_tex);
    let coord = vec2<i32>(gid.xy);
    let resolution = vec2f(f32(trace_resolution.x), f32(trace_resolution.y));
    let full_resolution_f = vec2f(f32(full_resolution.x), f32(full_resolution.y));
    let uv = (vec2f(f32(gid.x), f32(gid.y)) + 0.5) / resolution;
    let full_coord = uv_to_coord(uv, full_resolution);

    let normal_raw = textureLoad(normal_tex, full_coord, 0).xyz;
    let normal_len = length(normal_raw);
    let depth = textureSampleLevel(hzb_tex, non_filtering_sampler, uv, 0.0).r;
    if (normal_len < 1e-6 || depth >= 1.0) {
        textureStore(ao_output, coord, vec4f(1.0, 1.0, 1.0, 1.0));
        textureStore(bent_output, coord, vec4f(world_up, 1.0));
        return;
    }

    let view_index = u32(frame_info.view_index);
    let view = view_buffer[view_index];
    let normal_ws = normal_raw / normal_len;
    let normal_vs = safe_normalize((view.view_matrix * vec4f(normal_ws, 0.0)).xyz);
    let position_vs = reconstruct_view_position(uv, depth, view_index);
    let view_vec_vs = safe_normalize(-position_vs);

    let basis_vs = orthonormalize(normal_vs);
    let tangent_vs = basis_vs[0];
    let bitangent_vs = basis_vs[1];

    let projection_scale = 0.5 * max(
        abs(view.projection_matrix[0][0]) * resolution.x,
        abs(view.projection_matrix[1][1]) * resolution.y
    );
    let radius_px = clamp(
        settings.radius * projection_scale / max(-position_vs.z, 1e-3),
        1.0,
        settings.max_radius_px
    );
    let step_count = clamp(
        u32(settings.sample_count + 0.5),
        MIN_STEPS_PER_SIDE,
        MAX_STEPS_PER_SIDE
    );
    let inv_resolution = 1.0 / resolution;
    let hzb_step_scale = max(
        full_resolution_f.x / max(resolution.x, 1.0),
        full_resolution_f.y / max(resolution.y, 1.0)
    );

    let random = hash2(gid.xy, u32(frame_info.frame_index));
    let base_rotation = random.x * TWO_PI;
    let radial_jitter = random.y;

    var visibility_accum = 0.0;
    var visibility_weight_accum = 0.0;
    var bent_accum_vs = vec3f(0.0);
    var valid_slice_count = 0u;

    for (var slice = 0u; slice < NUM_SLICES; slice = slice + 1u) {
        let slice_angle = base_rotation + (TWO_PI * f32(slice)) / f32(NUM_SLICES);
        let tangent_dir_vs = safe_normalize(tangent_vs * cos(slice_angle) + bitangent_vs * sin(slice_angle));
        let ortho_dir = tangent_dir_vs - view_vec_vs * dot(tangent_dir_vs, view_vec_vs);
        let ortho_len = length(ortho_dir);
        if (ortho_len < 1e-5) {
            continue;
        }

        let slice_dir_perp_vs = ortho_dir / ortho_len;
        let axis_vec = safe_normalize(cross(slice_dir_perp_vs, view_vec_vs));
        let projected_normal = normal_vs - axis_vec * dot(normal_vs, axis_vec);
        let projected_normal_len = max(length(projected_normal), 1e-4);
        let projected_normal_cos = clamp(dot(projected_normal, view_vec_vs) / projected_normal_len, -1.0, 1.0);
        let projected_normal_sign = select(1.0, -1.0, dot(slice_dir_perp_vs, projected_normal) < 0.0);
        let normal_angle = projected_normal_sign * fast_acos(projected_normal_cos);

        var horizon_pos = cos(normal_angle + HALF_PI);
        var horizon_neg = cos(normal_angle - HALF_PI);

        let direction_probe_uv = project_view_to_uv(
            position_vs + slice_dir_perp_vs * max(settings.radius * 0.25, 0.05),
            view_index
        );
        let screen_dir = (direction_probe_uv - uv) * resolution;
        let screen_dir_len = length(screen_dir);
        if (screen_dir_len < 1e-4) {
            continue;
        }
        let screen_step_dir = screen_dir / screen_dir_len;

        for (var side = 0u; side < 2u; side = side + 1u) {
            let side_sign = select(-1.0, 1.0, side == 0u);
            for (var step = 0u; step < step_count; step = step + 1u) {
                let step_alpha = (f32(step) + 1.0 + radial_jitter) / f32(step_count);
                let step_px = step_alpha * radius_px;
                let sample_uv = uv + screen_step_dir * (step_px * side_sign) * inv_resolution;
                if (any(sample_uv <= vec2f(0.0)) || any(sample_uv >= vec2f(1.0))) {
                    break;
                }

                let hzb_depth = sample_hzb_depth(sample_uv, step_px * hzb_step_scale);
                if (hzb_depth >= 1.0) {
                    continue;
                }

                let sample_coord = uv_to_coord(sample_uv, full_resolution);
                let sample_depth = textureSampleLevel(hzb_tex, non_filtering_sampler, sample_uv, 0.0).r;
                if (sample_depth >= 1.0) {
                    continue;
                }

                let sample_position_vs = reconstruct_view_position(sample_uv, sample_depth, view_index);
                let delta_vs = sample_position_vs - position_vs;
                let sample_distance = length(delta_vs);
                if (sample_distance > settings.radius) {
                    continue;
                }

                let along = dot(delta_vs, slice_dir_perp_vs);
                if (along * side_sign <= settings.bias) {
                    continue;
                }

                let plane_height = dot(delta_vs, normal_vs);
                if (plane_height < -settings.thickness) {
                    continue;
                }

                let delta_plane = delta_vs - axis_vec * dot(delta_vs, axis_vec);
                let horizon_vec = safe_normalize(delta_plane);
                if (length(horizon_vec) < 1e-5) {
                    continue;
                }

                let horizon_cos = dot(horizon_vec, view_vec_vs);
                if (side_sign > 0.0) {
                    horizon_pos = max(horizon_pos, horizon_cos);
                } else {
                    horizon_neg = max(horizon_neg, horizon_cos);
                }
            }
        }

        let h0 = -fast_acos(clamp(horizon_neg, -1.0, 1.0));
        let h1 = fast_acos(clamp(horizon_pos, -1.0, 1.0));
        let contribution = integrate_slice(
            normal_angle,
            projected_normal_cos,
            h0,
            h1,
            projected_normal_len,
            slice_dir_perp_vs,
            view_vec_vs
        );
        let slice_visibility_max = max(
            integrate_slice(
                normal_angle,
                projected_normal_cos,
                normal_angle - HALF_PI,
                normal_angle + HALF_PI,
                projected_normal_len,
                slice_dir_perp_vs,
                view_vec_vs
            ).w,
            1e-4
        );
        let slice_visibility = clamp(contribution.w, 0.0, slice_visibility_max);
        let bent_scale = slice_visibility / max(contribution.w, 1e-4);

        visibility_accum += slice_visibility;
        visibility_weight_accum += slice_visibility_max;
        bent_accum_vs += contribution.xyz * bent_scale;
        valid_slice_count += 1u;
    }

    let ao_visibility = clamp(
        visibility_accum / max(visibility_weight_accum, max(1.0, f32(valid_slice_count)) * 1e-4),
        0.0,
        1.0
    );

    var bent_vs = normal_vs;
    if (visibility_accum > 1e-5 && dot(bent_accum_vs, bent_accum_vs) > 1e-6) {
        bent_vs = safe_normalize(bent_accum_vs / visibility_accum);
    }
    if (dot(bent_vs, normal_vs) < 0.0) {
        bent_vs = normal_vs;
    }

    let view_dir_ws = safe_normalize(view.view_direction.xyz);
    let view_right_ws = safe_normalize(view.view_right.xyz);
    let view_up_ws = safe_normalize(cross(view_right_ws, view_dir_ws));
    var bent_ws = safe_normalize(
        view_right_ws * bent_vs.x +
        view_up_ws * bent_vs.y +
        (-view_dir_ws) * bent_vs.z
    );
    if (dot(bent_ws, normal_ws) < 0.0) {
        bent_ws = normal_ws;
    }

    textureStore(ao_output, coord, vec4f(ao_visibility, ao_visibility, ao_visibility, 1.0));
    textureStore(bent_output, coord, vec4f(bent_ws, 1.0));
}

