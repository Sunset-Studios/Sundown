#include "common.wgsl"

@group(1) @binding(0) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(1) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(2) var gbuffer_smra: texture_2d<f32>;
@group(1) @binding(3) var hzb_texture: texture_2d<f32>;
@group(1) @binding(4) var out_raycast_hit: texture_storage_2d<rgba16float, write>;
@group(1) @binding(5) var out_raycast_mask: texture_storage_2d<rgba16float, write>;

// Number of ray directions per pixel; we cycle through these each frame for stable temporal convergence.
const SSR_NUM_RAY_SAMPLES = 16u;

fn project_to_uv(position: vec3f, view_index: u32) -> vec2f {
    let clip = view_buffer[view_index].view_projection_matrix * vec4f(position, 1.0);
    let ndc = clip.xyz / max(clip.w, epsilon);
    return vec2f(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
}

fn project_to_depth01(position: vec3f, view_index: u32) -> f32 {
    let clip = view_buffer[view_index].view_projection_matrix * vec4f(position, 1.0);
    let ndc_z = clip.z / max(clip.w, epsilon);
    return clamp(ndc_z, 0.0, 1.0);
}

fn uv_to_coord(uv: vec2f, resolution: vec2<u32>) -> vec2<i32> {
    let pixel = vec2<i32>(floor(uv * vec2f(f32(resolution.x), f32(resolution.y))));
    return vec2<i32>(
        clamp(pixel.x, 0, i32(resolution.x) - 1),
        clamp(pixel.y, 0, i32(resolution.y) - 1)
    );
}

fn tangent_basis(n: vec3f) -> mat3x3f {
    let up = select(vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0), abs(n.z) > 0.9);
    let t = safe_normalize(cross(up, n));
    let b = cross(n, t);
    return mat3x3f(t, b, n);
}

fn sample_ggx_half_vector(xi: vec2f, normal: vec3f, roughness: f32) -> vec3f {
    let a = roughness * roughness;
    let a2 = a * a;
    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / max(0.0001, 1.0 + (a2 - 1.0) * xi.y));
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let h_local = vec3f(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    return safe_normalize(tangent_basis(normal) * h_local);
}

fn d_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = max(roughness * roughness, 0.0001);
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 0.0001);
}

fn ggx_pdf(normal: vec3f, view_dir: vec3f, sample_dir: vec3f, roughness: f32) -> f32 {
    let h = safe_normalize(view_dir + sample_dir);
    let n_dot_h = max(dot(normal, h), 0.0001);
    let v_dot_h = max(dot(view_dir, h), 0.0001);
    let d = d_ggx(n_dot_h, roughness);
    return max(d * n_dot_h / max(4.0 * v_dot_h, 0.0001), 1e-5);
}

fn trace_hiz(
    origin: vec3f,
    ray_dir: vec3f,
    normal: vec3f,
    view_index: u32,
    roughness: f32,
    resolution: vec2<u32>,
    step_jitter: f32
) -> vec4f {
    let mip_count = textureNumLevels(hzb_texture);
    let max_steps = u32(floor(mix(48.0, 20.0, roughness)));
    let max_trace_distance = mix(1000.0, 36.0, roughness);
    let min_trace_distance = 0.05 + roughness * 0.15;
    let distance_curve_power = mix(1.45, 1.15, roughness);
    let thickness = mix(0.01, 0.2, roughness * roughness);

    var hit_uv = vec2f(-1.0, -1.0);
    var hit_depth = 0.0;
    var hit_mask = 0.0;
    var mip_level: i32 = 0;

    for (var i = 0u; i < max_steps; i++) {
        let sample_t = clamp((f32(i) + 1.0 + step_jitter) / f32(max_steps), 0.0, 1.0);
        let step_t = mix(min_trace_distance, max_trace_distance, pow(sample_t, distance_curve_power));
        let sample_pos = origin + ray_dir * step_t;

        let uv = project_to_uv(sample_pos, view_index);
        if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) {
            break;
        }

        let projected_depth = project_to_depth01(sample_pos, view_index);
        let mip = u32(clamp(mip_level, 0, i32(mip_count) - 1));
        let hzb_depth = textureSampleLevel(hzb_texture, non_filtering_sampler, uv, f32(mip)).r;
        let depth_delta = hzb_depth - projected_depth;
        let depth_tolerance = thickness * (1.0 + step_t * 0.05);

        if (depth_delta < -depth_tolerance) {
            mip_level = min(mip_level + 1, i32(mip_count) - 1);
            continue;
        }

        mip_level = max(mip_level - 1, 0);

        if (abs(depth_delta) <= depth_tolerance * 1.5) {
            let hit_coord = uv_to_coord(uv, resolution);
            let scene_pos = textureLoad(gbuffer_position, hit_coord, 0).xyz;
            let scene_normal = safe_normalize(textureLoad(gbuffer_normal, hit_coord, 0).xyz);
            let hit_error = distance(scene_pos, sample_pos);
            let facing = dot(scene_normal, -ray_dir);
            let normal_ok = dot(scene_normal, normal) > -0.3;
            let hit_tolerance = (0.18 + roughness * 0.55) + step_t * 0.035;

            if (hit_error < hit_tolerance && facing > 0.01 && normal_ok) {
                hit_uv = uv;
                hit_depth = project_to_depth01(scene_pos, view_index);
                hit_mask = 1.0;
                break;
            }
        }
    }

    return vec4f(hit_uv, hit_depth, hit_mask);
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let resolution = vec2<u32>(u32(frame_info.resolution.x), u32(frame_info.resolution.y));
    if (gid.x >= resolution.x || gid.y >= resolution.y) {
        return;
    }

    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let normal_data = textureLoad(gbuffer_normal, coord, 0).xyz;
    if (length(normal_data) < 1e-5) {
        textureStore(out_raycast_hit, coord, vec4f(0.0));
        textureStore(out_raycast_mask, coord, vec4f(0.0));
        return;
    }

    let smra = textureLoad(gbuffer_smra, coord, 0);
    let roughness = clamp(smra.g, 0.0, 1.0);
    let metallic = smra.b;
    let reflectance = smra.r;
    let reflection_strength = (1.0 - roughness) * max(reflectance, metallic);

    if (reflection_strength <= 0.001 || roughness >= 0.995) {
        textureStore(out_raycast_hit, coord, vec4f(0.0));
        textureStore(out_raycast_mask, coord, vec4f(0.0));
        return;
    }

    let position = textureLoad(gbuffer_position, coord, 0).xyz;
    let view_index = u32(frame_info.view_index);
    let normal = safe_normalize(normal_data);
    let view_dir = safe_normalize(view_buffer[view_index].view_position.xyz - position);

    // Low-discrepancy jitter: same (pixel, frame) always gets the same ray; we cycle over SSR_NUM_RAY_SAMPLES.
    let pixel_id = gid.x + gid.y * resolution.x;
    let sample_phase = hash(pixel_id) % SSR_NUM_RAY_SAMPLES;
    let sample_idx = (u32(frame_info.frame_index) + sample_phase) % SSR_NUM_RAY_SAMPLES;
    var xi = rand_halton_2d(pixel_id, sample_idx);
    xi.y = mix(xi.y, 0.0, 0.7);

    let h = sample_ggx_half_vector(xi, normal, max(roughness, 0.05));
    let ray_dir = safe_normalize(reflect(-view_dir, h));

    if (dot(normal, ray_dir) <= 0.0) {
        textureStore(out_raycast_hit, coord, vec4f(0.0));
        textureStore(out_raycast_mask, coord, vec4f(0.0));
        return;
    }

    let pdf = ggx_pdf(normal, view_dir, ray_dir, max(roughness, 0.05));
    let step_jitter = xi.x - 0.5;
    let trace = trace_hiz(
        position + normal * 0.04,
        ray_dir,
        normal,
        view_index,
        roughness,
        resolution,
        step_jitter
    );

    let valid_hit = trace.w > 0.5 && all(trace.xy >= vec2f(0.0)) && all(trace.xy <= vec2f(1.0));
    let out_hit = select(vec4f(0.0), vec4f(trace.xy, trace.z, pdf), valid_hit);
    let out_mask = select(vec4f(0.0), vec4f(trace.w * reflection_strength), valid_hit);

    textureStore(out_raycast_hit, coord, out_hit);
    textureStore(out_raycast_mask, coord, out_mask);
}

