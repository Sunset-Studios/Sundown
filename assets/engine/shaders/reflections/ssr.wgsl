#include "common.wgsl"

@group(1) @binding(0) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(1) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(2) var gbuffer_smra: texture_2d<f32>;
@group(1) @binding(3) var lighting_history_texture: texture_2d<f32>;
@group(1) @binding(4) var hzb_texture: texture_2d<f32>;
@group(1) @binding(5) var out_trace: texture_storage_2d<rgba16float, write>;

fn project_to_uv(position: vec3f, view_index: u32) -> vec2f {
    let clip = view_buffer[view_index].view_projection_matrix * vec4f(position, 1.0);
    let ndc = clip.xyz / max(clip.w, epsilon);
    return vec2f(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
}

fn project_to_depth01(position: vec3f, view_index: u32) -> f32 {
    let clip = view_buffer[view_index].view_projection_matrix * vec4f(position, 1.0);
    let ndc_z = clip.z / max(clip.w, epsilon);
    return clamp(ndc_z * 0.5 + 0.5, 0.0, 1.0);
}

fn uv_to_coord(uv: vec2f, resolution: vec2<u32>) -> vec2<i32> {
    let max_coord = vec2f(f32(max(1u, resolution.x) - 1u), f32(max(1u, resolution.y) - 1u));
    return vec2<i32>(clamp(uv * max_coord, vec2f(0.0), max_coord));
}

fn noise_2d(seed: u32) -> vec2f {
    let x = rand_float(seed * 1664525u + 1013904223u);
    let y = rand_float(seed * 22695477u + 1u);
    return vec2f(x, y);
}

fn tangent_basis(n: vec3f) -> mat3x3f {
    let up = select(vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0), abs(n.z) > 0.9);
    let t = safe_normalize(cross(up, n));
    let b = cross(n, t);
    return mat3x3f(t, b, n);
}

fn rough_reflection_dir(reflect_dir: vec3f, roughness: f32, jitter: vec2f) -> vec3f {
    let angle = jitter.x * 2.0 * PI;
    let radius = sqrt(jitter.y) * roughness * roughness;
    let local = vec3f(cos(angle) * radius, sin(angle) * radius, sqrt(max(0.0, 1.0 - radius * radius)));
    let basis = tangent_basis(reflect_dir);
    let jittered = safe_normalize(basis * local);
    return safe_normalize(mix(reflect_dir, jittered, clamp(roughness, 0.0, 1.0)));
}

fn trace_hiz(origin: vec3f, ray_dir: vec3f, normal: vec3f, view_index: u32, roughness: f32, resolution: vec2<u32>, step_jitter: f32) -> vec4f {
    let mip_count = textureNumLevels(hzb_texture);
    let max_steps = u32(floor(mix(16.0, 4.0, roughness)));
    let max_trace_distance = mix(80.0, 28.0, roughness);
    let min_trace_distance = 0.08 + roughness * 0.18;
    let distance_curve_power = mix(1.55, 1.25, roughness);
    let thickness = mix(0.015, 0.2, roughness * roughness);
    var hit_uv = vec2f(-1.0);
    var hit_confidence = 0.0;

    for (var i = 0u; i < max_steps; i++) {
        let sample_t = clamp((f32(i) + 1.0 + step_jitter) / f32(max_steps), 0.0, 1.0);
        let step_t = mix(min_trace_distance, max_trace_distance, pow(sample_t, distance_curve_power));

        let sample_pos = origin + ray_dir * step_t;
        let uv = project_to_uv(sample_pos, view_index);
        if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) {
            break;
        }

        let projected_depth = project_to_depth01(sample_pos, view_index);
        let cone_mip = clamp(u32(floor(f32(mip_count - 1u) * clamp(roughness + f32(i) * 0.02, 0.0, 1.0))), 0u, mip_count - 1u);
        let hzb_depth = textureSampleLevel(hzb_texture, non_filtering_sampler, uv, f32(cone_mip)).r;
        let depth_delta = hzb_depth - projected_depth;
        let depth_tolerance = thickness * (1.0 + step_t * 0.08);

        if (depth_delta >= -depth_tolerance && depth_delta <= depth_tolerance * 1.35) {
            let hit_coord = uv_to_coord(uv, resolution);
            let scene_pos = textureLoad(gbuffer_position, hit_coord, 0).xyz;
            let scene_normal = safe_normalize(textureLoad(gbuffer_normal, hit_coord, 0).xyz);
            let hit_error = distance(scene_pos, sample_pos);
            let facing = dot(scene_normal, -ray_dir);
            let normal_ok = dot(scene_normal, normal) > -0.2;
            let hit_tolerance = (0.2 + roughness * 0.45) + step_t * 0.05;
            if (hit_error < hit_tolerance && facing > 0.03 && normal_ok) {
                hit_uv = uv;
                hit_confidence = clamp((1.0 - roughness * 0.55) * facing * exp(-2.0 * hit_error), 0.0, 1.0);
                break;
            }
        }
    }

    if (hit_confidence <= 0.0) {
        return vec4f(0.0);
    }

    let hit_coord = uv_to_coord(hit_uv, resolution);
    let hit_color = textureLoad(lighting_history_texture, hit_coord, 0).rgb;
    return vec4f(hit_color, hit_confidence);
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
        textureStore(out_trace, coord, vec4f(0.0));
        return;
    }

    let smra = textureLoad(gbuffer_smra, coord, 0);
    let roughness = clamp(smra.g, 0.0, 1.0);
    let metallic = smra.b;
    let reflectance = smra.r;
    let reflection_strength = (1.0 - roughness) * max(reflectance, metallic);

    if (reflection_strength <= 0.001) {
        textureStore(out_reflections, coord, vec4f(0.0));
        return;
    }

    let position = textureLoad(gbuffer_position, coord, 0).xyz;
    let view_index = u32(frame_info.view_index);
    let normal = safe_normalize(normal_data);
    let view_dir = safe_normalize(position - view_buffer[view_index].view_position.xyz);
    let base_reflection = safe_normalize(reflect(view_dir, normal));

    let seed = u32(gid.x + gid.y * resolution.x) ^ (u32(frame_info.frame_index) * 747796405u);
    let ray_jitter = noise_2d(seed);
    let reflection_dir = rough_reflection_dir(base_reflection, roughness, ray_jitter);
    let step_jitter = ray_jitter.x - 0.5;
    let trace = trace_hiz(
        position + normal * 0.04,
        reflection_dir,
        normal,
        view_index,
        roughness,
        resolution,
        step_jitter
    );

    let uv = vec2f(f32(gid.x) / max(1.0, f32(resolution.x - 1u)), f32(gid.y) / max(1.0, f32(resolution.y - 1u)));
    let fallback_color = textureSampleLevel(lighting_history_texture, clamped_sampler, uv, roughness * 4.0).rgb * 0.35;
    let hit_color = mix(fallback_color, trace.rgb, trace.a);
    let confidence = trace.a * reflection_strength;
    textureStore(out_trace, coord, vec4f(hit_color * reflection_strength, confidence));
}
