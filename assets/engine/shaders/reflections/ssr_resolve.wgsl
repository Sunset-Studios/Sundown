#include "common.wgsl"

@group(1) @binding(0) var raycast_hit_texture: texture_2d<f32>;
@group(1) @binding(1) var raycast_mask_texture: texture_2d<f32>;
@group(1) @binding(2) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(3) var depth_texture: texture_2d<f32>;
@group(1) @binding(4) var gbuffer_smra: texture_2d<f32>;
@group(1) @binding(5) var lighting_history_texture: texture_2d<f32>;
@group(1) @binding(6) var motion_emissive_texture: texture_2d<f32>;
@group(1) @binding(7) var out_resolve: texture_storage_2d<rgba16float, write>;

const EDGE_FACTOR = 1.25;
const resolve_offsets = array<vec2<i32>, 4>(
    vec2<i32>(0, 0),
    vec2<i32>(0, 1),
    vec2<i32>(1, -1),
    vec2<i32>(-1, -1),
);

fn full_to_trace_coord(coord: vec2<i32>, full_resolution: vec2<u32>, trace_resolution: vec2<u32>) -> vec2<i32> {
    let x = min(i32((u32(max(coord.x, 0)) * trace_resolution.x) / max(full_resolution.x, 1u)), i32(trace_resolution.x) - 1);
    let y = min(i32((u32(max(coord.y, 0)) * trace_resolution.y) / max(full_resolution.y, 1u)), i32(trace_resolution.y) - 1);
    return vec2<i32>(x, y);
}

fn ray_atten_border(pos: vec2<f32>, value: f32) -> f32 {
    let border_dist = min(1.0 - max(pos.x, pos.y), min(pos.x, pos.y));
    return clamp(select(border_dist / max(value, 1e-4), 1.0, border_dist > value), 0.0, 1.0);
}

fn d_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = max(roughness * roughness, 0.0001);
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 0.0001);
}

fn v_smith_ggx_height_correlated_fast(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return 0.5 / max(mix(2.0 * n_dot_l * n_dot_v, n_dot_l + n_dot_v, roughness), 0.0001);
}

fn f_schlick_vec3(f0: vec3<f32>, f90: f32, v_dot_h: f32) -> vec3<f32> {
    let one_minus_voh = 1.0 - v_dot_h;
    let one_minus_voh_2 = one_minus_voh * one_minus_voh;
    return f0 + (f90 - f0) * one_minus_voh_2 * one_minus_voh_2 * one_minus_voh;
}

fn brdf_importance_weight(
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    light_dir: vec3<f32>,
    roughness: f32,
    metallic: f32,
    reflectance: f32
) -> f32 {
    let halfway = safe_normalize(light_dir + view_dir);
    let n_dot_v = max(dot(normal, view_dir), 0.0001);
    let n_dot_l = max(dot(normal, light_dir), 0.0001);
    let n_dot_h = max(dot(normal, halfway), 0.0001);
    let v_dot_h = max(dot(view_dir, halfway), 0.0001);

    let dielectric_f0 = 0.16 * reflectance * reflectance;
    let f0 = mix(vec3<f32>(dielectric_f0), vec3<f32>(1.0), metallic);
    let f = f_schlick_vec3(f0, 1.0, v_dot_h);

    let r = clamp(roughness, 0.02, 1.0);
    let d = d_ggx(n_dot_h, r);
    let v = v_smith_ggx_height_correlated_fast(n_dot_v, n_dot_l, r);
    let specular = (d * v) * f;

    let kd = (1.0 - metallic) * (vec3<f32>(1.0) - f);
    let diffuse = kd * vec3<f32>(1.0 / PI);
    let brdf = (diffuse + specular) * n_dot_l;

    return max(luminance(brdf), 1e-4);
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let full_resolution = vec2<u32>(u32(frame_info.resolution.x), u32(frame_info.resolution.y));
    if (gid.x >= full_resolution.x || gid.y >= full_resolution.y) {
        return;
    }

    let trace_resolution = textureDimensions(raycast_hit_texture);
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let trace_coord = full_to_trace_coord(coord, full_resolution, trace_resolution);

    let normal_data = textureLoad(gbuffer_normal, coord, 0).xyz;
    if (length(normal_data) < 1e-5) {
        textureStore(out_resolve, coord, vec4<f32>(0.0));
        return;
    }

    let smra = textureLoad(gbuffer_smra, coord, 0);
    let roughness = clamp(smra.g, 0.0, 1.0);
    let metallic = smra.b;
    let reflectance = smra.r;
    let normal = safe_normalize(normal_data);

    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) / vec2<f32>(f32(full_resolution.x), f32(full_resolution.y));
    let view_index = u32(frame_info.view_index);
    let depth = textureLoad(depth_texture, coord, 0).r;
    let position = reconstruct_world_position(uv, depth, view_index);
    let view_dir = safe_normalize(view_buffer[view_index].view_position.xyz - position);

    let max_mip = max(0.0, f32(textureNumLevels(lighting_history_texture) - 1u));

    var accum = vec3<f32>(0.0);
    var accum_weight = 0.0;
    var confidence_sum = 0.0;

    let stride = 1 + i32(round(roughness * 2.0));

    for (var i = 0u; i < 4u; i++) {
        let tap_coord = vec2<i32>(
            clamp(trace_coord.x + resolve_offsets[i].x * stride, 0, i32(trace_resolution.x) - 1),
            clamp(trace_coord.y + resolve_offsets[i].y * stride, 0, i32(trace_resolution.y) - 1)
        );

        let hit = textureLoad(raycast_hit_texture, tap_coord, 0);
        let mask = textureLoad(raycast_mask_texture, tap_coord, 0).r;

        if (mask <= 1e-4 || any(hit.xy < vec2<f32>(0.0)) || any(hit.xy > vec2<f32>(1.0))) {
            continue;
        }

        let hit_coord = uv_to_coord(hit.xy, full_resolution);
        let hit_depth = textureLoad(depth_texture, hit_coord, 0).r;
        if (hit_depth >= 1.0) {
            continue;
        }
        let hit_pos = reconstruct_world_position(hit.xy, hit_depth, view_index);

        let light_dir = safe_normalize(hit_pos - position);

        let brdf_w = brdf_importance_weight(normal, view_dir, light_dir, roughness, metallic, reflectance);
        let pdf = max(hit.w, 1e-4);
        let weight = brdf_w / pdf;

        let cone_tangent = roughness * roughness;
        let pixel_footprint = length(hit.xy - uv) * max(f32(full_resolution.x), f32(full_resolution.y));
        let source_mip = clamp(log2(max(1.0, cone_tangent * pixel_footprint)), 0.0, max_mip);

        let hit_emissive = textureLoad(motion_emissive_texture, hit_coord, 0).w;
        var sample_color = textureSampleLevel(lighting_history_texture, clamped_sampler, hit.xy, source_mip).rgb;
        sample_color += sample_color * hit_emissive;
        sample_color = sample_color / (1.0 + luminance(sample_color));

        let sample_alpha = ray_atten_border(hit.xy, EDGE_FACTOR) * mask;
        accum += sample_color * weight;
        accum_weight += weight;
        confidence_sum += sample_alpha;
    }

    let mask = textureLoad(raycast_mask_texture, trace_coord, 0).r;
    let fallback_color = textureSampleLevel(lighting_history_texture, clamped_sampler, uv, roughness * 4.0).rgb * 0.1 * mask;
    var resolved_color = accum / max(accum_weight, 1e-4);

    let confidence = clamp(confidence_sum * 0.25, 0.0, 1.0);
    let final_color = mix(fallback_color, resolved_color, confidence);

    textureStore(out_resolve, coord, vec4<f32>(final_color, confidence));
}

