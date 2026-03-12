#include "common.wgsl"
#include "lighting_common.wgsl"

@group(1) @binding(0) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(1) var gbuffer_smra: texture_2d<f32>;
@group(1) @binding(2) var hzb_texture: texture_2d<f32>;
@group(1) @binding(3) var out_raycast_hit: texture_storage_2d<rgba16float, write>;
@group(1) @binding(4) var out_raycast_mask: texture_storage_2d<rgba16float, write>;

// Number of ray directions per pixel; we cycle through these each frame for stable temporal convergence.
const SSR_NUM_RAY_SAMPLES = 32u;
// Retries for ray direction when the sampled direction goes below the surface. Lower = faster, 4 is a good balance.
const SSR_RAY_DIR_RETRIES = 4u;

fn project_to_uv(position: vec3<f32>, view_index: u32) -> vec2<f32> {
    let clip = view_buffer[view_index].view_projection_matrix * vec4<f32>(position, 1.0);
    let ndc = clip.xyz / max(clip.w, epsilon);
    return vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
}

fn project_to_depth01(position: vec3<f32>, view_index: u32) -> f32 {
    let clip = view_buffer[view_index].view_projection_matrix * vec4<f32>(position, 1.0);
    let ndc_z = clip.z / max(clip.w, epsilon);
    return clamp(ndc_z, 0.0, 1.0);
}

fn trace_to_full_coord(trace_coord: vec2<u32>, full_resolution: vec2<u32>, trace_resolution: vec2<u32>) -> vec2<i32> {
    let trace_uv = (vec2<f32>(trace_coord) + 0.5) / vec2<f32>(trace_resolution);
    let full_pixel = vec2<i32>(floor(trace_uv * vec2<f32>(full_resolution)));
    return vec2<i32>(
        clamp(full_pixel.x, 0, i32(full_resolution.x) - 1),
        clamp(full_pixel.y, 0, i32(full_resolution.y) - 1)
    );
}

fn trace_hiz(
    origin: vec3<f32>,
    ray_dir: vec3<f32>,
    normal: vec3<f32>,
    view_index: u32,
    roughness: f32,
    resolution: vec2<u32>,
    step_jitter: f32,
) -> vec4<f32> {
    let mip_count = textureNumLevels(hzb_texture);
    let max_steps = u32(floor(mix(48.0, 16.0, roughness)));
    let max_trace_distance = mix(100.0, 24.0, roughness);
    let min_trace_distance = 0.05 + roughness * 0.15;
    let distance_curve_power = mix(1.45, 1.15, roughness);
    let thickness = mix(0.01, 0.2, roughness * roughness);

    var hit_uv = vec2<f32>(-1.0, -1.0);
    var hit_depth = 0.0;
    var hit_mask = 0.0;
    var mip_level: i32 = 0;

    for (var i = 0u; i < max_steps; i++) {
        let sample_t = clamp((f32(i) + 1.0 + step_jitter) / f32(max_steps), 0.0, 1.0);
        let step_t = mix(min_trace_distance, max_trace_distance, pow(sample_t, distance_curve_power));
        let sample_pos = origin + ray_dir * step_t;

        let uv = project_to_uv(sample_pos, view_index);
        if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) {
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
            let scene_depth = textureSampleLevel(hzb_texture, non_filtering_sampler, uv, 0.0).r;
            let scene_pos = reconstruct_world_position(uv, scene_depth, view_index);
            let scene_normal = safe_normalize(textureLoad(gbuffer_normal, hit_coord, 0).xyz);
            let hit_error = distance(scene_pos, sample_pos);
            let facing = dot(scene_normal, -ray_dir);
            let normal_ok = dot(scene_normal, normal) > -0.3;
            let hit_tolerance = (0.18 + roughness * 0.55) + step_t * 0.035;

            if (hit_error < hit_tolerance && facing > 0.01 && normal_ok) {
                hit_uv = uv;
                hit_depth = hzb_depth;
                hit_mask = 1.0;
                break;
            }
        }
    }

    return vec4<f32>(hit_uv, hit_depth, hit_mask);
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let full_resolution = vec2<u32>(u32(frame_info.resolution.x), u32(frame_info.resolution.y));
    let trace_resolution = textureDimensions(out_raycast_hit);
    if (gid.x >= trace_resolution.x || gid.y >= trace_resolution.y) {
        return;
    }

    let trace_coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let full_coord = trace_to_full_coord(gid.xy, full_resolution, trace_resolution);
    let full_uv = (vec2<f32>(full_coord) + 0.5) / vec2<f32>(full_resolution);

    let normal_data = textureLoad(gbuffer_normal, full_coord, 0).xyz;
    if (length(normal_data) < 1e-5) {
        textureStore(out_raycast_hit, trace_coord, vec4<f32>(0.0));
        textureStore(out_raycast_mask, trace_coord, vec4<f32>(0.0));
        return;
    }

    let smra = textureLoad(gbuffer_smra, full_coord, 0);
    let roughness = clamp(smra.g, 0.0, 1.0);
    let metallic = smra.b;
    let reflectance = smra.r;
    let reflection_strength = (1.0 - roughness) * max(reflectance, metallic);

    if (reflection_strength <= 0.001 || roughness >= 0.7) {
        textureStore(out_raycast_hit, trace_coord, vec4<f32>(0.0));
        textureStore(out_raycast_mask, trace_coord, vec4<f32>(0.0));
        return;
    }

    let view_index = u32(frame_info.view_index);
    let depth = textureSampleLevel(hzb_texture, non_filtering_sampler, full_uv, 0.0).r;
    let position = reconstruct_world_position(full_uv, depth, view_index);
    let normal = safe_normalize(normal_data);
    let view_dir = safe_normalize(view_buffer[view_index].view_position.xyz - position);

    // Low-discrepancy jitter: same (pixel, frame) always gets the same ray; we cycle over SSR_NUM_RAY_SAMPLES.
    let pixel_id = gid.x + gid.y * trace_resolution.x;
    let sample_phase = hash(pixel_id) % SSR_NUM_RAY_SAMPLES;
    let sample_idx = (u32(frame_info.frame_index) + sample_phase) % SSR_NUM_RAY_SAMPLES;

    var ray_dir = normal.xyz;
    var xi = vec2<f32>(0.0);
    var pdf = 0.0;
    for (var retry = 0u; retry < SSR_RAY_DIR_RETRIES; retry++) {
        let sample_idx_retry = (sample_idx + retry * SSR_NUM_RAY_SAMPLES) % SSR_NUM_RAY_SAMPLES;
        xi = rand_halton_2d(pixel_id, sample_idx_retry);
        xi.y = mix(xi.y, 0.0, 0.7);
        let h = sample_ggx(normal, max(roughness, 0.001), xi.x, xi.y);
        ray_dir = safe_normalize(reflect(-view_dir, h));
        if (dot(normal, ray_dir) > 0.0) {
            pdf = ggx_pdf(normal, view_dir, ray_dir, max(roughness, 0.001));
            break;
        }
    }
    let step_jitter = xi.x - 0.5;
    let trace = trace_hiz(
        position + normal * 0.0001,
        ray_dir,
        normal,
        view_index,
        roughness,
        full_resolution,
        step_jitter
    );

    let valid_hit = trace.w > 0.0;
    let out_hit = select(vec4<f32>(0.0), vec4<f32>(trace.xy, trace.z, pdf), valid_hit);
    let out_mask = select(vec4<f32>(0.0), vec4<f32>(trace.w * reflection_strength), valid_hit);

    textureStore(out_raycast_hit, trace_coord, out_hit);
    textureStore(out_raycast_mask, trace_coord, out_mask);
}

