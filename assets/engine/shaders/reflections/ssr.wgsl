#include "common.wgsl"
#include "lighting_common.wgsl"

@group(1) @binding(0) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(1) var gbuffer_smra: texture_2d<f32>;
@group(1) @binding(2) var hzb_texture: texture_2d<f32>;
@group(1) @binding(3) var out_raycast_hit: texture_storage_2d<rgba16float, write>;

// Number of ray directions per pixel; we cycle through these each frame for stable temporal convergence.
const SSR_NUM_RAY_SAMPLES = 32u;
// Retries for ray direction when the sampled direction goes below the surface. Lower = faster, 4 is a good balance.
const SSR_RAY_DIR_RETRIES = 4u;
// Maximum number of steps to trace the ray.
const SSR_MAX_STEPS = 32u;

fn clip_to_projected_sample(clip: vec4<f32>) -> vec3<f32> {
    let ndc = clip.xyz / max(clip.w, epsilon);
    return vec3<f32>(
        ndc.x * 0.5 + 0.5,
        -ndc.y * 0.5 + 0.5,
        clamp(ndc.z, 0.0, 1.0)
    );
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
) -> vec3<f32> {
    let mip_count = textureNumLevels(hzb_texture);
    let max_trace_distance = mix(50.0, 16.0, roughness);
    let min_trace_distance = 0.05 + roughness * 0.15;
    let thickness = mix(0.008, 0.08, roughness * roughness);
    let view_projection = view_buffer[view_index].view_projection_matrix;

    let clip_origin = view_projection * vec4<f32>(origin, 1.0);
    let clip_step = view_projection * vec4<f32>(ray_dir, 0.0);

    var hit_uv = vec2<f32>(-1.0, -1.0);
    var hit_mask = 0.0;
    var mip_level: i32 = 0;

    for (var i = 0u; i < SSR_MAX_STEPS; i++) {
        let sample_t = clamp((f32(i) + 1.0 + step_jitter) / f32(SSR_MAX_STEPS), 0.0, 1.0);
        let step_t = mix(min_trace_distance, max_trace_distance, sample_t);
        let clip_sample = clip_origin + clip_step * step_t;
        let projected = clip_to_projected_sample(clip_sample);
        
        if (any(projected.xy < vec2<f32>(0.0)) || any(projected.xy > vec2<f32>(1.0))) { break; }

        let hzb_depth = textureSampleLevel(hzb_texture, non_filtering_sampler, projected.xy, f32(mip_level)).r;
        let depth_delta = hzb_depth - projected.z;
        let depth_tolerance = thickness * (1.0 + step_t * 0.02);

        mip_level = select(
            max(mip_level - 1, 0),
            min(mip_level + 1, i32(mip_count) - 1),
            depth_delta < -depth_tolerance
        );

        if (mip_level == 0 && abs(depth_delta) <= depth_tolerance) {
            let sample_pos = origin + ray_dir * step_t;
            let scene_pos = reconstruct_world_position(projected.xy, hzb_depth, view_index);
            let hit_error = distance(scene_pos, sample_pos);
            let hit_tolerance = 0.18 + roughness * 0.55 + step_t * 0.035;

            if (hit_error <= hit_tolerance) {
                hit_uv = projected.xy;
                hit_mask = 1.0;
                break;
            }
        }
    }

    return vec3<f32>(hit_uv, hit_mask);
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
        return;
    }

    let smra = textureLoad(gbuffer_smra, full_coord, 0);
    let roughness = clamp(smra.g, 0.0, 1.0);
    let metallic = smra.b;
    let reflectance = smra.r;
    let reflection_strength = (1.0 - roughness) * max(reflectance, metallic);

    if (reflection_strength <= 0.001 || roughness >= 0.7) {
        textureStore(out_raycast_hit, trace_coord, vec4<f32>(0.0));
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

    let out_hit = vec4<f32>(trace.xy, pdf, trace.z * reflection_strength);

    textureStore(out_raycast_hit, trace_coord, out_hit);
}

