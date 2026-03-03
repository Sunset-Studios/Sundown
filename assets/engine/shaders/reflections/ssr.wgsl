#include "common.wgsl"

@group(1) @binding(0) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(1) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(2) var gbuffer_smra: texture_2d<f32>;
@group(1) @binding(3) var skybox_texture: texture_2d<f32>;
@group(1) @binding(4) var out_reflections: texture_storage_2d<rgba16float, write>;

fn project_to_uv(position: vec3f, view_index: u32) -> vec2f {
    let clip = view_buffer[view_index].view_projection_matrix * vec4f(position, 1.0);
    let ndc = clip.xyz / clip.w;
    return vec2f(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let resolution = vec2<u32>(u32(frame_info.resolution.x), u32(frame_info.resolution.y));
    if (gid.x >= resolution.x || gid.y >= resolution.y) {
        return;
    }

    let coord = vec2<i32>(i32(gid.x), i32(gid.y));

    let normal_data = textureLoad(gbuffer_normal, coord, 0);
    let normal_len = length(normal_data.xyz);
    if (normal_len <= 0.0) {
        textureStore(out_reflections, coord, vec4f(0.0));
        return;
    }

    let smra = textureLoad(gbuffer_smra, coord, 0);
    let roughness = smra.g;
    let metallic = smra.b;
    let reflectance = smra.r;

    let reflection_strength = (1.0 - roughness) * max(reflectance, metallic);
    if (reflection_strength <= 0.001) {
        textureStore(out_reflections, coord, vec4f(0.0));
        return;
    }

    let position = textureLoad(gbuffer_position, coord, 0).xyz;
    let view_index = u32(frame_info.view_index);
    let view_dir = normalize(position - view_buffer[view_index].view_position.xyz);
    let normal = normalize(normal_data.xyz);
    let reflection_dir = normalize(reflect(view_dir, normal));

    var hit_color = vec3f(0.0);
    var hit = false;

    let stride = 0.5;
    let max_steps = 32u;
    var march_position = position + normal * 0.03;

    for (var step = 0u; step < max_steps; step++) {
        march_position += reflection_dir * stride;
        let sample_uv = project_to_uv(march_position, view_index);

        if (any(sample_uv < vec2f(0.0)) || any(sample_uv > vec2f(1.0))) {
            break;
        }

        let sample_coord = vec2<i32>(sample_uv * vec2f(f32(resolution.x - 1u), f32(resolution.y - 1u)));
        let scene_pos = textureLoad(gbuffer_position, sample_coord, 0).xyz;

        let depth_delta = distance(scene_pos, march_position);
        if (depth_delta < 0.25) {
            let sample_normal = textureLoad(gbuffer_normal, sample_coord, 0).xyz;
            let facing = max(dot(normalize(sample_normal), -reflection_dir), 0.0);
            hit_color = textureLoad(skybox_texture, sample_coord, 0).rgb * facing;
            hit = true;
            break;
        }
    }

    if (!hit) {
        let uv = vec2f(f32(gid.x) / max(1.0, f32(resolution.x - 1u)), f32(gid.y) / max(1.0, f32(resolution.y - 1u)));
        hit_color = textureSampleLevel(skybox_texture, clamped_sampler, uv, 0.0).rgb * 0.2;
    }

    textureStore(out_reflections, coord, vec4f(hit_color * reflection_strength, 1.0));
}
