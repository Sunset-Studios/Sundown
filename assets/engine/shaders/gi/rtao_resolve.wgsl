#include "common.wgsl"
#include "gi/gi_common.wgsl"

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read> pixel_path_state: array<AOPixelPathState>;
@group(1) @binding(2) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(3) var ao_output: texture_storage_2d<r32float, write>;
@group(1) @binding(4) var bent_normal_output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.xy;
    let full_resolution = vec2<u32>(u32(gi_params.full_resolution_x), u32(gi_params.full_resolution_y));
    let gi_resolution = vec2<u32>(u32(gi_params.gi_resolution_x), u32(gi_params.gi_resolution_y));
    let upscale_factor = max(1u, u32(gi_params.upscale_factor));
    let rays_per_pixel = max(1u, u32(gi_params.screen_ray_count));

    if (pixel.x >= full_resolution.x || pixel.y >= full_resolution.y) {
        return;
    }

    let gi_pixel = min(pixel / vec2<u32>(upscale_factor), gi_resolution - vec2<u32>(1u));
    let pixel_index = gi_pixel.y * gi_resolution.x + gi_pixel.x;
    let ray_base = pixel_index * rays_per_pixel;

    var total_ao_weight = 0.0;
    for (var ray_idx = 0u; ray_idx < rays_per_pixel; ray_idx++) {
        let path = pixel_path_state[ray_base + ray_idx];
        let weight = select(0.0, max(0.0, 1.0 - path.origin_tmin.w / gi_params.max_ray_length), path.state_u32.w != 0xffffffffu);
        total_ao_weight += weight;
    }

    let ao = 1.0 - (total_ao_weight / f32(rays_per_pixel));
    let normal = textureLoad(gbuffer_normal, vec2<i32>(pixel), 0).xyz;
    let bent = select(vec3<f32>(0.0, 1.0, 0.0), normalize(normal), dot(normal, normal) > 0.0);

    textureStore(ao_output, vec2<i32>(pixel), vec4<f32>(ao, 0.0, 0.0, 1.0));
    textureStore(bent_normal_output, vec2<i32>(pixel), vec4<f32>(bent, 1.0));
}
