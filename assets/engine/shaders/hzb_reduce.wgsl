#include "common.wgsl"

struct HZBParams {
    image_size: vec2<f32>,
}

@group(1) @binding(0) var input_texture: texture_2d<f32>;
@group(1) @binding(1) var output_texture: texture_storage_2d<r32float, write>;
@group(1) @binding(2) var<uniform> params: HZBParams;

@compute @workgroup_size(16, 16, 1)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let uv = (vec2<f32>(global_id.xy) + vec2<f32>(0.5)) / params.image_size;

    let depth = textureSampleLevel(input_texture, non_filtering_sampler, uv, 0).x;

    textureStore(output_texture, vec2<i32>(global_id.xy), vec4<f32>(depth));
}