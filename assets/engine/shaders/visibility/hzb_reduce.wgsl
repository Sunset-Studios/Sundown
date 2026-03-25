#include "common.wgsl"

// ------------------------------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------------------------------ 
struct HZBParams {
    input_image_size: vec2<f32>,
    output_image_size: vec2<f32>,
}

// ------------------------------------------------------------------------------------
// Compute Shader
// ------------------------------------------------------------------------------------ 

@group(1) @binding(0) var input_texture: texture_2d<f32>;
@group(1) @binding(1) var output_texture: texture_storage_2d<r32float, write>;
@group(1) @binding(2) var<uniform> params: HZBParams;

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= u32(params.output_image_size.x) || global_id.y >= u32(params.output_image_size.y)) {
        return;
    }

    let input_size = vec2<u32>(params.input_image_size);
    let src_min = vec2<u32>(
        u32(floor(f32(global_id.x) * params.input_image_size.x / params.output_image_size.x)),
        u32(floor(f32(global_id.y) * params.input_image_size.y / params.output_image_size.y))
    );
    let src_max = vec2<u32>(
        min(
            input_size.x,
            max(
                src_min.x + 1u,
                u32(ceil(f32(global_id.x + 1u) * params.input_image_size.x / params.output_image_size.x))
            )
        ),
        min(
            input_size.y,
            max(
                src_min.y + 1u,
                u32(ceil(f32(global_id.y + 1u) * params.input_image_size.y / params.output_image_size.y))
            )
        )
    );

    var max_depth = 0.0;
    for (var y = src_min.y; y < src_max.y; y = y + 1u) {
        for (var x = src_min.x; x < src_max.x; x = x + 1u) {
            max_depth = max(max_depth, textureLoad(input_texture, vec2<i32>(i32(x), i32(y)), 0).r);
        }
    }

    textureStore(output_texture, global_id.xy, vec4<f32>(max_depth));
}
