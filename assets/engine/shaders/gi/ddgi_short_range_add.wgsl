#include "common.wgsl"
#include "postprocess_common.wgsl"

@group(1) @binding(0) var ddgi_direct: texture_2d<f32>;
@group(1) @binding(1) var ddgi_indirect_diffuse: texture_2d<f32>;
@group(1) @binding(2) var ddgi_indirect_specular: texture_2d<f32>;
@group(1) @binding(3) var short_range_direct: texture_2d<f32>;
@group(1) @binding(4) var short_range_indirect_diffuse: texture_2d<f32>;
@group(1) @binding(5) var short_range_indirect_specular: texture_2d<f32>;
@group(1) @binding(6) var out_direct: texture_storage_2d<rgba16float, write>;
@group(1) @binding(7) var out_indirect_diffuse: texture_storage_2d<rgba16float, write>;
@group(1) @binding(8) var out_indirect_specular: texture_storage_2d<rgba16float, write>;

fn combine_short_range_modulation(
    base_value: vec3<f32>,
    detail_value: vec3<f32>
) -> vec3<f32> {
    return base_value * detail_value;
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let resolution = textureDimensions(out_direct);
    if (gid.x >= resolution.x || gid.y >= resolution.y) {
        return;
    }

    let pixel_coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let base_direct = textureLoad(ddgi_direct, pixel_coord, 0);
    let base_indirect_diffuse = textureLoad(ddgi_indirect_diffuse, pixel_coord, 0);
    let base_indirect_specular = textureLoad(ddgi_indirect_specular, pixel_coord, 0);

    let detail_direct = textureLoad(short_range_direct, pixel_coord, 0);
    let detail_indirect_diffuse = textureLoad(short_range_indirect_diffuse, pixel_coord, 0);
    let detail_indirect_specular = textureLoad(short_range_indirect_specular, pixel_coord, 0);

    let exposure = 1.2;
    let tonemapped_detail_direct = reinhard_tonemapping(detail_direct.rgb, exposure);
    let tonemapped_base_direct = reinhard_tonemapping(base_direct.rgb, exposure);
    let tonemapped_detail_indirect_diffuse = reinhard_tonemapping(detail_indirect_diffuse.rgb, exposure);
    let tonemapped_base_indirect_diffuse = reinhard_tonemapping(base_indirect_diffuse.rgb, exposure);
    let tonemapped_detail_indirect_specular = reinhard_tonemapping(detail_indirect_specular.rgb, exposure);
    let tonemapped_base_indirect_specular = reinhard_tonemapping(base_indirect_specular.rgb, exposure);

    textureStore(
        out_direct,
        pixel_coord,
        vec4f(combine_short_range_modulation(tonemapped_base_direct, tonemapped_detail_direct), 1.0)
    );
    textureStore(
        out_indirect_diffuse,
        pixel_coord,
        vec4f(combine_short_range_modulation(tonemapped_base_indirect_diffuse, tonemapped_detail_indirect_diffuse), 1.0)
    );
    textureStore(
        out_indirect_specular,
        pixel_coord,
        vec4f(combine_short_range_modulation(tonemapped_base_indirect_specular, tonemapped_detail_indirect_specular), 1.0)
    );
}
