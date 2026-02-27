#include "common.wgsl"

@group(1) @binding(0) var ddgi_direct: texture_2d<f32>;
@group(1) @binding(1) var ddgi_indirect_diffuse: texture_2d<f32>;
@group(1) @binding(2) var ddgi_indirect_specular: texture_2d<f32>;
@group(1) @binding(3) var short_range_direct: texture_2d<f32>;
@group(1) @binding(4) var short_range_indirect_diffuse: texture_2d<f32>;
@group(1) @binding(5) var short_range_indirect_specular: texture_2d<f32>;
@group(1) @binding(6) var out_direct: texture_storage_2d<rgba16float, write>;
@group(1) @binding(7) var out_indirect_diffuse: texture_storage_2d<rgba16float, write>;
@group(1) @binding(8) var out_indirect_specular: texture_storage_2d<rgba16float, write>;

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

    textureStore(
        out_direct,
        pixel_coord,
        base_direct * detail_direct
    );
    textureStore(
        out_indirect_diffuse,
        pixel_coord,
        base_indirect_diffuse * detail_indirect_diffuse
    );
    textureStore(
        out_indirect_specular,
        pixel_coord,
        base_indirect_specular * detail_indirect_specular
    );
}
