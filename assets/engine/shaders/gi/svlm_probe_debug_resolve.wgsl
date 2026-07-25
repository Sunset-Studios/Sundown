@group(1) @binding(0) var scene_color: texture_2d<f32>;
@group(1) @binding(1) var<storage, read> debug_depth: array<u32>;
@group(1) @binding(2) var output_debug: texture_storage_2d<rgba16float, write>;

fn svlm_probe_debug_unpack_rgb565(packed: u32) -> vec3<f32> {
    return vec3<f32>(
        f32((packed >> 11u) & 0x1fu) / 31.0,
        f32((packed >> 5u) & 0x3fu) / 63.0,
        f32(packed & 0x1fu) / 31.0
    );
}

// Composites the packed probe debug result over the lit scene. The splat pass
// stores closest-depth plus compact irradiance color data, keeping the
// transient debug buffer compact and atomic-friendly.
@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(output_debug);
    if (gid.x >= res.x || gid.y >= res.y) {
        return;
    }

    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let scene = textureLoad(scene_color, coord, 0);
    let pixel_index = gid.y * res.x + gid.x;
    let packed_depth = debug_depth[pixel_index];
    if (packed_depth == 0xffffffffu) {
        textureStore(output_debug, coord, scene);
        return;
    }

    let color = svlm_probe_debug_unpack_rgb565(packed_depth & 0xffffu);
    textureStore(output_debug, coord, vec4<f32>(color, 1.0));
}
