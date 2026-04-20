@group(1) @binding(0) var scene_color: texture_2d<f32>;
@group(1) @binding(1) var<storage, read> debug_depth: array<u32>;
@group(1) @binding(2) var output_debug: texture_storage_2d<rgba16float, write>;

fn svlm_probe_debug_level_color(color_index: u32) -> vec3<f32> {
    if (color_index == 0u) {
        return vec3<f32>(0.20, 0.95, 0.72);
    }
    if (color_index == 1u) {
        return vec3<f32>(0.38, 0.68, 1.00);
    }
    if (color_index == 2u) {
        return vec3<f32>(1.00, 0.77, 0.25);
    }
    if (color_index == 3u) {
        return vec3<f32>(1.00, 0.38, 0.46);
    }
    if (color_index == 4u) {
        return vec3<f32>(0.72, 0.54, 1.00);
    }
    return vec3<f32>(0.65, 1.00, 0.32);
}

// Composites the packed probe debug result over the lit scene. The splat pass
// stores closest-depth plus compact shade/color data, keeping the transient
// debug buffer compact and atomic-friendly.
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

    let shade = f32((packed_depth >> 3u) & 0x1fu) / 31.0;
    let color_index = packed_depth & 0x7u;
    let color = svlm_probe_debug_level_color(color_index) * shade;
    textureStore(output_debug, coord, vec4<f32>(color, 1.0));
}
