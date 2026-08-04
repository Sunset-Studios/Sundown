@group(1) @binding(0) var<storage, read_write> debug_depth: array<atomic<u32>>;
@group(1) @binding(1) var depth_texture: texture_2d<f32>;

// The baked-probe path uploads a compact resident-leaf list on the CPU, so it
// only clears the per-pixel depth/color target before sharing the probe splat.
@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(depth_texture);
    if (gid.x >= res.x || gid.y >= res.y) {
        return;
    }

    let pixel_index = gid.y * res.x + gid.x;
    atomicStore(&debug_depth[pixel_index], 0xffffffffu);
}
