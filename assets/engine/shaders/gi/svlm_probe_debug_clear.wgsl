@group(1) @binding(0) var<storage, read_write> debug_depth: array<atomic<u32>>;
@group(1) @binding(1) var depth_texture: texture_2d<f32>;
@group(1) @binding(2) var<storage, read_write> debug_leaf_indices: array<atomic<u32>>;

// Initializes the per-pixel probe debug depth buffer. Probe splatting uses
// atomicMin on a packed depth/shade value, so all pixels start at the far
// sentinel and the closest probe fragment wins.
@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(depth_texture);
    if (gid.x >= res.x || gid.y >= res.y) {
        return;
    }

    let pixel_index = gid.y * res.x + gid.x;
    atomicStore(&debug_depth[pixel_index], 0xffffffffu);

    if (pixel_index < arrayLength(&debug_leaf_indices)) {
        atomicStore(&debug_leaf_indices[pixel_index], 0xffffffffu);
    }

    if (gid.x == 0u && gid.y == 0u) {
        atomicStore(&debug_leaf_indices[0], 0u);
    }
}
