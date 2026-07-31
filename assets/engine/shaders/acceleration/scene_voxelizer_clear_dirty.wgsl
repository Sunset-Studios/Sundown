#include "common.wgsl"
#include "visibility/visibility_common.wgsl"
#include "acceleration/scene_voxelizer_common.wgsl"

#define SCENE_VOXEL_CLIP_LEVEL 0u

#define SCENE_VOXEL_CLIP_LEVEL 0u

@group(1) @binding(0) var<storage, read> dirty_brick_list: array<u32>;
@group(1) @binding(1) var<storage, read_write> voxel_grid: array<atomic<u32>>;

// One lane clears one x-aligned eight-voxel row. A brick therefore costs one
// 64-lane workgroup and never touches neighboring bricks sharing the same word.
@compute @workgroup_size(64)
fn cs(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) lane_index: u32
) {
    let dirty_index = workgroup_id.x;
    if (dirty_index >= arrayLength(&dirty_brick_list)) {
        return;
    }

    let brick_index = dirty_brick_list[dirty_index];
    let brick_x = brick_index % SCENE_VOXEL_BRICK_DIMENSION;
    let brick_y =
        (brick_index / SCENE_VOXEL_BRICK_DIMENSION) % SCENE_VOXEL_BRICK_DIMENSION;
    let brick_z =
        brick_index / (SCENE_VOXEL_BRICK_DIMENSION * SCENE_VOXEL_BRICK_DIMENSION);
    let local_y = lane_index & 7u;
    let local_z = lane_index >> 3u;
    let x = brick_x * SCENE_VOXEL_BRICK_SIZE;
    let y = brick_y * SCENE_VOXEL_BRICK_SIZE + local_y;
    let z = brick_z * SCENE_VOXEL_BRICK_SIZE + local_z;
    let linear_index = x + 256u * (y + 256u * z);
    let word_index =
        scene_voxel_clipmap_leaf_word_offset(SCENE_VOXEL_CLIP_LEVEL) +
        (linear_index >> 5u);
    let bit_offset = linear_index & 31u;
    atomicAnd(&voxel_grid[word_index], ~(0xffu << bit_offset));
}
