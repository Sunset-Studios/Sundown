#include "acceleration/scene_voxel_hierarchy_common.wgsl"

// Reduces each 8^3 leaf brick to one level-one occupancy bit. Because x is the
// tightly packed axis and brick boundaries are eight-voxel aligned, each yz row
// needs one masked word load instead of eight individual voxel loads.

@group(1) @binding(0) var<storage, read> scene_voxel_grid: array<u32>;
@group(1) @binding(1) var<storage, read_write> scene_voxel_hierarchy: array<atomic<u32>>;

const SCENE_VOXEL_BUILD_BRICK_WORKGROUP_SIZE: u32 = 128u;

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let brick_index = gid.x;
    let clip_level = gid.z;
    let brick_count =
        SCENE_VOXEL_HDDA_LEVEL_1_RESOLUTION *
        SCENE_VOXEL_HDDA_LEVEL_1_RESOLUTION *
        SCENE_VOXEL_HDDA_LEVEL_1_RESOLUTION;
    if (brick_index >= brick_count || clip_level >= SCENE_VOXEL_CLIPMAP_MAX_LEVEL_COUNT) {
        return;
    }

    let brick_resolution = SCENE_VOXEL_HDDA_LEVEL_1_RESOLUTION;
    let brick_coord = vec3<u32>(
        brick_index % brick_resolution,
        (brick_index / brick_resolution) % brick_resolution,
        brick_index / (brick_resolution * brick_resolution)
    );
    let leaf_min = brick_coord * 8u;
    let x_mask = 0xffu << (leaf_min.x & 31u);

    var occupied = false;
    for (var local_z = 0u; local_z < 8u && !occupied; local_z = local_z + 1u) {
        let z = leaf_min.z + local_z;
        for (var local_y = 0u; local_y < 8u; local_y = local_y + 1u) {
            let y = leaf_min.y + local_y;
            let leaf_index = leaf_min.x + 256u * (y + 256u * z);
            let leaf_word_index =
                scene_voxel_clipmap_leaf_word_offset(clip_level) +
                (leaf_index >> 5u);
            if ((scene_voxel_grid[leaf_word_index] & x_mask) != 0u) {
                occupied = true;
                break;
            }
        }
    }

    if (occupied) {
        let word_index = scene_voxel_clipmap_hierarchy_word_offset(clip_level) +
            SCENE_VOXEL_HDDA_LEVEL_1_WORD_OFFSET + (brick_index >> 5u);
        atomicOr(&scene_voxel_hierarchy[word_index], 1u << (brick_index & 31u));
    }
}
