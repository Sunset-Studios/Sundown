#include "acceleration/scene_voxel_hierarchy_common.wgsl"

// One workgroup reduces all 64 level-two nodes and then their single root.
// Each level-two lane reads 64 x-aligned rows from the 32^3 level-one bitset.

@group(1) @binding(0) var<storage, read_write> scene_voxel_hierarchy: array<atomic<u32>>;

@compute @workgroup_size(64, 1, 1)
fn cs(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) lane_index: u32
) {
    let clip_level = workgroup_id.z;
    if (clip_level >= SCENE_VOXEL_CLIPMAP_MAX_LEVEL_COUNT) {
        return;
    }
    let hierarchy_offset = scene_voxel_clipmap_hierarchy_word_offset(clip_level);
    let coarse_resolution = SCENE_VOXEL_HDDA_LEVEL_2_RESOLUTION;
    let coarse_coord = vec3<u32>(
        lane_index % coarse_resolution,
        (lane_index / coarse_resolution) % coarse_resolution,
        lane_index / (coarse_resolution * coarse_resolution)
    );
    let brick_min = coarse_coord * 8u;
    let x_mask = 0xffu << (brick_min.x & 31u);

    var occupied = false;
    for (var local_z = 0u; local_z < 8u && !occupied; local_z = local_z + 1u) {
        let z = brick_min.z + local_z;
        for (var local_y = 0u; local_y < 8u; local_y = local_y + 1u) {
            let y = brick_min.y + local_y;
            let brick_index =
                brick_min.x +
                SCENE_VOXEL_HDDA_LEVEL_1_RESOLUTION *
                (y + SCENE_VOXEL_HDDA_LEVEL_1_RESOLUTION * z);
            let word_index = hierarchy_offset +
                SCENE_VOXEL_HDDA_LEVEL_1_WORD_OFFSET + (brick_index >> 5u);
            if ((atomicLoad(&scene_voxel_hierarchy[word_index]) & x_mask) != 0u) {
                occupied = true;
                break;
            }
        }
    }

    if (occupied) {
        let word_index = hierarchy_offset +
            SCENE_VOXEL_HDDA_LEVEL_2_WORD_OFFSET + (lane_index >> 5u);
        atomicOr(&scene_voxel_hierarchy[word_index], 1u << (lane_index & 31u));
    }

    workgroupBarrier();
    storageBarrier();

    if (lane_index == 0u) {
        let low_word = atomicLoad(
            &scene_voxel_hierarchy[hierarchy_offset + SCENE_VOXEL_HDDA_LEVEL_2_WORD_OFFSET]
        );
        let high_word = atomicLoad(
            &scene_voxel_hierarchy[hierarchy_offset + SCENE_VOXEL_HDDA_LEVEL_2_WORD_OFFSET + 1u]
        );
        if ((low_word | high_word) != 0u) {
            atomicStore(
                &scene_voxel_hierarchy[hierarchy_offset + SCENE_VOXEL_HDDA_LEVEL_3_WORD_OFFSET],
                1u
            );
        }
    }
}
