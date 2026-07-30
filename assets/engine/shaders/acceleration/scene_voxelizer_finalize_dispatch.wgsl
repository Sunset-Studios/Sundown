#include "common.wgsl"
#include "visibility/visibility_common.wgsl"
#include "acceleration/scene_voxelizer_common.wgsl"

struct SceneVoxelIndirectArgs {
    workgroup_count_x: atomic<u32>,
    workgroup_count_y: atomic<u32>,
    workgroup_count_z: atomic<u32>,
};

@group(1) @binding(0) var<storage, read_write> voxel_dispatch_count: atomic<u32>;
@group(1) @binding(1) var<storage, read_write> voxel_dispatch: SceneVoxelIndirectArgs;

@compute @workgroup_size(1)
fn cs() {
    let item_count = atomicLoad(&voxel_dispatch_count);
    var dispatch_width = 0u;
    var dispatch_height = 0u;
    if (item_count != 0u) {
        dispatch_width = min(item_count, 65535u);
        dispatch_height = (item_count + dispatch_width - 1u) / dispatch_width;
    }
    atomicStore(&voxel_dispatch.workgroup_count_x, dispatch_width);
    atomicStore(&voxel_dispatch.workgroup_count_y, dispatch_height);
    atomicStore(&voxel_dispatch.workgroup_count_z, 1u);
}
