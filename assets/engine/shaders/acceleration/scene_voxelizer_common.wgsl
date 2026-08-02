#include "acceleration_common.wgsl"
#include "acceleration/scene_voxel_hierarchy_common.wgsl"

struct SceneVoxelDispatchArgs {
    workgroup_count_x: atomic<u32>,
    workgroup_count_y: atomic<u32>,
    workgroup_count_z: atomic<u32>,
    item_count: atomic<u32>,
    compact_workgroup_count_x: atomic<u32>,
    compact_workgroup_count_y: atomic<u32>,
    compact_workgroup_count_z: atomic<u32>,
};

struct SceneVoxelBrickRange {
    min: vec3<u32>,
    max: vec3<u32>,
};

const SCENE_VOXELIZER_FLAG_FULL_REBUILD: u32 = 1u;
const SCENE_VOXEL_BRICK_SIZE: u32 = 8u;
const SCENE_VOXEL_BRICK_DIMENSION: u32 = 32u;
const SCENE_VOXEL_BRICK_COUNT: u32 = 32768u;

fn scene_voxelizer_transform_bounds(
    meshlet: MeshletRecord,
    transform: mat4x4<f32>
) -> AABB {
    let local_center = (meshlet.bounds_min.xyz + meshlet.bounds_max.xyz) * 0.5;
    let local_extent = (meshlet.bounds_max.xyz - meshlet.bounds_min.xyz) * 0.5;
    let world_center = (transform * vec4<f32>(local_center, 1.0)).xyz;
    let world_extent =
        abs(transform[0].xyz) * local_extent.x +
        abs(transform[1].xyz) * local_extent.y +
        abs(transform[2].xyz) * local_extent.z;

    var bounds: AABB;
    bounds.min = vec4<f32>(world_center - world_extent, 0.0);
    bounds.max = vec4<f32>(world_center + world_extent, 0.0);
    return bounds;
}

fn scene_voxelizer_bounds_intersect_volume(
    bounds: AABB,
    voxelization_params: SceneVoxelizationParams
) -> bool {
    let volume_max =
        voxelization_params.grid_origin +
        vec3<f32>(f32(voxelization_params.resolution) * voxelization_params.voxel_size);
    return all(bounds.max.xyz >= voxelization_params.grid_origin) &&
        all(bounds.min.xyz < volume_max);
}

fn scene_voxelizer_brick_range(
    bounds: AABB,
    voxelization_params: SceneVoxelizationParams
) -> SceneVoxelBrickRange {
    let max_grid_coord = f32(voxelization_params.resolution) - 0.0001;
    let grid_min = clamp(
        (bounds.min.xyz - voxelization_params.grid_origin) / voxelization_params.voxel_size,
        vec3<f32>(0.0),
        vec3<f32>(max_grid_coord)
    );
    let grid_max = clamp(
        (bounds.max.xyz - voxelization_params.grid_origin) / voxelization_params.voxel_size,
        vec3<f32>(0.0),
        vec3<f32>(max_grid_coord)
    );

    var range: SceneVoxelBrickRange;
    range.min = vec3<u32>(floor(grid_min / f32(SCENE_VOXEL_BRICK_SIZE)));
    range.max = vec3<u32>(floor(grid_max / f32(SCENE_VOXEL_BRICK_SIZE)));
    return range;
}

fn scene_voxelizer_brick_index(coord: vec3<u32>) -> u32 {
    return coord.x +
        SCENE_VOXEL_BRICK_DIMENSION *
        (coord.y + SCENE_VOXEL_BRICK_DIMENSION * coord.z);
}

// Maps a dense index onto the non-overlapping union of newly exposed x, y, and z
// slabs. Excluding earlier-axis slabs keeps diagonal scrolling duplicate-free.
fn scene_voxelizer_scroll_dirty_brick(
    item_index: u32,
    voxelization_params: SceneVoxelizationParams
) -> vec3<u32> {
    let delta = voxelization_params.scroll_delta_bricks;
    let delta_abs = vec3<u32>(abs(delta));
    let retained_extent = vec3<u32>(SCENE_VOXEL_BRICK_DIMENSION) - delta_abs;
    let retained_min = select(vec3<u32>(0u), delta_abs, delta < vec3<i32>(0));
    let exposed_min = select(
        vec3<u32>(0u),
        vec3<u32>(SCENE_VOXEL_BRICK_DIMENSION) - delta_abs,
        delta > vec3<i32>(0)
    );

    let x_slab_count =
        delta_abs.x * SCENE_VOXEL_BRICK_DIMENSION * SCENE_VOXEL_BRICK_DIMENSION;
    if (item_index < x_slab_count) {
        let yz_index = item_index / delta_abs.x;
        return vec3<u32>(
            exposed_min.x + item_index % delta_abs.x,
            yz_index % SCENE_VOXEL_BRICK_DIMENSION,
            yz_index / SCENE_VOXEL_BRICK_DIMENSION
        );
    }

    var slab_index = item_index - x_slab_count;
    let y_slab_count =
        retained_extent.x * delta_abs.y * SCENE_VOXEL_BRICK_DIMENSION;
    if (slab_index < y_slab_count) {
        let yz_index = slab_index / retained_extent.x;
        return vec3<u32>(
            retained_min.x + slab_index % retained_extent.x,
            exposed_min.y + yz_index % delta_abs.y,
            yz_index / delta_abs.y
        );
    }

    slab_index = slab_index - y_slab_count;
    let xy_extent = retained_extent.x * retained_extent.y;
    return vec3<u32>(
        retained_min.x + slab_index % retained_extent.x,
        retained_min.y + (slab_index / retained_extent.x) % retained_extent.y,
        exposed_min.z + slab_index / xy_extent
    );
}
