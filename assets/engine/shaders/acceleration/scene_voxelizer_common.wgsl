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
