#include "common.wgsl"
#include "visibility/visibility_common.wgsl"
#include "acceleration/scene_voxelizer_common.wgsl"

@group(1) @binding(0) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(1) var<storage, read> entity_flags: array<u32>;
@group(1) @binding(2) var<storage, read> object_instances: array<ObjectInstance>;
@group(1) @binding(3) var<storage, read> meshlet_instances: array<MeshletInstance>;
@group(1) @binding(4) var<storage, read> entity_index_lookup: array<u32>;
@group(1) @binding(5) var<storage, read> meshlets: array<MeshletRecord>;
@group(1) @binding(6) var<uniform> params: SceneVoxelizationParams;
@group(1) @binding(7) var<storage, read_write> dirty_brick_words: array<atomic<u32>>;
@group(1) @binding(8) var<storage, read_write> dirty_brick_list: array<u32>;
@group(1) @binding(9) var<storage, read_write> dirty_dispatch: SceneVoxelDispatchArgs;

fn mark_dirty_brick(brick_index: u32) {
    let word = brick_index >> 5u;
    let mask = 1u << (brick_index & 31u);
    let previous = atomicOr(&dirty_brick_words[word], mask);
    if ((previous & mask) == 0u) {
        let slot = atomicAdd(&dirty_dispatch.item_count, 1u);
        if (slot < arrayLength(&dirty_brick_list)) {
            dirty_brick_list[slot] = brick_index;
            atomicAdd(&dirty_dispatch.workgroup_count_x, 1u);
            atomicMax(&dirty_dispatch.compact_workgroup_count_x, params.dispatch_width);
        }
    }
}

fn mark_dirty_range(range: SceneVoxelBrickRange) {
    for (var z = range.min.z; z <= range.max.z; z = z + 1u) {
        for (var y = range.min.y; y <= range.max.y; y = y + 1u) {
            for (var x = range.min.x; x <= range.max.x; x = x + 1u) {
                mark_dirty_brick(scene_voxelizer_brick_index(vec3<u32>(x, y, z)));
            }
        }
    }
}

@compute @workgroup_size(64)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let item_index = global_id.x;
    if ((params.flags & SCENE_VOXELIZER_FLAG_FULL_REBUILD) != 0u) {
        if (item_index < SCENE_VOXEL_BRICK_COUNT) {
            mark_dirty_brick(item_index);
        }
        return;
    }

    if (item_index >= params.meshlet_count || item_index >= arrayLength(&meshlet_instances)) {
        return;
    }

    let instance = meshlet_instances[item_index];
    if (
        instance.object_instance_index >= arrayLength(&object_instances) ||
        instance.meshlet_index >= arrayLength(&meshlets)
    ) {
        return;
    }

    let entity_row = get_entity_row(object_instances[instance.object_instance_index].row);
    if (entity_row >= arrayLength(&entity_index_lookup)) {
        return;
    }
    let entity_index = entity_index_lookup[entity_row];
    if (
        entity_index == INVALID_IDX ||
        entity_index >= arrayLength(&entity_transforms) ||
        entity_index >= arrayLength(&entity_flags) ||
        (entity_flags[entity_index] & EF_MOVED) == 0u
    ) {
        return;
    }

    let meshlet = meshlets[instance.meshlet_index];
    let transforms = entity_transforms[entity_index];
    let current_bounds = scene_voxelizer_transform_bounds(meshlet, transforms.transform);
    if (scene_voxelizer_bounds_intersect_volume(current_bounds, params)) {
        mark_dirty_range(scene_voxelizer_brick_range(current_bounds, params));
    }

    let previous_bounds = scene_voxelizer_transform_bounds(meshlet, transforms.prev_transform);
    if (scene_voxelizer_bounds_intersect_volume(previous_bounds, params)) {
        mark_dirty_range(scene_voxelizer_brick_range(previous_bounds, params));
    }
}
