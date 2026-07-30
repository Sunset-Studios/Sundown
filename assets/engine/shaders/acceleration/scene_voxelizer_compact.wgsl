#include "common.wgsl"
#include "visibility/visibility_common.wgsl"
#include "acceleration/scene_voxelizer_common.wgsl"

@group(1) @binding(0) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(1) var<storage, read> object_instances: array<ObjectInstance>;
@group(1) @binding(2) var<storage, read> meshlet_instances: array<MeshletInstance>;
@group(1) @binding(3) var<storage, read> entity_index_lookup: array<u32>;
@group(1) @binding(4) var<storage, read> meshlets: array<MeshletRecord>;
@group(1) @binding(5) var<uniform> params: SceneVoxelizationParams;
@group(1) @binding(6) var<storage, read_write> dirty_brick_words: array<atomic<u32>>;
@group(1) @binding(7) var<storage, read_write> compacted_meshlets: array<MeshletInstance>;
@group(1) @binding(8) var<storage, read_write> voxel_dispatch_count: atomic<u32>;

fn range_contains_dirty_brick(range: SceneVoxelBrickRange) -> bool {
    let width = range.max.x - range.min.x + 1u;
    var x_mask = 0xffffffffu;
    if (width < 32u) {
        x_mask = ((1u << width) - 1u) << range.min.x;
    }

    for (var z = range.min.z; z <= range.max.z; z = z + 1u) {
        for (var y = range.min.y; y <= range.max.y; y = y + 1u) {
            // Brick x is the low five bits of the linear index, so an entire x
            // interval maps to one word and one masked load for each yz row.
            let word = y + SCENE_VOXEL_BRICK_DIMENSION * z;
            if ((atomicLoad(&dirty_brick_words[word]) & x_mask) != 0u) {
                return true;
            }
        }
    }
    return false;
}

fn select_dirty_meshlet(
    item_index: u32,
    selected_instance: ptr<function, MeshletInstance>
) -> bool {
    if (item_index >= params.meshlet_count || item_index >= arrayLength(&meshlet_instances)) {
        return false;
    }

    let instance = meshlet_instances[item_index];
    if (
        instance.object_instance_index >= arrayLength(&object_instances) ||
        instance.meshlet_index >= arrayLength(&meshlets)
    ) {
        return false;
    }

    let entity_row = get_entity_row(object_instances[instance.object_instance_index].row);
    if (entity_row >= arrayLength(&entity_index_lookup)) {
        return false;
    }
    let entity_index = entity_index_lookup[entity_row];
    if (
        entity_index == INVALID_IDX ||
        entity_index >= arrayLength(&entity_transforms)
    ) {
        return false;
    }

    let bounds = scene_voxelizer_transform_bounds(
        meshlets[instance.meshlet_index],
        entity_transforms[entity_index].transform
    );
    if (
        !scene_voxelizer_bounds_intersect_volume(bounds, params) ||
        !range_contains_dirty_brick(scene_voxelizer_brick_range(bounds, params))
    ) {
        return false;
    }

    *selected_instance = instance;
    return true;
}

@compute @workgroup_size(128)
fn cs(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(subgroup_invocation_id) subgroup_lane: u32
) {
    var selected_instance: MeshletInstance;
    let selected = select_dirty_meshlet(global_id.x, &selected_instance);
    let selected_count = select(0u, 1u, selected);
    let subgroup_offset = subgroupExclusiveAdd(selected_count);
    let subgroup_count = subgroupAdd(selected_count);

    var subgroup_base = 0u;
    if (subgroup_lane == 0u && subgroup_count != 0u) {
        subgroup_base = atomicAdd(&voxel_dispatch_count, subgroup_count);
    }
    subgroup_base = subgroupBroadcastFirst(subgroup_base);

    if (selected) {
        let slot = subgroup_base + subgroup_offset;
        if (slot < arrayLength(&compacted_meshlets)) {
            compacted_meshlets[slot] = selected_instance;
        }
    }
}
