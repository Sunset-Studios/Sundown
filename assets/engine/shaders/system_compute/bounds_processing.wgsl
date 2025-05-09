#include "common.wgsl"

// ------------------------------------------------------------------------------------
// Constants
// ------------------------------------------------------------------------------------ 

const bounds_padding = 1.0;

// ------------------------------------------------------------------------------------
// Buffers
// ------------------------------------------------------------------------------------ 

@group(1) @binding(0) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(1) var<storage, read_write> entity_flags: array<i32>;
@group(1) @binding(2) var<storage, read> entity_dirty: array<u32>;
@group(1) @binding(3) var<storage, read> aabb_tree_nodes: array<AABBTreeNode>;
@group(1) @binding(4) var<storage, read_write> aabb_bounds: array<AABBNodeBounds>;
@group(1) @binding(5) var<storage, read> entity_aabb_node_indices: array<u32>;

// ------------------------------------------------------------------------------------
// Compute Shader
// ------------------------------------------------------------------------------------ 

@compute @workgroup_size(256)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= arrayLength(&aabb_tree_nodes)) {
        return;
    }

    let entity_id_offset = global_id.x;
    let node_index = entity_aabb_node_indices[entity_id_offset];
    let node = aabb_tree_nodes[node_index];
    let ent_flags = entity_flags[entity_id_offset];

    if (entity_dirty[entity_id_offset] == 0) {
        return;
    }

    var flags = i32(node.flags_and_node_data.x);
    var node_type = i32(node.flags_and_node_data.y);
    if ((flags & AABB_NODE_FLAGS_FREE) == AABB_NODE_FLAGS_FREE || node_type != AABB_NODE_TYPE_LEAF) {
        return;
    }

    // Get the entity's world transform
    let transform = entity_transforms[entity_id_offset].transform;
    let position = transform[3].xyz;
    let scale = vec3f(length(transform[0].xyz), length(transform[1].xyz), length(transform[2].xyz));

    // Calculate bounds based on position and scale
    // This is a simple axis-aligned box, but could be more sophisticated
    // based on the entity's mesh or collider
    let half_size = vec3f(
      abs(scale[0]) * 0.5,
      abs(scale[1]) * 0.5,
      abs(scale[2]) * 0.5,
    );

    // Add padding
    let padding = vec3f(
      half_size[0] * bounds_padding,
      half_size[1] * bounds_padding,
      half_size[2] * bounds_padding,
    );

    let min_point = vec3f(
      position[0] - half_size[0] - padding[0],
      position[1] - half_size[1] - padding[1],
      position[2] - half_size[2] - padding[2],
    );

    let max_point = vec3f(
      position[0] + half_size[0] + padding[0],
      position[1] + half_size[1] + padding[1],
      position[2] + half_size[2] + padding[2],
    );

    aabb_bounds[node_index].min_point = vec4f(min_point, 1.0);
    aabb_bounds[node_index].max_point = vec4f(max_point, 1.0);

    entity_flags[entity_id_offset] = ent_flags | ETF_AABB_DIRTY;
}