#include "common.wgsl"

// ------------------------------------------------------------------------------------
// Constants
// ------------------------------------------------------------------------------------ 

// For things like world space UI, this prevents z-fighting between parent-child elements that are positioned at the same z-depth
const layer_z_offset_amount = 0.05;

// ------------------------------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------------------------------ 

struct SceneGraphLayerData {
    count: u32,
    offset: u32,
    layer_index: u32
};

// ------------------------------------------------------------------------------------
// Buffers
// ------------------------------------------------------------------------------------ 

@group(1) @binding(0) var<storage, read> entity_positions: array<vec4f>;
@group(1) @binding(1) var<storage, read> entity_rotations: array<vec4f>;
@group(1) @binding(2) var<storage, read> entity_scales: array<vec4f>;
@group(1) @binding(3) var<storage, read_write> entity_dirty_flags: array<u32>;
@group(1) @binding(4) var<storage, read_write> entity_transforms: array<EntityTransform>;
@group(1) @binding(5) var<storage, read_write> entity_bounds_data: array<EntityBoundsData>;
@group(1) @binding(6) var<storage, read> scene_graph: array<vec2<i32>>;
@group(1) @binding(7) var<uniform> scene_graph_layer_data: SceneGraphLayerData;

// ------------------------------------------------------------------------------------
// Compute Shader
// ------------------------------------------------------------------------------------ 

@compute @workgroup_size(256)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= scene_graph_layer_data.count) {
        return;
    }

    let entity_id_offset = scene_graph_layer_data.offset + global_id.x;
    let entity_resolved = u32(scene_graph[entity_id_offset].x);

    if (entity_resolved >= arrayLength(&entity_dirty_flags)) {
        return;
    }

    let dirty_flag = entity_dirty_flags[entity_resolved];
    if (dirty_flag == 0u) {
        return;
    }

    let position = entity_positions[entity_resolved];
    let rotation = entity_rotations[entity_resolved];
    let scale = entity_scales[entity_resolved];

    let parent_resolved = scene_graph[entity_id_offset].y;
    var parent_transform = identity_matrix;
    if (parent_resolved >= 0) {
        parent_transform = entity_transforms[parent_resolved].transform;
    }

    let max_scale = max(
        max(scale.x, scale.y),
        scale.z
    );

    // Calculate world transform matrix
    let transform = parent_transform * mat4x4f(
        (1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z)) * scale.x,
        (2.0 * (rotation.x * rotation.y + rotation.w * rotation.z)) * scale.x,
        (2.0 * (rotation.x * rotation.z - rotation.w * rotation.y)) * scale.x,
        0.0,
        
        (2.0 * (rotation.x * rotation.y - rotation.w * rotation.z)) * scale.y,
        (1.0 - 2.0 * (rotation.x * rotation.x + rotation.z * rotation.z)) * scale.y,
        (2.0 * (rotation.y * rotation.z + rotation.w * rotation.x)) * scale.y,
        0.0,
        
        (2.0 * (rotation.x * rotation.z + rotation.w * rotation.y)) * scale.z,
        (2.0 * (rotation.y * rotation.z - rotation.w * rotation.x)) * scale.z,
        (1.0 - 2.0 * (rotation.x * rotation.x + rotation.y * rotation.y)) * scale.z,
        0.0,
        
        position.x,
        position.y,
        position.z + layer_z_offset_amount * f32(scene_graph_layer_data.layer_index),
        1.0
    );

    // Calculate inverse transform
    let det = transform[0][0] * (transform[1][1] * transform[2][2] - transform[2][1] * transform[1][2]) -
        transform[0][1] * (transform[1][0] * transform[2][2] - transform[1][2] * transform[2][0]) +
        transform[0][2] * (transform[1][0] * transform[2][1] - transform[1][1] * transform[2][0]);

    let inv_det = 1.0 / det;

    let inverse_transform = mat4x4f(
        (transform[1][1] * transform[2][2] - transform[2][1] * transform[1][2]) * inv_det,
        -(transform[0][1] * transform[2][2] - transform[0][2] * transform[2][1]) * inv_det,
        (transform[0][1] * transform[1][2] - transform[0][2] * transform[1][1]) * inv_det,
        0.0,

        -(transform[1][0] * transform[2][2] - transform[1][2] * transform[2][0]) * inv_det,
        (transform[0][0] * transform[2][2] - transform[0][2] * transform[2][0]) * inv_det,
        -(transform[0][0] * transform[1][2] - transform[0][2] * transform[1][0]) * inv_det,
        0.0,

        (transform[1][0] * transform[2][1] - transform[2][0] * transform[1][1]) * inv_det,
        -(transform[0][0] * transform[2][1] - transform[2][0] * transform[0][1]) * inv_det,
        (transform[0][0] * transform[1][1] - transform[1][0] * transform[0][1]) * inv_det,
        0.0,

        -(transform[1][0] * (transform[2][1] * transform[3][2] - transform[2][2] * transform[3][1]) -
           transform[1][1] * (transform[2][0] * transform[3][2] - transform[2][2] * transform[3][0]) +
           transform[1][2] * (transform[2][0] * transform[3][1] - transform[2][1] * transform[3][0])) * inv_det,
        (transform[0][0] * (transform[2][1] * transform[3][2] - transform[2][2] * transform[3][1]) -
           transform[0][1] * (transform[2][0] * transform[3][2] - transform[2][2] * transform[3][0]) +
           transform[0][2] * (transform[2][0] * transform[3][1] - transform[2][1] * transform[3][0])) * inv_det,
        -(transform[0][0] * (transform[1][1] * transform[3][2] - transform[1][2] * transform[3][1]) -
           transform[0][1] * (transform[1][0] * transform[3][2] - transform[1][2] * transform[3][0]) +
           transform[0][2] * (transform[1][0] * transform[3][1] - transform[1][1] * transform[3][0])) * inv_det,
        1.0
    );

    entity_transforms[entity_resolved].transform = transform;

    entity_transforms[entity_resolved].transpose_inverse_model_matrix = mat4x4f(
        inverse_transform[0][0], inverse_transform[1][0], inverse_transform[2][0], inverse_transform[3][0],
        inverse_transform[0][1], inverse_transform[1][1], inverse_transform[2][1], inverse_transform[3][1],
        inverse_transform[0][2], inverse_transform[1][2], inverse_transform[2][2], inverse_transform[3][2],
        inverse_transform[0][3], inverse_transform[1][3], inverse_transform[2][3], inverse_transform[3][3]
    );

    entity_bounds_data[entity_resolved].bounds_pos_radius = vec4f(transform[3][0], transform[3][1], transform[3][2], max_scale);

    entity_bounds_data[entity_resolved].bounds_extent_and_custom_scale = vec4f(1.0, 1.0, 1.0, 1.0);

    entity_dirty_flags[entity_resolved] = 0u;
}