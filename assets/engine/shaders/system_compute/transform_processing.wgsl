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
@group(1) @binding(3) var<storage, read_write> entity_transforms: array<EntityTransform>;
@group(1) @binding(4) var<storage, read_write> entity_flags: array<u32>;
@group(1) @binding(5) var<storage, read> scene_graph: array<vec2<i32>>;
@group(1) @binding(6) var<uniform> scene_graph_layer_data: SceneGraphLayerData;

// ------------------------------------------------------------------------------------
// Compute Shader
// ------------------------------------------------------------------------------------ 

@compute @workgroup_size(256)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= scene_graph_layer_data.count) {
        return;
    }

    let entity_id_offset = scene_graph_layer_data.offset + global_id.x;

    let entity_resolved = select(
        MAX_UINT,
        get_entity_row(u32(scene_graph[entity_id_offset].x)),
        scene_graph[entity_id_offset].x != -1
    );
    let parent_resolved = select(
        MAX_UINT,
        get_entity_row(u32(scene_graph[entity_id_offset].y)),
        scene_graph[entity_id_offset].y != -1
    );

    if (entity_resolved >= arrayLength(&entity_flags)) {
        return;
    }

    if ((entity_flags[entity_resolved] & EF_DIRTY) == 0 &&
        (entity_flags[parent_resolved] & EF_DIRTY) == 0) {
        return;
    }

    let position = entity_positions[entity_resolved];
    let rotation = entity_rotations[entity_resolved];
    let scale = entity_scales[entity_resolved];
    let flag = entity_flags[entity_resolved];

    var parent_transform = identity_matrix;

    if (parent_resolved < MAX_UINT) {
        parent_transform = entity_transforms[parent_resolved].transform;

        if ((flag & EF_IGNORE_PARENT_ROTATION) != 0) {
            // Extract translation from parent transform
            let parent_translation = vec3f(parent_transform[3].xyz);
            let parent_scale = vec3f(parent_transform[0].x, parent_transform[1].y, parent_transform[2].z);
            // Create a new parent transform that only has translation and scale
            parent_transform = mat4x4f(
                parent_scale.x, 0.0, 0.0, 0.0,
                0.0, parent_scale.y, 0.0, 0.0,
                0.0, 0.0, parent_scale.z, 0.0,
                parent_translation.x, parent_translation.y, parent_translation.z, 1.0
            );
        }
        if ((flag & EF_IGNORE_PARENT_SCALE) != 0) {
            // Create a new parent transform that only has translation and rotation
            parent_transform[0] = parent_transform[0] / max(length(vec3f(parent_transform[0].xyz)), 1e-6);
            parent_transform[1] = parent_transform[1] / max(length(vec3f(parent_transform[1].xyz)), 1e-6);
            parent_transform[2] = parent_transform[2] / max(length(vec3f(parent_transform[2].xyz)), 1e-6);
        }
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

    entity_flags[entity_resolved] |= entity_flags[parent_resolved] & EF_DIRTY;
}