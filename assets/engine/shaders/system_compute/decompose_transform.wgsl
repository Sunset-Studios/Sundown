#include "common.wgsl"

// ------------------------------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------------------------------ 

struct SceneGraphLayerData {
    count: u32,
    offset: u32,
    layer_index: u32 // May not be strictly needed here but kept for consistency
};

// ------------------------------------------------------------------------------------
// Buffers
// ------------------------------------------------------------------------------------ 

@group(1) @binding(0) var<storage, read> entity_transforms: array<EntityTransform>; // Input: World transforms from the first pass
@group(1) @binding(1) var<storage, read> entity_flags: array<u32>; // Input: To know which entities are active/dirty
@group(1) @binding(2) var<storage, read> scene_graph: array<vec2<i32>>; // Input: For consistent indexing with the first pass
@group(1) @binding(3) var<uniform> scene_graph_layer_data: SceneGraphLayerData; // Input: For consistent indexing
@group(1) @binding(4) var<storage, read_write> out_world_positions: array<vec4f>;
@group(1) @binding(5) var<storage, read_write> out_world_rotations: array<vec4f>; // Store as quaternion
@group(1) @binding(6) var<storage, read_write> out_world_scales: array<vec4f>; 

// ------------------------------------------------------------------------------------
// Compute Shader
// ------------------------------------------------------------------------------------ 

@compute @workgroup_size(256)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= scene_graph_layer_data.count) {
        return;
    }

    let entity_id_offset = scene_graph_layer_data.offset + global_id.x;

    // We use scene_graph[entity_id_offset].x to get the original entity ID
    // and then resolve it to its row index in the packed ECS data.
    let entity_resolved = select(
        MAX_UINT,
        get_entity_row(u32(scene_graph[entity_id_offset].x)),
        scene_graph[entity_id_offset].x != -1
    );

    if (entity_resolved >= arrayLength(&entity_flags) || entity_resolved >= arrayLength(&entity_transforms)) {
        return;
    }

    // Process only if the entity was marked dirty by the first pass (or initially)
    // or if we need to ensure all entities in the scene graph layer are processed.
    // For simplicity, we can rely on the first pass to have updated the transform if needed.
    // If entity_flags[entity_resolved] & EF_DIRTY == 0, it means the transform wasn't updated,
    // but its parent might have, causing a cascading update. The transform matrix should be correct.

    let transform = entity_transforms[entity_resolved].transform;

    // Decompose and write world components
    // World Position
    out_world_positions[entity_resolved] = vec4f(transform[3].xyz, 1.0);

    // World Scale
    let scale_x = length(transform[0].xyz);
    let scale_y = length(transform[1].xyz);
    let scale_z = length(transform[2].xyz);
    out_world_scales[entity_resolved] = vec4f(scale_x, scale_y, scale_z, 1.0);

    // World Rotation (as quaternion)
    var rot_mat_no_scale: mat3x3f;
    if (scale_x > 1e-6 && scale_y > 1e-6 && scale_z > 1e-6) {
        rot_mat_no_scale = mat3x3f(
            transform[0].xyz / scale_x,
            transform[1].xyz / scale_y,
            transform[2].xyz / scale_z
        );
    } else {
        rot_mat_no_scale = mat3x3f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0); // Identity
    }

    var qx: f32;
    var qy: f32;
    var qz: f32;
    var qw: f32;
    let trace = rot_mat_no_scale[0][0] + rot_mat_no_scale[1][1] + rot_mat_no_scale[2][2];
    if (trace > 0.0) {
        var S = sqrt(trace + 1.0) * 2.0;
        qw = 0.25 * S;
        qx = (rot_mat_no_scale[2][1] - rot_mat_no_scale[1][2]) / S;
        qy = (rot_mat_no_scale[0][2] - rot_mat_no_scale[2][0]) / S;
        qz = (rot_mat_no_scale[1][0] - rot_mat_no_scale[0][1]) / S;
    } else if ((rot_mat_no_scale[0][0] > rot_mat_no_scale[1][1]) && (rot_mat_no_scale[0][0] > rot_mat_no_scale[2][2])) {
        var S = sqrt(1.0 + rot_mat_no_scale[0][0] - rot_mat_no_scale[1][1] - rot_mat_no_scale[2][2]) * 2.0;
        qw = (rot_mat_no_scale[2][1] - rot_mat_no_scale[1][2]) / S;
        qx = 0.25 * S;
        qy = (rot_mat_no_scale[0][1] + rot_mat_no_scale[1][0]) / S;
        qz = (rot_mat_no_scale[0][2] + rot_mat_no_scale[2][0]) / S;
    } else if (rot_mat_no_scale[1][1] > rot_mat_no_scale[2][2]) {
        var S = sqrt(1.0 + rot_mat_no_scale[1][1] - rot_mat_no_scale[0][0] - rot_mat_no_scale[2][2]) * 2.0;
        qw = (rot_mat_no_scale[0][2] - rot_mat_no_scale[2][0]) / S;
        qx = (rot_mat_no_scale[0][1] + rot_mat_no_scale[1][0]) / S;
        qy = 0.25 * S;
        qz = (rot_mat_no_scale[1][2] + rot_mat_no_scale[2][1]) / S;
    } else {
        var S = sqrt(1.0 + rot_mat_no_scale[2][2] - rot_mat_no_scale[0][0] - rot_mat_no_scale[1][1]) * 2.0;
        qw = (rot_mat_no_scale[1][0] - rot_mat_no_scale[0][1]) / S;
        qx = (rot_mat_no_scale[0][2] + rot_mat_no_scale[2][0]) / S;
        qy = (rot_mat_no_scale[1][2] + rot_mat_no_scale[2][1]) / S;
        qz = 0.25 * S;
    }
    out_world_rotations[entity_resolved] = vec4f(qx, qy, qz, qw);
} 