#include "common.wgsl"

@group(1) @binding(0) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(1) var<storage, read_write> ray_instance_transforms: array<RayInstanceTransform>;

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let entity_index = gid.x;
    if (entity_index >= arrayLength(&ray_instance_transforms) || entity_index >= arrayLength(&entity_transforms)) {
        return;
    }

    let entity_transform = entity_transforms[entity_index];
    var ray_instance_transform: RayInstanceTransform;

    ray_instance_transform.local_to_world0 = entity_transform.transform[0];
    ray_instance_transform.local_to_world1 = entity_transform.transform[1];
    ray_instance_transform.local_to_world2 = entity_transform.transform[2];
    ray_instance_transform.local_to_world3 = entity_transform.transform[3];

    ray_instance_transform.world_to_local0 = entity_transform.transpose_inverse_model_matrix[0];
    ray_instance_transform.world_to_local1 = entity_transform.transpose_inverse_model_matrix[1];
    ray_instance_transform.world_to_local2 = entity_transform.transpose_inverse_model_matrix[2];
    ray_instance_transform.world_to_local3 = entity_transform.transpose_inverse_model_matrix[3];

    ray_instance_transforms[entity_index] = ray_instance_transform;
}
