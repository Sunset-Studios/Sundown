#include "common.wgsl"
#include "acceleration_common.wgsl"

const bounds_padding = 0.01;
const CHUNK_CAPACITY = 256u;

@group(1) @binding(0) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(1) var<storage, read_write> entity_flags: array<u32>;
@group(1) @binding(2) var<storage, read> entity_index_lookup: array<u32>;
@group(1) @binding(3) var<storage, read_write> aabb_bounds: array<AABB>;
@group(1) @binding(4) var<storage, read_write> scene_aabb: AABB;
@group(1) @binding(5) var<storage, read> entity_mesh_ids: array<u32>;
@group(1) @binding(6) var<storage, read> mesh_local_bounds: array<AABB>;
@group(1) @binding(7) var<storage, read> active_chunk_indices: array<u32>;

@compute @workgroup_size(256)
fn cs(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let active_chunk_idx = workgroup_id.x;
    if (active_chunk_idx >= arrayLength(&active_chunk_indices)) {
        return;
    }

    let chunk_index = active_chunk_indices[active_chunk_idx];
    let dense_row = active_chunk_idx * CHUNK_CAPACITY + local_id.x;
    let entity_row = chunk_index * CHUNK_CAPACITY + local_id.x;

    var min_node_bounds = vec4<f32>(0.0, 0.0, 0.0, -1.0);
    var max_node_bounds = vec4<f32>(0.0, 0.0, 0.0, -1.0);

    if (entity_row < arrayLength(&entity_index_lookup)) {
        let entity_resolved = entity_index_lookup[entity_row];
        let entity_valid = entity_resolved != INVALID_IDX && entity_resolved < arrayLength(&entity_transforms);

        if (entity_valid) {
            let transform = entity_transforms[entity_resolved].transform;
            let flags = entity_flags[entity_resolved];
            let has_mesh = (flags & EF_HAS_MESH) != 0u;
            let mesh_id = select(INVALID_IDX, entity_mesh_ids[entity_resolved], has_mesh);
            let is_active = transform[3].w != 0.0 && has_mesh && mesh_id != INVALID_IDX;

            if (is_active) {
                let mesh_min_local = mesh_local_bounds[mesh_id].min.xyz;
                let mesh_max_local = mesh_local_bounds[mesh_id].max.xyz;
                let center_local = 0.5 * (mesh_min_local + mesh_max_local);
                let half_local = 0.5 * (mesh_max_local - mesh_min_local);

                let world_center = (transform * vec4<f32>(center_local, 1.0)).xyz;
                let c0 = abs(transform[0].xyz);
                let c1 = abs(transform[1].xyz);
                let c2 = abs(transform[2].xyz);
                let world_half = vec3<f32>(
                    c0.x * half_local.x + c1.x * half_local.y + c2.x * half_local.z,
                    c0.y * half_local.x + c1.y * half_local.y + c2.y * half_local.z,
                    c0.z * half_local.x + c1.z * half_local.y + c2.z * half_local.z,
                );
                let padding = world_half * bounds_padding;
                let min_point = world_center - (world_half + padding);
                let max_point = world_center + (world_half + padding);

                min_node_bounds = vec4<f32>(min_point, f32(mesh_id));
                max_node_bounds = vec4<f32>(max_point, -1.0 - f32(entity_row));

                entity_flags[entity_resolved] |= EF_AABB_DIRTY;
            }
        }
    }

    if (dense_row < arrayLength(&aabb_bounds)) {
        aabb_bounds[dense_row].min = min_node_bounds;
        aabb_bounds[dense_row].max = max_node_bounds;
    }
}
