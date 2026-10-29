#include "common.wgsl"
#include "acceleration_common.wgsl"

@group(1) @binding(0) var<uniform> tlas_bvh_info: BVHInfo;
@group(1) @binding(1) var<storage, read> entity_index_lookup: array<u32>;
@group(1) @binding(2) var<storage, read> material_table_offset: array<u32>;
@group(1) @binding(3) var<storage, read_write> emissive_light_instances: array<vec2<u32>>;

@compute @workgroup_size(128)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let prim_store = global_id.x;
    if (prim_store >= tlas_bvh_info.prim_count) {
        return;
    }

    let entity_row = entity_index_lookup[prim_store];
    // Unoccupied ECS slots have no material offset and cannot appear as TLAS leaves.
    if (entity_row == INVALID_IDX) {
        emissive_light_instances[prim_store] = vec2<u32>(INVALID_IDX, 0u);
        return;
    }
    emissive_light_instances[prim_store] = vec2<u32>(entity_row, material_table_offset[entity_row]);
}
