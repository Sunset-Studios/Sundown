#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "gi/surface_cache_common.wgsl"

@group(1) @binding(0) var<storage, read_write> counters: SurfaceCacheCounters;
@group(1) @binding(1) var<storage, read> tlas_bvh2_bounds: array<AABB>;
@group(1) @binding(2) var<uniform> tlas_bvh_info: BVHInfo;
@group(1) @binding(3) var<storage, read> entity_index_lookup: array<u32>;
@group(1) @binding(4) var<storage, read> entity_flags: array<u32>;

@compute @workgroup_size(128)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_index = global_id.x;
    if (node_index >= tlas_bvh_info.bvh2_count) {
        return;
    }

    let node = tlas_bvh2_bounds[node_index];
    if (!is_leaf(node)) {
        return;
    }

    let prim_store = u32(-node.max.w - 1.0);
    if (prim_store >= arrayLength(&entity_index_lookup)) {
        return;
    }
    let entity_resolved = entity_index_lookup[prim_store];
    if (entity_resolved >= arrayLength(&entity_flags)) {
        return;
    }

    // Scan TLAS leaves instead of every entity so camera/UI motion cannot wake
    // the cache. A single atomic flag makes all visible mature patches sample
    // immediately when ray-visible geometry actually changes.
    if ((entity_flags[entity_resolved] & (EF_DIRTY | EF_MOVED)) != 0u) {
        atomicStore(&counters.force_full_update, 1u);
    }
}
