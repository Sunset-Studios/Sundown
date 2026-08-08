#include "common.wgsl"
#include "gi/surface_cache_common.wgsl"

@group(1) @binding(0) var<storage, read_write> counters: SurfaceCacheCounters;

// Stale entries are recycled in-place during probing, so frame preparation is
// constant work rather than a full-table eviction scan.
@compute @workgroup_size(1, 1, 1)
fn cs() {
    atomicStore(&counters.active_patch_count, 0u);
    atomicStore(&counters.update_patch_count, 0u);
    atomicStore(&counters.padding1, 0u);
    atomicStore(&counters.padding2, 0u);
}
