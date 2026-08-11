#include "common.wgsl"
#include "gi/surface_cache_common.wgsl"

@group(1) @binding(0) var<storage, read_write> counters: SurfaceCacheCounters;

@compute @workgroup_size(1, 1, 1)
fn cs() {
    atomicStore(&counters.active_patch_count, 0u);
    atomicStore(&counters.update_patch_count, 0u);
    atomicStore(&counters.bootstrap_patch_count, 0u);
    atomicStore(&counters.padding, 0u);
}
