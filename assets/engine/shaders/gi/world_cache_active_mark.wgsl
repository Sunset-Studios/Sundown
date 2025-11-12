// =============================================================================
// GI-1.0 World Cache Compact
// - Compacts active entries in the world cache using parallel prefix-sum
// - Produces dense array of indices of active entries
// - Generates indirect dispatch parameters for subsequent passes
// =============================================================================
#include "common.wgsl"
#include "gi/gi_common.wgsl"
#include "gi/world_cache_common.wgsl"

// =============================================================================
// Pass 1: Mark Active Entries
// =============================================================================
@group(1) @binding(0) var<storage, read_write> world_cache: array<WorldCacheCell>;
@group(1) @binding(1) var<storage, read_write> active_flags: array<u32>;

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&world_cache)) {
        return;
    }
    active_flags[idx] = select(0u, 1u, atomicLoad(&world_cache[idx].fingerprint) != WORLD_CACHE_CELL_EMPTY);
}
