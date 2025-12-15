// =============================================================================
// GI-1.0 World Cache Compact
// - Compacts active entries in the world cache using parallel prefix-sum
// - Produces dense array of indices of active entries
// - Generates indirect dispatch parameters for subsequent passes
// =============================================================================
#include "common.wgsl"
#include "gi/gi_common.wgsl"

// =============================================================================
// Pass 3: Add Block Prefixes and Scatter
// =============================================================================
@group(1) @binding(0) var<storage, read> active_flags_in: array<u32>;
@group(1) @binding(1) var<storage, read> prefix_sum_in: array<u32>;
@group(1) @binding(2) var<storage, read> block_prefixes_in: array<u32>;
@group(1) @binding(3) var<storage, read_write> compacted_indices: array<u32>;

const WORKGROUP_SIZE = 128u;

// Shared memory to hold the block prefix (computed once per workgroup)
var<workgroup> shared_block_prefix: u32;

@compute @workgroup_size(128, 1, 1)
fn cs(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let idx = gid.x;
    let local_idx = lid.x;
    let array_len = arrayLength(&active_flags_in);
    
    // =============================================================================
    // STEP 1: Compute block prefix for scattering
    // =============================================================================
    if (local_idx == 0u) {
        // O(1): prefix for this workgroup was precomputed from `block_sums_in`.
        shared_block_prefix = block_prefixes_in[wid.x];
    }
    workgroupBarrier();
    
    // =============================================================================
    // STEP 2: Scatter active indices
    // =============================================================================
    if (idx < array_len) {
        let global_prefix = prefix_sum_in[idx] + shared_block_prefix;
        // If this entry is active, scatter it to the compacted array
        if (active_flags_in[idx] != 0u) {
            compacted_indices[global_prefix] = idx;
        }
    }
}

