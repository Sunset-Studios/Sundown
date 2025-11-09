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
@group(1) @binding(2) var<storage, read> block_sums_in: array<u32>;
@group(1) @binding(3) var<storage, read_write> compacted_indices: array<u32>;
@group(1) @binding(4) var<storage, read_write> dispatch_params: array<u32>; // [x, y, z, total_count]
@group(1) @binding(5) var<storage, read_write> gi_counters: GICounters;

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
    
    // First thread computes the block prefix once for the entire workgroup
    if (local_idx == 0u) {
        var block_prefix = 0u;
        for (var b = 0u; b < wid.x; b = b + 1u) {
            block_prefix = block_prefix + block_sums_in[b];
        }
        shared_block_prefix = block_prefix;
    }
    let block_prefix = workgroupUniformLoad(&shared_block_prefix);

    // Early exit for out-of-bounds threads
    if (idx >= arrayLength(&active_flags_in)) {
        return;
    }
    
    // Get exclusive prefix sum for this element
    let local_prefix = prefix_sum_in[idx];
    let global_prefix = local_prefix + block_prefix;
    
    // If this entry is active, scatter it
    if (active_flags_in[idx] != 0u) {
        compacted_indices[global_prefix] = idx;
    }
    
    // Update total count and dispatch params (only last thread)
    if (idx == arrayLength(&active_flags_in) - 1u) {
        let total_active = global_prefix + active_flags_in[idx];
        atomicStore(&gi_counters.active_cache_cell_count, total_active);

        // Write dispatch parameters: [x, y, z, total_count]
        dispatch_params[0] = (total_active + (WORKGROUP_SIZE - 1u)) / WORKGROUP_SIZE; // Ceiling division
        dispatch_params[1] = 1u;
        dispatch_params[2] = 1u;
    }
}

