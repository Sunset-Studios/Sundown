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
var<workgroup> shared_total_count: u32;

@compute @workgroup_size(128, 1, 1)
fn cs(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let idx = gid.x;
    let local_idx = lid.x;
    let array_len = arrayLength(&active_flags_in);
    let num_blocks = arrayLength(&block_sums_in);
    
    // =============================================================================
    // STEP 1: Compute block prefix for scattering
    // =============================================================================
    if (local_idx == 0u) {
        var block_prefix = 0u;
        for (var b = 0u; b < wid.x; b = b + 1u) {
            block_prefix = block_prefix + block_sums_in[b];
        }
        shared_block_prefix = block_prefix;
        
        // Also compute total count from all block sums (do this once)
        if (wid.x == 0u) {
            var total = 0u;
            for (var b = 0u; b < num_blocks; b = b + 1u) {
                total = total + block_sums_in[b];
            }
            shared_total_count = total;
        }
    }
    workgroupBarrier();
    
    let block_prefix = shared_block_prefix;
    
    // =============================================================================
    // STEP 2: Scatter active indices
    // =============================================================================
    if (idx < array_len) {
        let global_prefix = prefix_sum_in[idx] + block_prefix;
        let is_active = active_flags_in[idx];
        
        // If this entry is active, scatter it to the compacted array
        if (is_active != 0u) {
            compacted_indices[global_prefix] = idx;
        }
    }
    
    // =============================================================================
    // STEP 3: Write total count and dispatch params (first workgroup only)
    // =============================================================================
    if (wid.x == 0u && local_idx == 0u) {
        let total_active = shared_total_count;
        
        // Update global counter
        atomicStore(&gi_counters.active_cache_cell_count, total_active);

        // Write dispatch parameters: [x, y, z, total_count]
        // 2x for shadow + primary ray processing
        dispatch_params[0] = (2 * total_active + (WORKGROUP_SIZE - 1u)) / WORKGROUP_SIZE; // Ceiling division
        dispatch_params[1] = 1u;
        dispatch_params[2] = 1u;
    }
}

