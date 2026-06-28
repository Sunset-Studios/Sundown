// =============================================================================
// DDGI Probe Indices Initialization
// Scatters depth-visible candidates into the per-frame update list by priority.
// =============================================================================

#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read_write> probe_update_indices: array<u32>;
@group(1) @binding(2) var<storage, read> candidate_priorities: array<u32>;
@group(1) @binding(3) var<storage, read> prefix_sum: array<u32>;
@group(1) @binding(4) var<storage, read> block_prefixes: array<u32>;
@group(1) @binding(5) var<storage, read> gi_counters: GICountersReadOnly;

@compute @workgroup_size(128, 1, 1)
fn cs(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let probe_count = u32(ddgi_params.probe_counts.x);
    let slot = gid.x;
    if (slot >= probe_count) {
        return;
    }

    let active_priority = candidate_priorities[slot];
    if (active_priority == DDGI_PROBE_SCHEDULE_PRIORITY_NONE) {
        return;
    }

    let priority_bucket = ddgi_probe_schedule_priority_bucket(active_priority);
    let num_blocks = (arrayLength(&block_prefixes) - DDGI_PROBE_SCHEDULE_PRIORITY_COUNT) /
        DDGI_PROBE_SCHEDULE_PRIORITY_COUNT;
    let priority_prefix =
        prefix_sum[priority_bucket * probe_count + slot] +
        block_prefixes[priority_bucket * num_blocks + wid.x];
    let total_fresh =
        block_prefixes[
            num_blocks * DDGI_PROBE_SCHEDULE_PRIORITY_COUNT +
            DDGI_PROBE_SCHEDULE_PRIORITY_BUCKET_FRESH
        ];
    let output_slot = select(
        total_fresh + priority_prefix,
        priority_prefix,
        priority_bucket == DDGI_PROBE_SCHEDULE_PRIORITY_BUCKET_FRESH
    );

    if (output_slot >= gi_counters.probe_update_count) {
        return;
    }

    let probe_index = ddgi_probe_index_from_permuted_slot(
        slot,
        probe_count,
        u32(ddgi_params.frame_index),
        u32(ddgi_params.permutation_stride),
        u32(ddgi_params.permutation_base_offset),
        u32(ddgi_params.permutation_frame_stride)
    );

    probe_update_indices[output_slot] = probe_index;
}
