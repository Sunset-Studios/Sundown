// =============================================================================
// DDGI Active Probe Block Prefix Scan
// Scans per-workgroup priority totals and publishes scheduling counters.
// =============================================================================

#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

@group(1) @binding(0) var<storage, read> block_sums_in: array<u32>;
@group(1) @binding(1) var<storage, read_write> block_prefixes_out: array<u32>;
@group(1) @binding(2) var<storage, read_write> gi_counters: GICounters;
@group(1) @binding(3) var<uniform> ddgi_params: DDGIParams;

const WORKGROUP_SIZE = 128u;

var<workgroup> shared_subgroup_sums: array<u32, 32>;
var<workgroup> shared_running_offset: u32;
var<workgroup> shared_chunk_total: u32;
var<workgroup> shared_total_active: u32;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn cs(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_size) sg_size: u32
) {
    let local_idx = lid.x;
    let num_blocks = arrayLength(&block_sums_in) / DDGI_PROBE_SCHEDULE_PRIORITY_COUNT;
    let priority_totals_base = num_blocks * DDGI_PROBE_SCHEDULE_PRIORITY_COUNT;

    if (local_idx == 0u) {
        shared_total_active = 0u;
    }
    workgroupBarrier();

    for (
        var priority_bucket = 0u;
        priority_bucket < DDGI_PROBE_SCHEDULE_PRIORITY_COUNT;
        priority_bucket = priority_bucket + 1u
    ) {
        if (local_idx == 0u) {
            shared_running_offset = 0u;
        }
        workgroupBarrier();

        for (var chunk_base = 0u; chunk_base < num_blocks; chunk_base = chunk_base + WORKGROUP_SIZE) {
            let chunk_idx = chunk_base + local_idx;
            var value = 0u;
            if (chunk_idx < num_blocks) {
                value = block_sums_in[priority_bucket * num_blocks + chunk_idx];
            }

            let warp_ctx = make_warp_ctx(local_idx, sg_lane, sg_size);
            let subgroup_exclusive = warp_scan_exclusive_add_u32(warp_ctx, value);
            let subgroup_total = warp_reduce_add_u32(warp_ctx, value);

            if (is_warp_leader(warp_ctx)) {
                shared_subgroup_sums[warp_ctx.warp_id] = subgroup_total;
            }
            workgroupBarrier();

            var prefix_from_prev_sg = 0u;
            for (var s = 0u; s < warp_ctx.warp_id; s = s + 1u) {
                prefix_from_prev_sg = prefix_from_prev_sg + shared_subgroup_sums[s];
            }

            let chunk_exclusive = subgroup_exclusive + prefix_from_prev_sg;

            if (local_idx == 0u) {
                var chunk_total = 0u;
                let subgroup_count = (WORKGROUP_SIZE + sg_size - 1u) / sg_size;
                for (var ss = 0u; ss < subgroup_count; ss = ss + 1u) {
                    chunk_total = chunk_total + shared_subgroup_sums[ss];
                }
                shared_chunk_total = chunk_total;
            }
            workgroupBarrier();

            let running_offset = shared_running_offset;
            if (chunk_idx < num_blocks) {
                block_prefixes_out[priority_bucket * num_blocks + chunk_idx] = running_offset + chunk_exclusive;
            }
            workgroupBarrier();

            if (local_idx == 0u) {
                shared_running_offset = running_offset + shared_chunk_total;
            }
            workgroupBarrier();
        }

        if (local_idx == 0u) {
            let priority_total = shared_running_offset;
            shared_total_active = shared_total_active + priority_total;
            block_prefixes_out[priority_totals_base + priority_bucket] = priority_total;
        }
        workgroupBarrier();
    }

    if (local_idx == 0u) {
        let total_active_probes = shared_total_active;
        let probes_per_frame = u32(ddgi_params.probe_counts.z);
        atomicStore(&gi_counters.probe_update_count, min(total_active_probes, probes_per_frame));
        atomicStore(&gi_counters.ray_queue_shadow_head, total_active_probes);
        atomicStore(&gi_counters.ray_queue_primary_head, 0u);
    }
}
