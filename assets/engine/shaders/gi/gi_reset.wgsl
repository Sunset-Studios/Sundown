// =============================================================================
// GI-1.0 Reset Pass
// - Resets per-frame counters (active probe count, light count)
// - Active probe count = number of probes selected for update this frame
// - All GPU-side, no CPU readbacks needed
// =============================================================================
#include "common.wgsl"
#include "gi/gi_common.wgsl"

@group(1) @binding(0) var<storage, read_write> gi_counters: GICounters;
@group(1) @binding(1) var<storage, read_write> screen_probe_tile_counters: TileCounters;
@group(1) @binding(2) var<storage, read> light_count_buffer: array<u32>;

@compute @workgroup_size(1, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Only first thread does the work
    if (gid.x == 0u) {
        // Reset active probe count for this frame (how many will be updated)
        atomicStore(&gi_counters.active_probe_count, 0u);
        // Reset active cache cell count for this frame (how many will be updated)
        atomicStore(&gi_counters.active_cache_cell_count, 0u);
        // Reset screen probe empty tile counters
        atomicStore(&screen_probe_tile_counters.empty_count, 0u);
        // Reset screen probe override tile counters
        atomicStore(&screen_probe_tile_counters.override_count, 0u);
        // Copy light count from lighting system buffer
        gi_counters.light_count = light_count_buffer[0];
    }
}

