// =============================================================================
// DDGI Reset Pass
// - Resets per-frame counters for the DDGI system:
//   - Active cache cell count
//   - Shadow and primary ray queue heads
//   - Ray queue count
//   - DDGI probe update count
//   - Probe ray data header active ray count
//   - Light count
// =============================================================================
#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<storage, read_write> gi_counters: GICounters;
@group(1) @binding(1) var<storage, read> dense_lights_buffer: DenseLightsBuffer;
@group(1) @binding(2) var<storage, read_write> probe_ray_data: DDGIProbeRayDataBuffer;

// =============================================================================
// MAIN COMPUTE SHADER
// =============================================================================

@compute @workgroup_size(1, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Only first thread does the work
    if (gid.x == 0u) {
        // Reset active cache cell count for this frame
        atomicStore(&gi_counters.active_cache_cell_count, 0u);
        // Reset both shadow and primary ray queue heads for parallel traversal
        atomicStore(&gi_counters.ray_queue_shadow_head, 0u);
        // Reset primary ray queue head for parallel traversal
        atomicStore(&gi_counters.ray_queue_primary_head, 0u);
        // Reset ray queue count (will be incremented by active rays in init pass)
        atomicStore(&gi_counters.ray_queue_count, 0u);
        // Reset DDGI probe update count (compacted active probes for tracing)
        atomicStore(&gi_counters.probe_update_count, 0u);
        // Reset probe ray data header active ray count
        atomicStore(&probe_ray_data.header.active_ray_count, 0u);
        // Copy light count from dense lights buffer
        gi_counters.light_count = dense_lights_buffer.header.light_count;
    }
}

