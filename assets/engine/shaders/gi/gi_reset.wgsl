// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                        GI RESET PASS                                      ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Resets per-frame counters for the GI system:                             ║
// ║  • Active pixel count (for per-pixel tracing)                             ║
// ║  • Active cache cell count (for world cache updates)                      ║
// ║  • Light count (copied from lighting system)                              ║
// ║                                                                           ║
// ║  All GPU-side, no CPU readbacks needed.                                   ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

#include "common.wgsl"
#include "gi/gi_common.wgsl"

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<storage, read_write> gi_counters: GICounters;
@group(1) @binding(1) var<storage, read> dense_lights_buffer: DenseLightsBuffer;

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
        atomicStore(&gi_counters.ray_queue_primary_head, 0u);

        // Reset ray queue count (will be incremented by active rays in init pass)
        atomicStore(&gi_counters.ray_queue_count, 0u);

        // Reset DDGI probe update count (compacted active probes for tracing)
        atomicStore(&gi_counters.probe_update_count, 0u);
        
        gi_counters.light_count = dense_lights_buffer.header.light_count;
    }
}
