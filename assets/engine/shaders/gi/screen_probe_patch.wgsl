// =============================================================================
// GI-1.0 Screen Probe Patch Pass (Algorithm 2) - Parallel Version
// 
// This pass implements the "patching" strategy to maintain a fixed ray budget
// while filling in disocclusion holes (empty tiles).
//
// Strategy:
// - We have two queues from reprojection pass:
//   1. empty_tiles: Tiles that need new probes
//   2. override_tiles: Tiles with reprojected probes (candidates for reassignment)
//
// - Goal: Reassign some override tiles to fill empty tiles
// - Method: Parallel processing with per-tile random decision
//
// Parallel Algorithm:
// 1. Dispatch one thread per override tile
// 2. Each thread independently decides whether to "steal" its tile based on:
//    - Random hash (frame-dependent, deterministic)
//    - Target steal rate based on empty/override ratio
// 3. If stealing, atomically append to empty queue
// 4. No serial bottleneck - fully parallel execution
//
// This approach:
// - Avoids serial loops
// - Distributes work across GPU cores
// - Maintains randomness for spatial uniformity
// - Adapts steal rate based on needs (more empty tiles = higher steal rate)
// =============================================================================
#include "common.wgsl"
#include "gi/gi_common.wgsl"

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read_write> empty_tiles: array<u32>;
@group(1) @binding(2) var<storage, read> override_tiles: array<u32>;
@group(1) @binding(3) var<storage, read_write> tile_counters: TileCounters;

@compute @workgroup_size(128, 1, 1)
fn cs(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    // Read current counts (non-atomically for decision making)
    // Note: These are read-only for this thread, so no race condition
    let empty_count = atomicLoad(&tile_counters.empty_count);
    let override_count = atomicLoad(&tile_counters.override_count);
    
    // Bounds check: only process valid override tiles
    if (gid.x >= override_count) {
        return;
    }
    
    // Early exit if no empty tiles to fill
    if (empty_count == 0u) {
        return;
    }
    
    // =========================================================================
    // Adaptive Steal Rate Calculation
    // =========================================================================
    // Calculate target steal probability based on empty/override ratio
    // More empty tiles = higher steal rate to fill disocclusions faster
    // Fewer empty tiles = lower steal rate to preserve temporal stability
    //
    // Target: steal enough to roughly balance ray budget while filling holes
    // We want to steal approximately min(empty_count, override_count/2) tiles
    // =========================================================================
    
    let max_steal = min(empty_count, max(override_count / 2u, 1u));
    
    // Calculate steal probability for this frame
    // Each override tile has (max_steal / override_count) chance of being stolen
    let steal_probability = f32(max_steal) / f32(max(override_count, 1u));
    
    // =========================================================================
    // Per-Tile Random Decision (Deterministic per frame)
    // =========================================================================
    // Generate deterministic random value for this tile this frame
    // Uses tile index + frame index to ensure:
    // - Different tiles make different decisions (spatial distribution)
    // - Same tile makes consistent decision within frame (deterministic)
    // - Different decision each frame (temporal variation)
    // =========================================================================
    
    let seed = u32(gi_params.frame_index) * 1664525u + gid.x * 1013904223u;
    let hash_val = hash_u32(seed);
    let random_value = f32(hash_val) / 4294967295.0; // Normalize to [0, 1]
    
    // Decide whether to steal this override tile
    let should_steal = random_value < steal_probability;
    
    if (should_steal) {
        // =====================================================================
        // STEAL: Convert this override tile to empty tile
        // =====================================================================
        let stolen_tile_index = override_tiles[gid.x];
        
        // Atomically append to empty queue
        let new_empty_index = atomicAdd(&tile_counters.empty_count, 1u);
        empty_tiles[new_empty_index] = stolen_tile_index;
        
        // Note: We don't remove from override_tiles array for efficiency
        // The override tiles that were stolen will still be in the override queue,
        // but they'll also be in the empty queue. The spawn pass only reads
        // from empty_tiles, so this is safe. The queues are reset each frame anyway.
    }
    
    // =====================================================================
    // If not stolen, this override tile keeps its reprojected probe
    // (no action needed - it stays in override queue and won't be in empty queue)
    // =====================================================================
}

// Simple hash function for pseudo-random number generation
// Uses MurmurHash3 finalizer for good bit distribution
fn hash_u32(x: u32) -> u32 {
    var h = x;
    h = h ^ (h >> 16u);
    h = h * 0x85ebca6bu;
    h = h ^ (h >> 13u);
    h = h * 0xc2b2ae35u;
    h = h ^ (h >> 16u);
    return h;
}
