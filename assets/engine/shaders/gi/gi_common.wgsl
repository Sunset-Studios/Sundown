// =============================================================================
// GI-1.0 Common Definitions
// Shared structures and utilities for the GI system
// =============================================================================

// =============================================================================
// GI Counters
// - light_count: Number of lights in the scene (copied from lighting system)
// - active_probe_count: Number of probes updated THIS frame (reset each frame)
// 
// Note: Total probe count is derived from grid dimensions and stored in GIParams.total_screen_probes
// =============================================================================
struct GICounters {
    light_count: u32,                      // Number of lights
    active_probe_count: atomic<u32>,       // Probes updated this frame (resets)
    _padding0: u32,
    _padding1: u32,
};

// =============================================================================
// Helper: Pack half float into u32
// =============================================================================
fn pack_half_float(value: f32) -> u32 {
    let clamped = clamp(value, 0.0, 65504.0);
    return u32(clamped * 2.0); // Simple packing (not true fp16, but sufficient for distance comparison)
}

