// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                    DDGI PROBE STATE CLASSIFICATION                        ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Classifies probes into states for adaptive update scheduling.            ║
// ║                                                                           ║
// ║  This shader runs after probe tracing and performs:                       ║
// ║  1. Initial classification of UNINITIALIZED probes                        ║
// ║     - OFF: inside static geometry (>70% backface hits)                    ║
// ║     - SLEEPING: no geometry within probe_spacing                          ║
// ║     - NEWLY_VIGILANT: near static geometry                                ║
// ║  2. State transitions for convergence                                     ║
// ║     - NEWLY_VIGILANT → VIGILANT after convergence frames                  ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read> probe_update_indices: array<u32>;
@group(1) @binding(2) var<storage, read> probe_ray_data: array<DDGIProbeRayData>;
@group(1) @binding(3) var<storage, read_write> probe_states: array<u32>;

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// Analyze ray hit data to determine backface ratio and nearest hit
// ─────────────────────────────────────────────────────────────────────────────
fn analyze_probe_rays(
    probe_slot: u32,
    rays_per_probe: u32
) -> vec2<f32> {
    // Returns: x = backface_ratio, y = nearest_hit_distance
    var backface_count = 0u;
    var hit_count = 0u;
    var nearest_hit = 1e30;
    
    let ray_base = probe_slot * rays_per_probe;
    
    for (var i = 0u; i < rays_per_probe; i = i + 1u) {
        let ray_index = ray_base + i;
        let hit_data = probe_ray_data[ray_index];
        let t = hit_data.hit_pos_t.w;
        
        // Check if ray hit something (t > 0)
        if (t > 0.0) {
            hit_count = hit_count + 1u;
            nearest_hit = min(nearest_hit, t);
            
            // Check if backface hit (flagged in state_u32.y bit 1)
            let is_backface = (hit_data.state_u32.y & 2u) != 0u;
            if (is_backface) {
                backface_count = backface_count + 1u;
            }
        }
    }
    
    let backface_ratio = select(
        0.0,
        f32(backface_count) / f32(hit_count),
        hit_count > 0u
    );
    
    return vec2<f32>(backface_ratio, nearest_hit);
}

// =============================================================================
// MAIN COMPUTE SHADER
// =============================================================================

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    // ─────────────────────────────────────────────────────────────────────────
    // Early exit if beyond probe count
    // ─────────────────────────────────────────────────────────────────────────
    let probes_per_frame = u32(ddgi_params.probe_counts.z);
    let rays_per_probe = u32(ddgi_params.probe_counts.y);
    let probe_spacing = ddgi_params.probe_counts.w;
    
    if (gid.x >= probes_per_frame) {
        return;
    }
    
    let probe_index = probe_update_indices[gid.x];
    
    // ─────────────────────────────────────────────────────────────────────────
    // Read current probe state (using read_write version for modify pass)
    // ─────────────────────────────────────────────────────────────────────────
    var state_data = probe_state_read_rw(&probe_states, probe_index);
    let current_state = probe_state_get_state(state_data.packed_state);
    var init_frames = probe_state_get_init_frames(state_data.packed_state);
    var convergence_frames = probe_state_get_convergence_frames(state_data.packed_state);
    let flags = probe_state_get_flags(state_data.packed_state);
    
    var new_state = current_state;
    
    // ─────────────────────────────────────────────────────────────────────────
    // Process based on current state
    // ─────────────────────────────────────────────────────────────────────────
    
    switch (current_state) {
        // ═══════════════════════════════════════════════════════════════════
        // UNINITIALIZED: Accumulate classification data over init frames
        // ═══════════════════════════════════════════════════════════════════
        case PROBE_STATE_UNINITIALIZED: {
            let ray_analysis = analyze_probe_rays(gid.x, rays_per_probe);
            let backface_ratio = ray_analysis.x;
            let nearest_hit = ray_analysis.y;
            
            // Accumulate statistics
            let prev_backface_count = state_data.backface_count;
            let prev_nearest = bitcast<f32>(state_data.nearest_hit_dist);
            
            // Running average for backface ratio
            let weight = 1.0 / f32(init_frames + 1u);
            let avg_backface = f32(prev_backface_count) / 100.0;
            let new_avg_backface = mix(avg_backface, backface_ratio, weight);
            state_data.backface_count = u32(new_avg_backface * 100.0);
            
            // Track minimum nearest hit
            let new_nearest = select(
                min(prev_nearest, nearest_hit),
                nearest_hit,
                init_frames == 0u
            );
            state_data.nearest_hit_dist = bitcast<u32>(new_nearest);
            
            init_frames = init_frames + 1u;
            
            // After enough frames, classify the probe
            if (init_frames >= PROBE_STATE_INIT_FRAMES) {
                let final_backface_ratio = f32(state_data.backface_count) / 100.0;
                let final_nearest = bitcast<f32>(state_data.nearest_hit_dist);
                
                new_state = probe_state_classify_initial(
                    final_backface_ratio,
                    final_nearest,
                    probe_spacing
                );
                
                // Reset counters for the new state
                init_frames = 0u;
                convergence_frames = 0u;
            }
        }
        
        // ═══════════════════════════════════════════════════════════════════
        // OFF: Never update - probe is inside geometry
        // ═══════════════════════════════════════════════════════════════════
        case PROBE_STATE_OFF: {
            // OFF probes stay off permanently
            new_state = PROBE_STATE_OFF;
        }
        
        // ═══════════════════════════════════════════════════════════════════
        // SLEEPING: Probe has no nearby geometry - stays sleeping
        // (Dynamic object wake-up to be handled externally)
        // ═══════════════════════════════════════════════════════════════════
        case PROBE_STATE_SLEEPING: {
            // Sleeping probes stay sleeping until externally woken
            new_state = PROBE_STATE_SLEEPING;
        }
        
        // ═══════════════════════════════════════════════════════════════════
        // NEWLY_VIGILANT: Converging after initial classification
        // ═══════════════════════════════════════════════════════════════════
        case PROBE_STATE_NEWLY_VIGILANT: {
            convergence_frames = convergence_frames + 1u;
            
            if (convergence_frames >= PROBE_STATE_CONVERGENCE_FRAMES) {
                new_state = PROBE_STATE_VIGILANT;
                convergence_frames = 0u;
            }
        }
        
        // ═══════════════════════════════════════════════════════════════════
        // VIGILANT: Always trace - near static geometry
        // ═══════════════════════════════════════════════════════════════════
        case PROBE_STATE_VIGILANT: {
            // Vigilant probes stay vigilant (they shade static geometry)
            new_state = PROBE_STATE_VIGILANT;
        }
        
        // ═══════════════════════════════════════════════════════════════════
        // NEWLY_AWAKE / AWAKE: Reserved for future dynamic object handling
        // ═══════════════════════════════════════════════════════════════════
        case PROBE_STATE_NEWLY_AWAKE, PROBE_STATE_AWAKE: {
            // For now, treat awake probes as vigilant
            new_state = PROBE_STATE_VIGILANT;
        }
        
        default: {
            // Unknown state - reset to uninitialized
            new_state = PROBE_STATE_UNINITIALIZED;
            init_frames = 0u;
            convergence_frames = 0u;
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Write updated state
    // ─────────────────────────────────────────────────────────────────────────
    state_data.packed_state = probe_state_pack(new_state, init_frames, convergence_frames, flags);
    probe_state_write(&probe_states, probe_index, state_data);
}
