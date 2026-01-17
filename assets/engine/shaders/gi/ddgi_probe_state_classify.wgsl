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
@group(1) @binding(2) var<storage, read_write> probe_ray_data: DDGIProbeRayDataBuffer;
@group(1) @binding(3) var<storage, read_write> probe_states: array<ProbeStateData>;
@group(1) @binding(4) var<storage, read_write> gi_counters: GICounters;

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// Analyze ray hit data to determine backface ratio and nearest hit
// Backface hits are encoded as negative t values in the ray data.
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
        let t_raw = probe_ray_data.rays[ray_index].hit_pos_t.w;
        
        // Backface hits are encoded as negative t values.
        // t_raw < 0 = backface hit
        // t_raw > 0 = frontface hit
        let is_hit = probe_ray_data.rays[ray_index].state_u32.w != INVALID_IDX;
        
        if (is_hit) {
            hit_count = hit_count + 1u;
            backface_count = select(backface_count, backface_count + 1u, t_raw < 0.0);
            nearest_hit = select(nearest_hit, min(nearest_hit, t_raw), t_raw > 0.0);
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
    let active_probe_count = atomicLoad(&gi_counters.probe_update_count);
    let rays_per_probe = u32(ddgi_params.probe_counts.y);
    
    if (gid.x >= active_probe_count) {
        return;
    }

    let probe_index = probe_update_indices[gid.x];
    let spacing = ddgi_params.probe_counts.w;
    
    // ─────────────────────────────────────────────────────────────────────────
    // Read current probe state (using read_write version for modify pass)
    // ─────────────────────────────────────────────────────────────────────────
    var current_state = probe_state_get_state(probe_states[probe_index].packed_state);
    var init_frames = probe_state_get_init_frames(probe_states[probe_index].packed_state);
    var convergence_frames = probe_state_get_convergence_frames(probe_states[probe_index].packed_state);
    let flags = probe_state_get_flags(probe_states[probe_index].packed_state);
    
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
            let prev_backface_ratio = probe_states[probe_index].backface_ratio;
            let prev_nearest = bitcast<f32>(probe_states[probe_index].nearest_hit_dist);
            
            // Running average for backface ratio
            let weight = 1.0 / f32(init_frames + 1u);
            let new_avg_backface = mix(prev_backface_ratio, backface_ratio, weight);
            probe_states[probe_index].backface_ratio = new_avg_backface;
            
            // Track minimum nearest hit
            let new_nearest = select(
                min(prev_nearest, nearest_hit),
                nearest_hit,
                init_frames == 0u
            );
            probe_states[probe_index].nearest_hit_dist = bitcast<u32>(new_nearest);
            
            init_frames = init_frames + 1u;
            
            // After enough frames, classify the probe
            if (init_frames >= PROBE_STATE_INIT_FRAMES) {
                current_state = probe_state_classify_initial(
                    probe_states[probe_index].backface_ratio,
                    bitcast<f32>(probe_states[probe_index].nearest_hit_dist),
                    spacing
                );
                
                init_frames = 0u;
                convergence_frames = 0u;
            }
        }
        
        // ═══════════════════════════════════════════════════════════════════
        // OFF: Never update - probe is inside geometry
        // ═══════════════════════════════════════════════════════════════════
        case PROBE_STATE_OFF: {
            // OFF probes stay off permanently
            current_state = PROBE_STATE_OFF;
        }
        
        // ═══════════════════════════════════════════════════════════════════
        // SLEEPING: Probe has no nearby geometry - stays sleeping
        // (Dynamic object wake-up to be handled externally)
        // ═══════════════════════════════════════════════════════════════════
        case PROBE_STATE_SLEEPING: {
            // Sleeping probes stay sleeping until externally woken
            current_state = PROBE_STATE_SLEEPING;
        }
        
        // ═══════════════════════════════════════════════════════════════════
        // NEWLY_VIGILANT: Converging after initial classification
        // ═══════════════════════════════════════════════════════════════════
        case PROBE_STATE_NEWLY_VIGILANT: {
            convergence_frames = convergence_frames + 1u;
            
            if (convergence_frames >= PROBE_STATE_CONVERGENCE_FRAMES) {
                current_state = PROBE_STATE_VIGILANT;
                convergence_frames = 0u;
            }
        }
        
        // ═══════════════════════════════════════════════════════════════════
        // VIGILANT: Always trace - near static geometry
        // ═══════════════════════════════════════════════════════════════════
        case PROBE_STATE_VIGILANT: {
            // Vigilant probes stay vigilant (they shade static geometry)
            current_state = PROBE_STATE_VIGILANT;
        }
        
        // ═══════════════════════════════════════════════════════════════════
        // NEWLY_AWAKE / AWAKE: Reserved for future dynamic object handling
        // ═══════════════════════════════════════════════════════════════════
        case PROBE_STATE_NEWLY_AWAKE, PROBE_STATE_AWAKE: {
            // For now, treat awake probes as vigilant
            current_state = PROBE_STATE_VIGILANT;
        }
        
        default: {
            // Unknown state - reset to uninitialized
            current_state = PROBE_STATE_UNINITIALIZED;
            init_frames = 0u;
            convergence_frames = 0u;
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Write updated state
    // ─────────────────────────────────────────────────────────────────────────
    probe_states[probe_index].packed_state = probe_state_pack(current_state, init_frames, convergence_frames, flags);
}
