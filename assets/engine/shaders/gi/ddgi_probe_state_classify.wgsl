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
@group(1) @binding(2) var<storage, read> probe_ray_allocations: array<vec2<u32>>;
@group(1) @binding(3) var<storage, read_write> probe_ray_data: DDGIProbeRayDataBuffer;
@group(1) @binding(4) var<storage, read_write> probe_states: array<ProbeStateData>;
@group(1) @binding(5) var<storage, read> gi_counters: GICountersReadOnly;

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// Analyze ray hit data to determine backface ratio and nearest hit
// Backface hits are encoded as negative t values in the ray data.
// ─────────────────────────────────────────────────────────────────────────────
struct ProbeRayAnalysis {
    backface_ratio: f32,
    nearest_hit_pos: vec3<f32>,
    nearest_hit_normal: vec3<f32>,
};

fn analyze_probe_rays(
    ray_base: u32,
    rays_per_probe: u32
) -> ProbeRayAnalysis {
    var result: ProbeRayAnalysis;
    result.nearest_hit_pos = vec3<f32>(0.0, 0.0, 0.0);
    result.nearest_hit_normal = vec3<f32>(0.0, 0.0, 0.0);
    var backface_count = 0u;
    var nearest_hit_dist = 1e30;

    for (var i = 0u; i < rays_per_probe; i = i + 1u) {
        let ray_index = ray_base + i;
        let t_raw = probe_ray_data.rays[ray_index].hit_pos_t.w;
        
        // Backface hits are encoded as negative t values.
        // t_raw < 0 = backface hit
        // t_raw > 0 = frontface hit
        let is_hit = probe_ray_data.rays[ray_index].state_u32.w != INVALID_IDX;
        
        if (is_hit) {
            backface_count = select(backface_count, backface_count + 1u, t_raw < 0.0);
            if (t_raw > 0.0 && t_raw < nearest_hit_dist) {
                nearest_hit_dist = t_raw;
                result.nearest_hit_pos = probe_ray_data.rays[ray_index].hit_pos_t.xyz;
                result.nearest_hit_normal = probe_ray_data.rays[ray_index].world_n_section.xyz;
            }
        }
    }
    
    result.backface_ratio = f32(backface_count) / f32(rays_per_probe);

    return result;
}

// =============================================================================
// MAIN COMPUTE SHADER
// =============================================================================

@compute @workgroup_size(256, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    // ─────────────────────────────────────────────────────────────────────────
    // Early exit if beyond probe count
    // ─────────────────────────────────────────────────────────────────────────
    let active_probe_count = gi_counters.probe_update_count;
    
    if (gid.x >= active_probe_count) {
        return;
    }

    let probe_index = probe_update_indices[gid.x];
    let allocation = probe_ray_allocations[gid.x];
    let ray_base = allocation.x;
    let rays_per_probe = max(1u, allocation.y);
    let spacing = ddgi_probe_spacing_from_index(&ddgi_params, probe_index);
    let probe_radius = ddgi_params.probe_grid_dims.w;
    let ray_analysis = analyze_probe_rays(ray_base, rays_per_probe);
    
    // ─────────────────────────────────────────────────────────────────────────
    // Read current probe state (using read_write version for modify pass)
    // ─────────────────────────────────────────────────────────────────────────
    var current_state = probe_state_get_state(probe_states[probe_index].packed_state);
    var init_frames = probe_state_get_init_frames(probe_states[probe_index].packed_state);
    var convergence_frames = probe_state_get_convergence_frames(probe_states[probe_index].packed_state);
    var flags = probe_state_get_flags(probe_states[probe_index].packed_state);

    // We pack convergence frames as a u8, so we need to clamp it.
    convergence_frames = min(convergence_frames + 1u, 255u);
    
    // ─────────────────────────────────────────────────────────────────────────
    // Process based on current state
    // ─────────────────────────────────────────────────────────────────────────
    
    switch (current_state) {
        // ═══════════════════════════════════════════════════════════════════
        // UNINITIALIZED: Accumulate classification data over init frames
        // ═══════════════════════════════════════════════════════════════════
        case PROBE_STATE_UNINITIALIZED: {
            init_frames = init_frames + 1u;
            
            // After enough frames, classify the probe
            if (init_frames >= PROBE_STATE_INIT_FRAMES) {
                current_state = probe_state_classify_initial(ray_analysis.backface_ratio);
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
            if (convergence_frames >= PROBE_STATE_CONVERGENCE_FRAMES) {
                current_state = PROBE_STATE_VIGILANT;
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
