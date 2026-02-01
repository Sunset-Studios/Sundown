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
@group(1) @binding(4) var<storage, read> gi_counters: GICountersReadOnly;

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// Analyze ray hit data to determine backface ratio and nearest hit
// Backface hits are encoded as negative t values in the ray data.
// ─────────────────────────────────────────────────────────────────────────────
struct ProbeRayAnalysis {
    backface_ratio: f32,
    nearest_hit_dist: f32,
    nearest_hit_pos: vec3<f32>,
    nearest_hit_normal: vec3<f32>,
};

fn analyze_probe_rays(
    probe_slot: u32,
    rays_per_probe: u32
) -> ProbeRayAnalysis {
    var result: ProbeRayAnalysis;
    result.nearest_hit_dist = 1e30;
    result.nearest_hit_pos = vec3<f32>(0.0, 0.0, 0.0);
    result.nearest_hit_normal = vec3<f32>(0.0, 0.0, 0.0);
    var backface_count = 0u;

    let ray_base = probe_slot * rays_per_probe;
    
    for (var i = 0u; i < rays_per_probe; i = i + 1u) {
        let ray_index = ray_base + i;
        let t_raw = probe_ray_data.rays[ray_index].hit_pos_t.w;
        
        // Backface hits are encoded as negative t values.
        // t_raw < 0 = backface hit
        // t_raw > 0 = frontface hit
        let is_hit = probe_ray_data.rays[ray_index].state_u32.w != INVALID_IDX;
        
        if (is_hit) {
            backface_count = select(backface_count, backface_count + 1u, t_raw < 0.0);
            if (t_raw > 0.0 && t_raw < result.nearest_hit_dist) {
                result.nearest_hit_dist = t_raw;
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
    let rays_per_probe = u32(ddgi_params.probe_counts.y);
    
    if (gid.x >= active_probe_count) {
        return;
    }

    let probe_slot = gid.x;
    let probe_index = probe_update_indices[gid.x];
    let spacing = ddgi_probe_spacing_from_index(&ddgi_params, probe_index);
    let probe_radius = ddgi_params.probe_grid_dims.w;
    let ray_analysis = analyze_probe_rays(gid.x, rays_per_probe);
    
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
            let backface_ratio = ray_analysis.backface_ratio;
            let nearest_hit = ray_analysis.nearest_hit_dist;
            
            // Accumulate statistics
            let prev_nearest = probe_states[probe_index].nearest_hit_dist;
            
            // Running average for backface ratio
            probe_states[probe_index].backface_ratio = backface_ratio;
            
            // Track minimum nearest hit
            let new_nearest = select(
                min(prev_nearest, nearest_hit),
                nearest_hit,
                init_frames == 0u
            );
            probe_states[probe_index].nearest_hit_dist = new_nearest;
            
            init_frames = init_frames + 1u;
            
            // After enough frames, classify the probe
            if (init_frames >= PROBE_STATE_INIT_FRAMES) {
                current_state = probe_state_classify_initial(
                    probe_states[probe_index].backface_ratio,
                    probe_states[probe_index].nearest_hit_dist,
                    spacing
                );
                
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
    // Probe relocation from hit data (surface alignment)
    // ─────────────────────────────────────────────────────────────────────────
    let near_threshold = spacing * 0.5;
    let base_probe_pos = ddgi_probe_world_position_from_index(&ddgi_params, probe_index);
    let current_probe_pos = ddgi_probe_world_position_from_index_with_offset(&ddgi_params, &probe_states, probe_index);

    var new_offset = probe_states[probe_index].probe_offset.xyz;

    if (init_frames < PROBE_STATE_INIT_FRAMES) {
        let target_distance = min(near_threshold, spacing * 0.5);
        let candidate_pos = ray_analysis.nearest_hit_pos + ray_analysis.nearest_hit_normal * target_distance;

        var candidate_is_clear = true;
        var min_candidate_dist = 1e30;
        let ray_base = probe_slot * rays_per_probe;

        for (var i = 0u; i < rays_per_probe; i = i + 1u) {
            let ray_index = ray_base + i;
            let t_raw = probe_ray_data.rays[ray_index].hit_pos_t.w;
            let is_hit = probe_ray_data.rays[ray_index].state_u32.w != INVALID_IDX;
            let is_front_hit = is_hit && t_raw > 0.0;

            if (is_front_hit) {
                let hit_pos = probe_ray_data.rays[ray_index].hit_pos_t.xyz;
                let dist_current = length(hit_pos - current_probe_pos);
                let dist_candidate = length(hit_pos - candidate_pos);
                min_candidate_dist = min(min_candidate_dist, dist_candidate);

                let gets_closer = dist_candidate + 1e-4 < dist_current;
                candidate_is_clear = candidate_is_clear && !gets_closer;
            }
        }

        if (candidate_is_clear && min_candidate_dist >= target_distance) {
            new_offset = candidate_pos - base_probe_pos;
        } else {
            let mid_pos = (current_probe_pos + candidate_pos) * 0.5;
            var min_mid_dist = 1e30;

            for (var i = 0u; i < rays_per_probe; i = i + 1u) {
                let ray_index = ray_base + i;
                let t_raw = probe_ray_data.rays[ray_index].hit_pos_t.w;
                let is_hit = probe_ray_data.rays[ray_index].state_u32.w != INVALID_IDX;
                let is_front_hit = is_hit && t_raw > 0.0;

                if (is_front_hit) {
                    let hit_pos = probe_ray_data.rays[ray_index].hit_pos_t.xyz;
                    let dist_mid = length(hit_pos - mid_pos);
                    min_mid_dist = min(min_mid_dist, dist_mid);
                }
            }

            if (min_mid_dist >= target_distance) {
                new_offset = mid_pos - base_probe_pos;
            }
        }

        probe_states[probe_index].probe_offset = vec4<f32>(new_offset, probe_states[probe_index].probe_offset.w);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Write updated state
    // ─────────────────────────────────────────────────────────────────────────
    probe_states[probe_index].packed_state = probe_state_pack(current_state, init_frames, convergence_frames, flags);
}
