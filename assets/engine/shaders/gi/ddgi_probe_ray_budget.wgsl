#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read> probe_update_indices: array<u32>;
@group(1) @binding(2) var<storage, read> probe_states: array<ProbeStateData>;
@group(1) @binding(3) var<storage, read> gi_counters: GICountersReadOnly;
@group(1) @binding(4) var<storage, read_write> probe_ray_allocations: array<vec2<u32>>; // x=base, y=count
@group(1) @binding(5) var<storage, read_write> probe_ray_data: DDGIProbeRayDataBuffer;

fn ddgi_probe_ray_priority(state_data: ProbeStateData) -> f32 {
    let packed = state_data.packed_state;
    let state = probe_state_get_state(packed);
    let convergence_frames = probe_state_get_convergence_frames(packed);

    if (state == PROBE_STATE_UNINITIALIZED) {
        return 3.0;
    }
    if (state == PROBE_STATE_NEWLY_VIGILANT || state == PROBE_STATE_NEWLY_AWAKE) {
        return 2.5;
    }
    if (state == PROBE_STATE_OFF || state == PROBE_STATE_SLEEPING) {
        return 0.5;
    }

    // Prefer probes that are still noisy / less converged.
    let convergence_t = clamp(f32(convergence_frames) / 64.0, 0.0, 1.0);
    let convergence_boost = 1.0 - convergence_t;
    return 1.0 + convergence_boost;
}

@compute @workgroup_size(256, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let active_probe_count = gi_counters.probe_update_count;
    let max_probes_per_frame = u32(ddgi_params.probe_counts.z);
    let max_rays_per_probe = max(1u, u32(ddgi_params.probe_counts.y));
    let min_rays_per_probe = min(max_rays_per_probe, max(1u, u32(ddgi_params.min_rays_per_probe)));
    let probe_in_frame = gid.x < max_probes_per_frame;
    let probe_is_active = probe_in_frame && gid.x < active_probe_count;

    if (gid.x == 0u) {
        atomicStore(&probe_ray_data.header.active_ray_count, 0u);
    }

    if (probe_in_frame && !probe_is_active) {
        probe_ray_allocations[gid.x] = vec2<u32>(0u, 0u);
    }

    let extra_capacity = max_rays_per_probe - min_rays_per_probe;
    if (probe_is_active) {
        // First pass: per-probe extra-ray demand estimate (stored in .x temporarily).
        let probe_index = probe_update_indices[gid.x];
        let priority = ddgi_probe_ray_priority(probe_states[probe_index]);
        let extra_weight = clamp((priority - 1.0) * 0.5, 0.0, 1.0);
        let desired_extra = min(extra_capacity, u32(extra_weight * f32(extra_capacity) + 0.5));
        probe_ray_allocations[gid.x] = vec2<u32>(desired_extra, 0u);
    }

    workgroupBarrier();
    storageBarrier();

    // Simple single-thread finalize pass keeps logic compact and deterministic.
    if (gid.x != 0u) {
        return;
    }

    var total = 0u;
    for (var i = 0u; i < active_probe_count; i = i + 1u) {
        total += min_rays_per_probe + probe_ray_allocations[i].x;
    }

    let target_total = active_probe_count * ((min_rays_per_probe + max_rays_per_probe) / 2u);

    // Keep total budget near target by trimming or adding extras.
    if (total > target_total) {
        var over = total - target_total;
        var i = 0u;
        loop {
            if (over == 0u || i >= active_probe_count * 2u) { break; }
            let slot = i % active_probe_count;
            let extra = probe_ray_allocations[slot].x;
            if (extra > 0u) {
                probe_ray_allocations[slot].x = extra - 1u;
                over -= 1u;
            }
            i += 1u;
        }
    } else if (total < target_total) {
        var missing = target_total - total;
        var i = 0u;
        loop {
            if (missing == 0u || i >= active_probe_count * 2u) { break; }
            let slot = i % active_probe_count;
            let extra = probe_ray_allocations[slot].x;
            if (extra < extra_capacity) {
                probe_ray_allocations[slot].x = extra + 1u;
                missing -= 1u;
            }
            i += 1u;
        }
    }

    var ray_base = 0u;
    for (var i = 0u; i < active_probe_count; i = i + 1u) {
        let ray_count = min_rays_per_probe + probe_ray_allocations[i].x;
        probe_ray_allocations[i] = vec2<u32>(ray_base, ray_count);
        ray_base += ray_count;
    }

    atomicStore(&probe_ray_data.header.active_ray_count, ray_base);
}
