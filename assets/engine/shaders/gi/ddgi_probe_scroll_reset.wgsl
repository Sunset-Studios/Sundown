// =============================================================================
// DDGI Probe Scroll Reset Pass
// - Clears newly revealed probe planes after ring-buffer scroll.
// =============================================================================
#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read_write> sh_probes: array<u32>;
@group(1) @binding(2) var<storage, read_write> probe_depth_moments: array<u32>;
@group(1) @binding(3) var<storage, read_write> probe_states: array<ProbeStateData>;

@compute @workgroup_size(256, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let probe_index = gid.x;
    let probe_count = u32(ddgi_params.probe_counts.x);
    if (probe_index >= probe_count) {
        return;
    }

    let cascade_index = ddgi_probe_cascade_index(&ddgi_params, probe_index);
    let snap_active = ddgi_params.cascades[cascade_index].snap_delta.w > 0.0;
    if (!snap_active) {
        return;
    }

    let delta = vec3<i32>(
        i32(ddgi_params.cascades[cascade_index].snap_delta.x),
        i32(ddgi_params.cascades[cascade_index].snap_delta.y),
        i32(ddgi_params.cascades[cascade_index].snap_delta.z)
    );
    let dims = vec3<u32>(
        u32(ddgi_params.probe_grid_dims.x),
        u32(ddgi_params.probe_grid_dims.y),
        u32(ddgi_params.probe_grid_dims.z)
    );
    let world_coord = ddgi_probe_coord_from_index(&ddgi_params, probe_index);

    var reset = false;

    if (delta.x > 0) {
        reset = reset || world_coord.x > dims.x - u32(delta.x) - 1u;
    } else if (delta.x < 0) {
        reset = reset || world_coord.x <= u32(-delta.x);
    }

    if (delta.y > 0) {
        reset = reset || world_coord.y > dims.y - u32(delta.y) - 1u;
    } else if (delta.y < 0) {
        reset = reset || world_coord.y <= u32(-delta.y);
    }

    if (delta.z > 0) {
        reset = reset || world_coord.z > dims.z - u32(delta.z) - 1u;
    } else if (delta.z < 0) {
        reset = reset || world_coord.z <= u32(-delta.z);
    }

    if (!reset) {
        return;
    }

    let cascade_count = ddgi_cascade_count(&ddgi_params);
    let coarser_cascade = cascade_index + 1u;

    // Seed SH from the coarser cascade as a warm start for irradiance.
    // SH coefficients represent angular radiance distribution, which is a
    // reasonable approximation for nearby positions across cascades.
    if (coarser_cascade < cascade_count) {
        let world_pos = ddgi_probe_world_position_from_coord(&ddgi_params, cascade_index, world_coord);
        let coarse_origin = ddgi_cascade_origin(&ddgi_params, coarser_cascade);
        let coarse_spacing = ddgi_cascade_spacing(&ddgi_params, coarser_cascade);
        let coarse_rel = (world_pos - coarse_origin) / coarse_spacing;
        let coarse_coord_f = clamp(
            round(coarse_rel),
            vec3<f32>(0.0),
            vec3<f32>(f32(dims.x - 1u), f32(dims.y - 1u), f32(dims.z - 1u))
        );
        let coarse_coord = vec3<u32>(coarse_coord_f);
        let coarse_probe_index = ddgi_probe_index_from_coord(&ddgi_params, coarser_cascade, coarse_coord);

        let coarse_sh = ddgi_sh_probe_read(&sh_probes, coarse_probe_index);
        ddgi_sh_probe_write(&sh_probes, probe_index, coarse_sh);
        let coarse_sample_count = ddgi_probe_state_get_sample_count(&probe_states[coarse_probe_index]);
        ddgi_probe_state_set_sample_count(&probe_states[probe_index], coarse_sample_count);
    } else {
        let sh_base = probe_index * DDGI_SH_PROBE_SIZE_U32;
        for (var i = 0u; i < DDGI_SH_PROBE_SIZE_U32; i = i + 1u) {
            sh_probes[sh_base + i] = 0u;
        }
        ddgi_probe_state_set_sample_count(&probe_states[probe_index], 0u);
    }

    // Initialize depth moments to "not visible" (0).
    let depth_base = ddgi_depth_base_for_probe(&ddgi_params, probe_index);
    let depth_texel_count = ddgi_depth_texel_count_for_probe(&ddgi_params, probe_index);
    for (var texel = 0u; texel < depth_texel_count; texel = texel + 1u) {
        probe_depth_moments[depth_base + texel] = 0u;
    }

    let reset_sample_count = ddgi_probe_state_get_sample_count(&probe_states[probe_index]);
    ddgi_probe_state_set_sample_count(&probe_states[probe_index], reset_sample_count);
    probe_states[probe_index].packed_state = probe_state_pack(PROBE_STATE_UNINITIALIZED, 0u, 0u, 0u);
}
