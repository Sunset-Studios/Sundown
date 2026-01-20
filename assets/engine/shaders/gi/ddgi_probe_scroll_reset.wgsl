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
@group(1) @binding(2) var<storage, read_write> sample_counts: array<u32>;
@group(1) @binding(3) var<storage, read_write> probe_depth_moments: array<vec4<f32>>;
@group(1) @binding(4) var<storage, read_write> probe_states: array<ProbeStateData>;

@compute @workgroup_size(128, 1, 1)
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

    let sh_base = probe_index * DDGI_SH_PROBE_SIZE_U32;
    for (var i = 0u; i < DDGI_SH_PROBE_SIZE_U32; i = i + 1u) {
        sh_probes[sh_base + i] = 0u;
    }

    sample_counts[probe_index] = 0u;

    let depth_base = probe_index * DDGI_DEPTH_TEXEL_COUNT;
    for (var texel = 0u; texel < DDGI_DEPTH_TEXEL_COUNT; texel = texel + 1u) {
        probe_depth_moments[depth_base + texel] = vec4<f32>(0.0);
    }

    probe_states[probe_index].packed_state = PROBE_STATE_UNINITIALIZED;
    probe_states[probe_index].nearest_hit_dist = 0u;
    probe_states[probe_index].backface_ratio = 0.0;
    probe_states[probe_index].cull_flags = 0u;
    probe_states[probe_index].probe_offset = vec4<f32>(0.0);
}
