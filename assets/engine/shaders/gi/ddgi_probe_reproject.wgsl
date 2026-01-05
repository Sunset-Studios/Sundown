// =============================================================================
// DDGI Probe Radiance Reprojection (Snap Shift)
// - Runs ONLY when the probe grid "snaps" with the camera (origin moves by whole
//   multiples of probe spacing).
// - Shifts the per-probe radiance history buffer by the snap delta so probes that
//   remain in the grid keep their history.
// - Newly uncovered probes are initialized to NaN so accumulation can treat them
//   as "no history" and seed with fresh samples.
// =============================================================================

#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read> probe_radiance_in: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read_write> probe_radiance_out: array<vec4<f32>>;

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let probe_count = u32(ddgi_params.probe_counts.x);
    if (gid.x >= probe_count) {
        return;
    }

    let dims = vec3<i32>(
        i32(ddgi_params.probe_grid_dims.x),
        i32(ddgi_params.probe_grid_dims.y),
        i32(ddgi_params.probe_grid_dims.z)
    );

    let dst_coord = ddgi_probe_coord_from_index(&ddgi_params, gid.x);
    let dst_coord_i32 = vec3<i32>(dst_coord);
    let src_coord_i32 = dst_coord_i32 + vec3<i32>(ddgi_params.probe_grid_snap_delta.xyz);
    let src_coord = vec3<u32>(src_coord_i32);

    let in_bounds =
        src_coord_i32.x >= 0 && src_coord_i32.x < dims.x &&
        src_coord_i32.y >= 0 && src_coord_i32.y < dims.y &&
        src_coord_i32.z >= 0 && src_coord_i32.z < dims.z;
    let src_index = ddgi_probe_index_from_coord(&ddgi_params, src_coord);

    if (in_bounds && src_index != gid.x) {
        probe_radiance_out[gid.x] = probe_radiance_in[src_index];
    }
}


