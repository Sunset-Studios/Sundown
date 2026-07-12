#include "common.wgsl"
#include "gi/scgi_common.wgsl"

@group(1) @binding(0) var<uniform> scgi_params: SCGIParams;
@group(1) @binding(1) var<storage, read_write> surface_cache: array<SurfacePatchReadOnly>;
@group(1) @binding(2) var<storage, read_write> surface_cache_sh: array<u32>;
@group(1) @binding(3) var<storage, read_write> surface_cache_sh_filtered: array<u32>;
@group(1) @binding(4) var<storage, read_write> counters: SCGICounters;

// Reclaim expired cache entries before depth feedback allocates this frame's
// visible surfaces. One invocation owns one patch, so clearing metadata and SH
// has no cross-thread read/modify/write race.
@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let patch_index = gid.x;
    if (patch_index == 0u) {
        atomicStore(&counters.active_patch_count, 0u);
        atomicStore(&counters.padding0, 0u);
        atomicStore(&counters.padding1, 0u);
        atomicStore(&counters.padding2, 0u);
    }
    if (patch_index >= u32(scgi_params.total_patch_count)) {
        return;
    }
    if (surface_cache[patch_index].fingerprint == SCGI_PATCH_EMPTY) {
        return;
    }

    let age = scgi_params.frame_index - surface_cache[patch_index].position_frame.w;
    if (age <= scgi_params.cache_entry_lifetime) {
        return;
    }

    surface_cache[patch_index].position_frame = vec4<f32>(0.0);
    surface_cache[patch_index].normal_unused = vec4<f32>(0.0);
    surface_cache[patch_index].albedo_roughness = vec4<f32>(0.0);
    surface_cache[patch_index].material_props = vec4<f32>(0.0);
    surface_cache[patch_index].history = vec4<f32>(0.0);

    scgi_sh_patch_write(&surface_cache_sh, patch_index, sh_l1_rgb_zero());
    scgi_sh_patch_write(&surface_cache_sh_filtered, patch_index, sh_l1_rgb_zero());

    surface_cache[patch_index].update_frame = 0u;
    surface_cache[patch_index].fingerprint = SCGI_PATCH_EMPTY;
}
