#include "common.wgsl"
#include "acceleration_common.wgsl"

// ==================================
// Bindings
// ==================================

@group(1) @binding(0) var<storage, read_write>  bounds            : array<AABB>;
@group(1) @binding(1) var<storage, read_write>  morton_codes      : array<u32>;
@group(1) @binding(2) var<storage, read_write>  bound_indices     : array<u32>;
@group(1) @binding(3) var<storage, read>        scene_aabb        : AABB;

// ==================================
// Helpers Functions
// ==================================
// 10 bits per axis, 30 bits total.
fn morton_code(p: vec3<f32>) -> u32 {
    let scene_size = scene_aabb.max.xyz - scene_aabb.min.xyz;
    let safe_size = select(scene_size, vec3<f32>(1.0), scene_size == vec3<f32>(0.0));
    let normalized_p = (p - scene_aabb.min.xyz) / safe_size;
    let x = min(max(u32(normalized_p.x * 1023.0), 0u), 1023u);
    let y = min(max(u32(normalized_p.y * 1023.0), 0u), 1023u);
    let z = min(max(u32(normalized_p.z * 1023.0), 0u), 1023u);
    var code: u32 = 0u;
    for (var i: u32 = 0u; i < 10u; i = i + 1u) {
        let bit_mask = 1u << i;
        code = code | ((x & bit_mask) << (2u * i)) 
                   | ((y & bit_mask) << (2u * i + 1u)) 
                   | ((z & bit_mask) << (2u * i + 2u));
    }
    return code;
}

// ==================================
// Kernels
// ==================================

@compute @workgroup_size(256)
fn compute_morton_codes(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&bounds)) { return; }
    let bound = bounds[gid.x];
    let center = (bound.min + bound.max) * 0.5;
    morton_codes[gid.x] = morton_code(center.xyz);
    bound_indices[gid.x] = gid.x;
}