#include "common.wgsl"
#include "acceleration_common.wgsl"

// -----------------------------------------------------------------------------
// Data Structures
// -----------------------------------------------------------------------------

struct BVHData {
    leaf_count: u32,
    bvh2_count: u32,
    prim_count: u32,
    prim_base: u32,
    node_base: u32,
    is_blas: u32,
};

// ==================================
// Bindings
// ==================================

@group(1) @binding(0) var<storage, read_write>  bounds            : array<AABB>;
@group(1) @binding(1) var<storage, read_write>  morton_codes      : array<u32>;
@group(1) @binding(2) var<storage, read_write>  bound_indices     : array<u32>;
@group(1) @binding(3) var<storage, read>        scene_aabb        : AABB;
@group(1) @binding(4) var<storage, read>        bvh_info          : BVHData;

// ==================================
// Helpers Functions
// ==================================
fn interleave_bits_32(x: u32) -> u32
{
    var result = x;
    /*
	 * Current Mask:           0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 1111 1111
	 * Which bits to shift:    0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 0000 0000  hex: 0x300
	 * Shifted part (<< 16):   0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 0000 0000 0000 0000 0000 0000  hex: 0x3000000
	 * NonShifted Part:        0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 1111 1111  hex: 0xff
	 * Bitmask is now :        0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 0000 0000 0000 0000 1111 1111  hex: 0x30000ff
	 */
	result = (result | (result << 16)) & 0x30000ff;
	/*
	 * Current Mask:           0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 0000 0000 0000 0000 1111 1111
	 * Which bits to shift:    0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 1111 0000  hex: 0xf0
	 * Shifted part (<< 8):    0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 1111 0000 0000 0000  hex: 0xf000
	 * NonShifted Part:        0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 0000 0000 0000 0000 0000 1111  hex: 0x300000f
	 * Bitmask is now :        0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 0000 0000 1111 0000 0000 1111  hex: 0x300f00f
	 */
	result = (result | (result << 8)) & 0x300f00f;
	/*
	 * Current Mask:           0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 0000 0000 1111 0000 0000 1111
	 * Which bits to shift:    0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 1100 0000 0000 1100  hex: 0xc00c
	 * Shifted part (<< 4):    0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 1100 0000 0000 1100 0000  hex: 0xc00c0
	 * NonShifted Part:        0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 0000 0000 0011 0000 0000 0011  hex: 0x3003003
	 * Bitmask is now :        0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 0000 1100 0011 0000 1100 0011  hex: 0x30c30c3
	 */
	result = (result | (result << 4)) & 0x30c30c3;
	/*
	 * Current Mask:           0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 0000 1100 0011 0000 1100 0011
	 * Which bits to shift:    0000 0000 0000 0000 0000 0000 0000 0000 0000 0010 0000 1000 0010 0000 1000 0010  hex: 0x2082082
	 * Shifted part (<< 2):    0000 0000 0000 0000 0000 0000 0000 0000 0000 1000 0010 0000 1000 0010 0000 1000  hex: 0x8208208
	 * NonShifted Part:        0000 0000 0000 0000 0000 0000 0000 0000 0000 0001 0000 0100 0001 0000 0100 0001  hex: 0x1041041
	 * Bitmask is now :        0000 0000 0000 0000 0000 0000 0000 0000 0000 1001 0010 0100 1001 0010 0100 1001  hex: 0x9249249
	 */
	result = (result | (result << 2)) & 0x9249249;

	return result;
}

fn morton_code(p: vec3<f32>) -> u32 {
    let scene_size = scene_aabb.max.xyz - scene_aabb.min.xyz;
    let safe_size = select(scene_size, vec3<f32>(1.0), scene_size == vec3<f32>(0.0));
    let normalized_p = (p - scene_aabb.min.xyz) / safe_size;
    let x = u32(normalized_p.x * 0x3ff);
    let y = u32(normalized_p.y * 0x3ff);
    let z = u32(normalized_p.z * 0x3ff);
    return interleave_bits_32(x) | (interleave_bits_32(y) << 1) | (interleave_bits_32(z) << 2);
}

// ==================================
// Kernels
// ==================================

@compute @workgroup_size(256)
fn compute_morton_codes(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&bounds)) { return; }
    // bvh_info layout compatibility: [leaf_count, bvh2_count, prim_count, prim_base]
    let prim_base = bvh_info.prim_base;
    let bound = bounds[prim_base + gid.x];
    let center = (bound.min + bound.max) * 0.5;
    let extent = bound.max - bound.min;
    let is_invalid = all(bound.min.xyz == vec3<f32>(0.0)) && all(bound.max.xyz == vec3<f32>(0.0));
    morton_codes[gid.x] = select(morton_code(center.xyz), INVALID_IDX, is_invalid);
    bound_indices[gid.x] = gid.x;
}