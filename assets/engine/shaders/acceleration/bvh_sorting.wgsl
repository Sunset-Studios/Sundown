#include "common.wgsl"
#include "acceleration_common.wgsl"

// H-PLOC Step 1: GPU-based morton code calculation and radix sort.

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

const radix_bits = 4u;
const radix = 1u << radix_bits;

// -----------------------------------------------------------------------------
// Data Structures
// -----------------------------------------------------------------------------
struct SortUniforms {
  bit_start: u32,
  element_count: u32,
}

//------------------------------------------------------------------------------
// Utility Functions
//------------------------------------------------------------------------------
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

//------------------------------------------------------------------------------
// Bindings & Uniforms
//------------------------------------------------------------------------------
@group(1) @binding(0) var<storage, read> bounds: array<AABB>;
@group(1) @binding(1) var<storage, read_write> in_morton_codes: array<u32>;
@group(1) @binding(2) var<storage, read_write> in_sorted_indices: array<u32>;
@group(1) @binding(3) var<storage, read_write> out_morton_codes: array<u32>;
@group(1) @binding(4) var<storage, read_write> out_sorted_indices: array<u32>;
@group(1) @binding(5) var<storage, read_write> histogram: array<atomic<u32>, radix>;
@group(1) @binding(6) var<uniform> scene_aabb: AABB;
@group(1) @binding(7) var<uniform> sort_uniforms: SortUniforms;

// Shared histogram for the current workgroup (one bin per radix digit).
var<workgroup> local_histogram: array<atomic<u32>, radix>;

//------------------------------------------------------------------------------
// Morton Code Calculation & Sorting
//------------------------------------------------------------------------------

@compute @workgroup_size(256)
fn compute_morton_codes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let prim_idx = gid.x;
    if (prim_idx >= arrayLength(&bounds)) { return; }
    let bound = bounds[prim_idx];
    let center = (bound.min + bound.max) * 0.5;
    in_morton_codes[prim_idx] = morton_code(center.xyz);
    in_sorted_indices[prim_idx] = prim_idx;
}

// Clear the global radix histogram between passes
@compute @workgroup_size(256)
fn clear_histogram(@builtin(local_invocation_id) lid: vec3<u32>) {
    if (lid.x < radix) {
        atomicStore(&histogram[lid.x], 0u);
    }
}

@compute @workgroup_size(256)
fn compute_histogram(@builtin(global_invocation_id) gid: vec3<u32>,
                     @builtin(local_invocation_id) lid: vec3<u32>) {
    // Initialise the per-workgroup histogram in shared memory.
    if (lid.x < radix) {
        atomicStore(&local_histogram[lid.x], 0u);
    }
    workgroupBarrier();

    // Accumulate counts into the local histogram.
    if (gid.x < sort_uniforms.element_count) {
        let key = in_morton_codes[gid.x];
        let digit = (key >> sort_uniforms.bit_start) & (radix - 1u);
        atomicAdd(&local_histogram[digit], 1u);
    }
    workgroupBarrier();

    // One thread per digit writes the local result to the global histogram.
    if (lid.x < radix) {
        let count = atomicLoad(&local_histogram[lid.x]);
        if (count > 0u) {
            atomicAdd(&histogram[lid.x], count);
        }
    }
}

@compute @workgroup_size(256)
fn prefix_sum(@builtin(local_invocation_id) lid: vec3<u32>) {
    // Load global histogram into shared memory.
    if (lid.x < radix) {
        let value = atomicLoad(&histogram[lid.x]);
        atomicStore(&local_histogram[lid.x], value);
    }
    workgroupBarrier();

    // Serial exclusive scan within shared memory by thread 0.
    if (lid.x == 0u) {
        var running_total: u32 = 0u;
        for (var i: u32 = 0u; i < radix; i = i + 1u) {
            let tmp = atomicLoad(&local_histogram[i]);
            atomicStore(&local_histogram[i], running_total);
            running_total = running_total + tmp;
        }
    }
    workgroupBarrier();

    // Write results back to the global histogram buffer.
    if (lid.x < radix) {
        atomicStore(&histogram[lid.x], atomicLoad(&local_histogram[lid.x]));
    }
}

@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>) {
    // --- Phase 0 : Initialise shared memory ---
    if (lid.x < radix) {
        // Reset per-digit counters / offsets.
        atomicStore(&local_histogram[lid.x], 0u);
    }
    workgroupBarrier();

    // Determine whether this thread corresponds to a valid element.
    let active_thread = gid.x < sort_uniforms.element_count;

    // --- Phase 1 : Per-workgroup counting & local rank ---
    var local_rank: u32 = 0u;
    var digit: u32 = 0u;
    var key: u32 = 0u;
    var value: u32 = 0u;

    if (active_thread) {
        key = in_morton_codes[gid.x];
        value = in_sorted_indices[gid.x];
        digit = (key >> sort_uniforms.bit_start) & (radix - 1u);
        // atomicAdd returns the previous value, giving us the rank within this digit
        local_rank = atomicAdd(&local_histogram[digit], 1u);
    }
    workgroupBarrier();

    // --- Phase 2 : Reserve global space once per digit ---
    if (lid.x == 0u) {
        for (var i: u32 = 0u; i < radix; i = i + 1u) {
            let count = atomicLoad(&local_histogram[i]); // number of elements of digit i in this WG
            let base  = atomicAdd(&histogram[i], count); // reserve slice in the global array
            atomicStore(&local_histogram[i], base);      // reuse slot to store base offset
        }
    }
    workgroupBarrier();

    // --- Phase 3 : Scatter without further global atomics ---
    if (active_thread) {
        let base = atomicLoad(&local_histogram[digit]);
        let pos  = base + local_rank;
        out_morton_codes[pos]   = key;
        out_sorted_indices[pos] = value;
    }
}
