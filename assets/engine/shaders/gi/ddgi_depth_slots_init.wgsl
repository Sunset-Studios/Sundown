#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

@group(1) @binding(0) var<storage, read_write> probe_depth_slots: array<atomic<u32>>;
@group(1) @binding(1) var<storage, read_write> depth_slot_owners: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read_write> depth_slot_last_used: array<u32>;
@group(1) @binding(3) var<storage, read_write> depth_slot_free_list: array<u32>;
@group(1) @binding(4) var<storage, read_write> depth_slot_allocator: DDGIDepthSlotAllocatorState;

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index < arrayLength(&probe_depth_slots)) {
        atomicStore(&probe_depth_slots[index], 0u);
    }
    if (index < arrayLength(&depth_slot_owners)) {
        atomicStore(&depth_slot_owners[index], 0u);
        depth_slot_last_used[index] = 0u;
        depth_slot_free_list[index] = index;
    }
    if (index == 0u) {
        atomicStore(&depth_slot_allocator.free_count, arrayLength(&depth_slot_owners));
        atomicStore(&depth_slot_allocator.allocation_failures, 0u);
        atomicStore(&depth_slot_allocator._pad0, 0u);
        atomicStore(&depth_slot_allocator._pad1, 0u);
    }
}
