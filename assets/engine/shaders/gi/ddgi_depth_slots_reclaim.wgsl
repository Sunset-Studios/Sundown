#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read_write> probe_depth_slots: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read_write> depth_slot_owners: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read> depth_slot_last_used: array<u32>;
@group(1) @binding(4) var<storage, read_write> depth_slot_free_list: array<u32>;
@group(1) @binding(5) var<storage, read_write> depth_slot_allocator: DDGIDepthSlotAllocatorState;

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot_index = gid.x;
    if (slot_index >= ddgi_depth_slot_count(&ddgi_params)) {
        return;
    }

    let encoded_owner = atomicLoad(&depth_slot_owners[slot_index]);
    if (encoded_owner == 0u) {
        return;
    }

    let retention_frames = max(1u, u32(ddgi_params.depth_slot_params.w));
    let frame_index = u32(ddgi_params.frame_index);
    if (frame_index - depth_slot_last_used[slot_index] <= retention_frames) {
        return;
    }

    let release = atomicCompareExchangeWeak(
        &depth_slot_owners[slot_index],
        encoded_owner,
        0u
    );
    if (!release.exchanged) {
        return;
    }

    let probe_index = encoded_owner - 1u;
    let unlink = atomicCompareExchangeWeak(
        &probe_depth_slots[probe_index],
        slot_index + 1u,
        0u
    );
    if (!unlink.exchanged) {
        atomicStore(&depth_slot_owners[slot_index], encoded_owner);
        return;
    }

    let free_index = atomicAdd(&depth_slot_allocator.free_count, 1u);
    depth_slot_free_list[free_index] = slot_index;
}
