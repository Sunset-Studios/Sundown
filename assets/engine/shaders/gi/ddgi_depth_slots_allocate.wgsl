#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read> probe_update_indices: array<u32>;
@group(1) @binding(2) var<storage, read> gi_counters: GICountersReadOnly;
@group(1) @binding(3) var<storage, read_write> probe_depth_slots: array<atomic<u32>>;
@group(1) @binding(4) var<storage, read_write> depth_slot_owners: array<atomic<u32>>;
@group(1) @binding(5) var<storage, read_write> depth_slot_last_used: array<u32>;
@group(1) @binding(6) var<storage, read> depth_slot_free_list: array<u32>;
@group(1) @binding(7) var<storage, read_write> depth_slot_allocator: DDGIDepthSlotAllocatorState;
@group(1) @binding(8) var<storage, read_write> probe_depth_moments: array<u32>;

fn ddgi_depth_slot_pop() -> u32 {
    loop {
        let free_count = atomicLoad(&depth_slot_allocator.free_count);
        if (free_count == 0u) {
            return INVALID_IDX;
        }
        let pop = atomicCompareExchangeWeak(
            &depth_slot_allocator.free_count,
            free_count,
            free_count - 1u
        );
        if (pop.exchanged) {
            return depth_slot_free_list[free_count - 1u];
        }
    }
}

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= gi_counters.probe_update_count) {
        return;
    }

    let probe_index = probe_update_indices[gid.x];
    let existing_slot = atomicLoad(&probe_depth_slots[probe_index]);
    let frame_index = u32(ddgi_params.frame_index);
    if (existing_slot != 0u) {
        depth_slot_last_used[existing_slot - 1u] = frame_index;
        return;
    }

    let slot_index = ddgi_depth_slot_pop();
    if (slot_index == INVALID_IDX) {
        atomicAdd(&depth_slot_allocator.allocation_failures, 1u);
        return;
    }

    atomicStore(&depth_slot_owners[slot_index], probe_index + 1u);
    atomicStore(&probe_depth_slots[probe_index], slot_index + 1u);
    depth_slot_last_used[slot_index] = frame_index;

    let invalid_word =
        DDGI_DEPTH_MOMENTS_INVALID_TEXEL |
        (DDGI_DEPTH_MOMENTS_INVALID_TEXEL << 16u);
    let slot_base = ddgi_depth_base_for_slot(&ddgi_params, slot_index);
    let words_per_slot = ddgi_depth_words_per_slot(&ddgi_params);
    for (var word = 0u; word < words_per_slot; word = word + 1u) {
        probe_depth_moments[slot_base + word] = invalid_word;
    }
}
