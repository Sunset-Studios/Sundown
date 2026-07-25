#include "gi/svlm_common.wgsl"

// Compacts all visible-level leaves into the probe debug source list.

@group(1) @binding(0) var<storage, read_write> svlm_params: SVLMParams;
@group(1) @binding(1) var<storage, read_write> svlm_counters: SVLMCounters;
@group(1) @binding(2) var<storage, read> leaf_bricks: array<SVLMLeafBrick>;
@group(1) @binding(3) var<storage, read_write> debug_leaf_indices: array<atomic<u32>>;

fn svlm_leaf_visible_for_debug_level(leaf_level: u32) -> bool {
    let debug_level = i32(svlm_params.debug_level);
    return debug_level < 0 || leaf_level == u32(debug_level);
}

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let page_groups_x = max(u32(max(svlm_params.debug_gather_page_groups_x, 0.0)), 1u);
    let page_groups_y = max(u32(max(svlm_params.debug_gather_page_groups_y, 0.0)), 1u);
    let page_group = gid.y + gid.z * page_groups_y;
    let leaf_index = gid.x + page_group * page_groups_x * 128u;
    let leaf_count = min(atomicLoad(&svlm_counters.leaf_count), arrayLength(&leaf_bricks));
    if (leaf_index >= leaf_count || leaf_index >= arrayLength(&leaf_bricks)) {
        return;
    }

    let leaf = leaf_bricks[leaf_index];
    if (!svlm_leaf_visible_for_debug_level(leaf.level)) {
        return;
    }

    if (arrayLength(&debug_leaf_indices) <= 1u) {
        return;
    }

    let selected_slot = atomicAdd(&debug_leaf_indices[0], 1u);
    if (selected_slot + 1u >= arrayLength(&debug_leaf_indices)) {
        return;
    }

    atomicStore(&debug_leaf_indices[selected_slot + 1u], leaf_index);
}
