#include "gi/svlm_common.wgsl"

// Root seeding pass.
//
// The begin pass has already chosen the root grid dimensions. This pass maps
// each linear root slot to a 3D root coordinate, writes the first level of the
// node pool, and fills the current frontier queue used by classification.

@group(1) @binding(0) var<storage, read_write> svlm_params: SVLMParams;
@group(1) @binding(1) var<storage, read_write> svlm_counters: SVLMCounters;
@group(1) @binding(2) var<storage, read_write> node_pool: array<SVLMNode>;
@group(1) @binding(3) var<storage, read_write> curr_nodes: array<u32>;

fn svlm_write_node(index: u32, level: u32, coord: vec3<u32>, parent: u32, flags: u32) {
    node_pool[index].morton = svlm_morton3(coord);
    node_pool[index].level = level;
    node_pool[index].flags = flags;
    node_pool[index].child_base = INVALID_IDX;
    node_pool[index].leaf_index = INVALID_IDX;
    node_pool[index].parent_index = parent;
    node_pool[index].coord_x = coord.x;
    node_pool[index].coord_y = coord.y;
    node_pool[index].coord_z = coord.z;
    node_pool[index].score = 0.0;
    node_pool[index].reserved_10 = 0u;
    node_pool[index].reserved_11 = 0u;
    node_pool[index].reserved_12 = 0u;
    node_pool[index].reserved_13 = 0u;
    node_pool[index].reserved_14 = 0u;
    node_pool[index].reserved_15 = 0u;
}

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let root_index = gid.x;
    let root_count = atomicLoad(&svlm_counters.curr_count);
    if (root_index >= root_count) {
        return;
    }

    let dims = svlm_root_dims(&svlm_params);
    let xy = max(1u, dims.x * dims.y);
    // Row-major root indexing is only for dispatch/queue construction. The
    // Morton key written into the node record is the stable spatial key.
    let coord = vec3<u32>(
        root_index % max(1u, dims.x),
        (root_index / max(1u, dims.x)) % max(1u, dims.y),
        root_index / xy
    );

    svlm_write_node(root_index, 0u, coord, INVALID_IDX, SVLM_FLAG_ACTIVE);
    curr_nodes[root_index] = root_index;
}
