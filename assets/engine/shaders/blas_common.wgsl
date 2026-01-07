// ------------------------------------------------------------------------------------
// BLAS Atlas (single storage buffer view)
// - Packs BLAS BVH2 AABBs, BVH8 nodes (with co-located leaf data), and MeshDirectory entries
// - BVH8 nodes are now 12 vec4s: [min, max, children0, children1, leaf0_indices...leaf7_indices]
// - Offsets are expressed in vec4 units relative to the start of the runtime array
// ------------------------------------------------------------------------------------
struct BLASAtlasHeader {
    bvh8_base_v4: u32,     // start of BVH8 node data (in vec4 units)
    bvh8_vec4_count: u32,  // total vec4s used by BVH8 section
    dir_base_v4: u32,      // start of directory entries (in vec4 units)
    dir_vec4_count: u32,   // total vec4s used by directory section
};

struct BLASAtlas {
    header: BLASAtlasHeader,
    data: array<vec4<f32>>,
};

// ------------------------------------------------------------------------------------
// Functions
// ------------------------------------------------------------------------------------
fn atlas_load_bvh8_node(idx: u32) -> BVH8Node {
    let base = blas_atlas.header.bvh8_base_v4 + idx * 12u;  // 12 vec4s per node
    let mn = blas_atlas.data[base + 0u];
    let mx = blas_atlas.data[base + 1u];
    let ch0 = blas_atlas.data[base + 2u];
    let ch1 = blas_atlas.data[base + 3u];
    return BVH8Node(mn, mx, ch0, ch1);
}

fn atlas_load_bvh8_leaf_mask(idx: u32) -> u32 {
    return bitcast<u32>(blas_atlas.data[blas_atlas.header.bvh8_base_v4 + idx * 12u + 0u].w);
}

fn atlas_load_bvh8_child(idx: u32, slot: u32) -> f32 {
    let base = blas_atlas.header.bvh8_base_v4 + idx * 12u;
    if (slot < 4u) {
        return blas_atlas.data[base + 2u][slot];
    }
    return blas_atlas.data[base + 3u][slot - 4u];
}

fn atlas_load_bvh8_node_min(idx: u32) -> vec3<f32> {
    return blas_atlas.data[blas_atlas.header.bvh8_base_v4 + idx * 12u + 0u].xyz;
}

fn atlas_load_bvh8_node_max(idx: u32) -> vec3<f32> {
    return blas_atlas.data[blas_atlas.header.bvh8_base_v4 + idx * 12u + 1u].xyz;
}

fn atlas_load_bvh8_leaf_indices(node_idx: u32, child_slot: u32) -> vec4<u32> {
    return bitcast<vec4<u32>>(blas_atlas.data[blas_atlas.header.bvh8_base_v4 + node_idx * 12u + 4u + child_slot]);
}

fn atlas_load_directory_entry(idx: u32) -> MeshDirectoryEntry {
    let base = blas_atlas.header.dir_base_v4 + idx * 2u;
    let u0 = blas_atlas.data[base + 0u];
    let u1 = blas_atlas.data[base + 1u];
    var entry: MeshDirectoryEntry;
    entry.bvh2_base = u32(u0.x);
    entry.bvh2_capacity = u32(u0.y);
    entry.bvh8_base = u32(u0.z);
    entry.bvh8_capacity = u32(u0.w);
    entry.leaf_count = u32(u1.x);
    entry.first_vertex = u32(u1.y);
    entry.first_index = u32(u1.z);
    entry.padding = u32(u1.w);
    return entry;
}

fn atlas_load_directory_entry_bvh8_base(idx: u32) -> u32 {
    return u32(blas_atlas.data[blas_atlas.header.dir_base_v4 + idx * 2u].z);
}

fn atlas_load_directory_entry_bvh8_capacity(idx: u32) -> u32 {
    return u32(blas_atlas.data[blas_atlas.header.dir_base_v4 + idx * 2u].w);
}

fn atlas_load_directory_entry_leaf_count(idx: u32) -> u32 {
    return u32(blas_atlas.data[blas_atlas.header.dir_base_v4 + idx * 2u + 1u].x);
}

fn atlas_load_directory_entry_first_vertex(idx: u32) -> u32 {
    return u32(blas_atlas.data[blas_atlas.header.dir_base_v4 + idx * 2u + 1u].y);
}

fn atlas_load_directory_entry_first_index(idx: u32) -> u32 {
    return u32(blas_atlas.data[blas_atlas.header.dir_base_v4 + idx * 2u + 1u].z);
}
