// ------------------------------------------------------------------------------------
// BLAS Atlas (single storage buffer view)
// - Packs BLAS BVH2 AABBs, BVH4 nodes (with co-located leaf data), and MeshDirectory entries
// - BVH4 nodes are now 7 vec4s: [min, max, children, leaf0_indices, leaf1_indices, leaf2_indices, leaf3_indices]
// - Offsets are expressed in vec4 units relative to the start of the runtime array
// ------------------------------------------------------------------------------------
struct BLASAtlasHeader {
    bvh4_base_v4: u32,     // start of BVH4 node data (in vec4 units)
    bvh4_vec4_count: u32,  // total vec4s used by BVH4 section
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
fn atlas_load_bvh4_node(idx: u32) -> BVH4Node {
    let base = blas_atlas.header.bvh4_base_v4 + idx * 7u;  // Now 7 vec4s per node
    let mn = blas_atlas.data[base + 0u];
    let mx = blas_atlas.data[base + 1u];
    let ch = blas_atlas.data[base + 2u];
    return BVH4Node(mn, mx, ch);
}

fn atlas_load_bvh4_leaf_mask(idx: u32) -> u32 {
    let base = blas_atlas.header.bvh4_base_v4 + idx * 7u;  // Now 7 vec4s per node
    return bitcast<u32>(blas_atlas.data[base + 0u].w);
}

fn atlas_load_bvh4_node_children(idx: u32) -> vec4<f32> {
    let base = blas_atlas.header.bvh4_base_v4 + idx * 7u;  // Now 7 vec4s per node
    return blas_atlas.data[base + 2u];
}

fn atlas_load_bvh4_node_min(idx: u32) -> vec3<f32> {
    let base = blas_atlas.header.bvh4_base_v4 + idx * 7u;  // Now 7 vec4s per node
    return blas_atlas.data[base + 0u].xyz;
}

fn atlas_load_bvh4_node_max(idx: u32) -> vec3<f32> {
    let base = blas_atlas.header.bvh4_base_v4 + idx * 7u;  // Now 7 vec4s per node
    return blas_atlas.data[base + 1u].xyz;
}

fn atlas_load_bvh4_leaf_indices(node_idx: u32, child_slot: u32) -> vec4<u32> {
    let base = blas_atlas.header.bvh4_base_v4 + node_idx * 7u;
    return bitcast<vec4<u32>>(blas_atlas.data[base + 3u + child_slot]);
}

fn atlas_load_directory_entry(idx: u32) -> MeshDirectoryEntry {
    let base = blas_atlas.header.dir_base_v4 + idx * 2u;
    let u0 = blas_atlas.data[base + 0u];
    let u1 = blas_atlas.data[base + 1u];
    var entry: MeshDirectoryEntry;
    entry.bvh2_base = u32(u0.x);
    entry.bvh2_capacity = u32(u0.y);
    entry.bvh4_base = u32(u0.z);
    entry.bvh4_capacity = u32(u0.w);
    entry.leaf_count = u32(u1.x);
    entry.first_vertex = u32(u1.y);
    entry.first_index = u32(u1.z);
    entry.padding = u32(u1.w);
    return entry;
}

fn atlas_load_directory_entry_bvh4_base(idx: u32) -> u32 {
    let base = blas_atlas.header.dir_base_v4 + idx * 2u;
    return u32(blas_atlas.data[base + 0u].z);
}

fn atlas_load_directory_entry_bvh4_capacity(idx: u32) -> u32 {
    let base = blas_atlas.header.dir_base_v4 + idx * 2u;
    return u32(blas_atlas.data[base + 0u].w);
}

fn atlas_load_directory_entry_leaf_count(idx: u32) -> u32 {
    let base = blas_atlas.header.dir_base_v4 + idx * 2u;
    return u32(blas_atlas.data[base + 1u].x);
}

fn atlas_load_directory_entry_first_vertex(idx: u32) -> u32 {
    let base = blas_atlas.header.dir_base_v4 + idx * 2u;
    return u32(blas_atlas.data[base + 1u].y);
}

fn atlas_load_directory_entry_first_index(idx: u32) -> u32 {
    let base = blas_atlas.header.dir_base_v4 + idx * 2u;
    return u32(blas_atlas.data[base + 1u].z);
}
