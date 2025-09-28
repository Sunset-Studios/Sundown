// ------------------------------------------------------------------------------------
// BLAS Atlas (single storage buffer view)
// - Packs BLAS BVH2 AABBs, BVH4 nodes, and MeshDirectory entries into one buffer
// - Offsets are expressed in vec4 units relative to the start of the runtime array
// ------------------------------------------------------------------------------------
struct BLASAtlasHeader {
    bvh2_base_v4: u32,     // start of BVH2 AABB data (in vec4 units)
    bvh2_vec4_count: u32,  // total vec4s used by BVH2 section
    bvh4_base_v4: u32,     // start of BVH4 node data (in vec4 units)
    bvh4_vec4_count: u32,  // total vec4s used by BVH4 section
    dir_base_v4: u32,      // start of directory entries (in vec4 units)
    dir_vec4_count: u32,   // total vec4s used by directory section
    _pad0: u32,
    _pad1: u32,
};

struct BLASAtlas {
    header: BLASAtlasHeader,
    data: array<vec4<f32>>,
};

// ------------------------------------------------------------------------------------
// Functions
// ------------------------------------------------------------------------------------

// Helpers to read structures from the atlas
fn atlas_load_aabb(idx: u32) -> AABB {
    let base = blas_atlas.header.bvh2_base_v4 + idx * 2u;
    let v0 = blas_atlas.data[base + 0u];
    let v1 = blas_atlas.data[base + 1u];
    return AABB(v0, v1);
}

fn atlas_load_bvh4_node(idx: u32) -> BVH4Node {
    let base = blas_atlas.header.bvh4_base_v4 + idx * 3u;
    let mn = blas_atlas.data[base + 0u];
    let mx = blas_atlas.data[base + 1u];
    let ch = blas_atlas.data[base + 2u];
    return BVH4Node(mn, mx, ch);
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
