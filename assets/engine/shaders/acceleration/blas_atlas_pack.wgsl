#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "blas_common.wgsl"

@group(1) @binding(0) var<storage, read_write> blas_atlas: BLASAtlas;
@group(1) @binding(1) var<storage, read> src_bvh2: array<AABB>;
@group(1) @binding(2) var<storage, read> src_bvh4: array<BVH4Node>;
@group(1) @binding(3) var<storage, read> src_dir: array<MeshDirectoryEntry>;

@compute @workgroup_size(256)
fn pack_bvh2(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let total = u32(blas_atlas.header.bvh2_vec4_count) / 2;
    if (i >= total) { return; }
    let dst_base = u32(blas_atlas.header.bvh2_base_v4) + i * 2u;
    blas_atlas.data[dst_base + 0u] = src_bvh2[i].min;
    blas_atlas.data[dst_base + 1u] = src_bvh2[i].max;
}

@compute @workgroup_size(256)
fn pack_bvh4(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let total = u32(blas_atlas.header.bvh4_vec4_count) / 3;
    if (i >= total) { return; }
    let dst_base = u32(blas_atlas.header.bvh4_base_v4) + i * 3u;
    let node = src_bvh4[i];
    blas_atlas.data[dst_base + 0u] = node.min;
    blas_atlas.data[dst_base + 1u] = node.max;
    blas_atlas.data[dst_base + 2u] = node.children;
}

@compute @workgroup_size(256)
fn pack_directory(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let total = u32(blas_atlas.header.dir_vec4_count) / 2;
    if (i >= total) { return; }
    let e = src_dir[i];
    let v0 = vec4f(f32(e.bvh2_base), f32(e.bvh2_capacity), f32(e.bvh4_base), f32(e.bvh4_capacity));
    let v1 = vec4f(f32(e.leaf_count), f32(e.first_vertex), f32(e.first_index), f32(e.padding));
    let dst_base = u32(blas_atlas.header.dir_base_v4) + i * 2u;
    blas_atlas.data[dst_base + 0u] = v0;
    blas_atlas.data[dst_base + 1u] = v1;
}


