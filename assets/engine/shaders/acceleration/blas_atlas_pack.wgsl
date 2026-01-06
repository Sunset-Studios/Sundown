#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "blas_common.wgsl"

@group(1) @binding(0) var<storage, read_write> blas_atlas: BLASAtlas;
@group(1) @binding(1) var<storage, read> src_bvh8: array<BVH8Node>;
@group(1) @binding(2) var<storage, read> src_dir: array<MeshDirectoryEntry>;
@group(1) @binding(3) var<storage, read> index_buffer: array<u32>;

@compute @workgroup_size(256)
fn pack_bvh8(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let total = u32(blas_atlas.header.bvh8_vec4_count) / 12;  // 12 vec4s per node (4 node + 8 leaf data)
    if (i >= total) { return; }
    let dst_base = u32(blas_atlas.header.bvh8_base_v4) + i * 12u;
    let node = src_bvh8[i];
    
    // Write node data (first 4 vec4s)
    blas_atlas.data[dst_base + 0u] = node.min;
    blas_atlas.data[dst_base + 1u] = node.max;
    blas_atlas.data[dst_base + 2u] = node.children0;
    blas_atlas.data[dst_base + 3u] = node.children1;
    
    // Find which mesh this node belongs to by searching the directory
    let dir_entry_count = u32(blas_atlas.header.dir_vec4_count) / 2u;
    var first_index = 0u;
    var first_vertex = 0u;
    var found = false;
    
    for (var mesh_id = 0u; mesh_id < dir_entry_count; mesh_id = mesh_id + 1u) {
        let entry = src_dir[mesh_id];
        let bvh8_start = entry.bvh8_base;
        let bvh8_end = entry.bvh8_base + entry.bvh8_capacity;
        
        if (i >= bvh8_start && i < bvh8_end) {
            first_index = entry.first_index;
            first_vertex = entry.first_vertex;
            found = true;
            break;
        }
    }
    
    // Generate co-located leaf data (8 vec4s) by resolving triangle vertex indices
    let leaf_mask = bitcast<u32>(node.min.w);
    
    for (var child = 0u; child < 8u; child = child + 1u) {
        var leaf_indices = vec4<u32>(0u, 0u, 0u, 0u);
        
        // Check if this child is a leaf and valid
        if (found && bvh8_child(node, child) >= 0.0 && ((leaf_mask >> child) & 1u) != 0u) {
            let tri_id = u32(bvh8_child(node, child));
            
            // Load vertex indices from the mesh's index buffer region
            leaf_indices.x = first_vertex + index_buffer[first_index + tri_id * 3u + 0u];
            leaf_indices.y = first_vertex + index_buffer[first_index + tri_id * 3u + 1u];
            leaf_indices.z = first_vertex + index_buffer[first_index + tri_id * 3u + 2u];
        }
        
        // Write leaf data co-located with node
        blas_atlas.data[dst_base + 4u + child] = bitcast<vec4<f32>>(leaf_indices);
    }
}

@compute @workgroup_size(256)
fn pack_directory(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let total = u32(blas_atlas.header.dir_vec4_count) / 2;
    if (i >= total) { return; }
    let e = src_dir[i];
    let v0 = vec4f(f32(e.bvh2_base), f32(e.bvh2_capacity), f32(e.bvh8_base), f32(e.bvh8_capacity));
    let v1 = vec4f(f32(e.leaf_count), f32(e.first_vertex), f32(e.first_index), f32(e.padding));
    let dst_base = u32(blas_atlas.header.dir_base_v4) + i * 2u;
    blas_atlas.data[dst_base + 0u] = v0;
    blas_atlas.data[dst_base + 1u] = v1;
}

