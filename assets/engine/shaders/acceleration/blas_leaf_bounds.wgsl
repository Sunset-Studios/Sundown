#include "common.wgsl"
#include "acceleration_common.wgsl"

struct MeshSelector { mesh_id: u32 };

@group(1) @binding(0) var<storage, read_write> out_bounds: array<AABB>;
@group(1) @binding(1) var<storage, read> blas_directory: array<MeshDirectoryEntry>;
@group(1) @binding(2) var<storage, read> mesh_selector: MeshSelector;
@group(1) @binding(3) var<storage, read> index_buffer: array<u32>;

fn load_position(vertex_index: u32) -> vec3<f32> {
    return vertex_buffer[vertex_index].position.xyz;
}

@compute @workgroup_size(64)
fn write_leaf_bounds(@builtin(global_invocation_id) gid: vec3u) {
    let tri_id = gid.x;
    let mesh_id = mesh_selector.mesh_id;
    let entry = blas_directory[mesh_id];
    let total = entry.leaf_count;
    if (tri_id >= total) { return; }

    let first_vertex = entry.first_vertex;
    let first_index = entry.first_index;

    // Load triangle indices from packed 16-bit buffer
    let i0 = index_buffer[first_index + tri_id * 3u + 0u];
    let i1 = index_buffer[first_index + tri_id * 3u + 1u];
    let i2 = index_buffer[first_index + tri_id * 3u + 2u];

    let v0 = load_position(first_vertex + i0);
    let v1 = load_position(first_vertex + i1);
    let v2 = load_position(first_vertex + i2);

    let mn = min(v0, min(v1, v2));
    let mx = max(v0, max(v1, v2));

    let write_index = entry.bvh2_base + tri_id;
    out_bounds[write_index].min = vec4<f32>(mn, f32(tri_id));
    out_bounds[write_index].max = vec4<f32>(mx, -1.0);
}


