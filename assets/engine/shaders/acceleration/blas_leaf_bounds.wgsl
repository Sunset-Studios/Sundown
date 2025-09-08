#include "common.wgsl"
#include "acceleration_common.wgsl"

struct MeshDirectoryEntry {
    base: u32,
    capacity: u32,
    leaf_count: u32,
};

struct LeafUniforms {
    mesh_id: u32,
    base_node: u32,
    first_vertex: u32,
    triangle_count: u32,
};

@group(1) @binding(0) var<storage, read_write> out_bounds: array<AABB>;
@group(1) @binding(1) var<storage, read> blas_directory: array<MeshDirectoryEntry>;
@group(1) @binding(2) var<uniform> leaf_uniforms: LeafUniforms;
@group(1) @binding(3) var<storage, read> index_buffer: array<u32>;

fn load_position(vertex_index: u32) -> vec3f {
    return vertex_buffer[vertex_index].position.xyz;
}

@compute @workgroup_size(64)
fn write_leaf_bounds(@builtin(global_invocation_id) gid: vec3u) {
    let tri_id = gid.x;
    let total = leaf_uniforms.triangle_count;
    if (tri_id >= total) { return; }

    let first_vertex = leaf_uniforms.first_vertex;
    let base_node = leaf_uniforms.base_node;

    let i0 = index_buffer[tri_id * 3u + 0u];
    let i1 = index_buffer[tri_id * 3u + 1u];
    let i2 = index_buffer[tri_id * 3u + 2u];

    let v0 = load_position(first_vertex + i0);
    let v1 = load_position(first_vertex + i1);
    let v2 = load_position(first_vertex + i2);

    var mn = min(v0, min(v1, v2));
    var mx = max(v0, max(v1, v2));

    // Guard against degenerate triangles by nudging bounds minimally
    let eps = 1e-6;
    let extent = mx - mn;
    let is_degenerate = any(extent <= vec3f(eps));
    let delta = select(vec3f(0.0), vec3f(eps, eps, eps), is_degenerate);
    mn = mn - delta;
    mx = mx + delta;

    let write_index = base_node + tri_id;
    out_bounds[write_index].min = vec4f(mn, f32(tri_id));
    out_bounds[write_index].max = vec4f(mx, -1.0);
}


