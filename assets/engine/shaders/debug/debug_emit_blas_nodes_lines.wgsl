#include "common.wgsl"
#include "acceleration_common.wgsl"

const LINES_PER_BOX = 12u;
const BVH_COLOR = vec4<f32>(0.0, 0.8, 1.0, 1.0);
const MAX_STACK = 256u;
const EPS = 0.0001;
const MAX_NODES_DEBUG = 8096u;

const EDGES: array<vec2<u32>, 12> = array<vec2<u32>, 12>(
    vec2u(0, 1), vec2u(1, 3), vec2u(3, 2), vec2u(2, 0),
    vec2u(4, 5), vec2u(5, 7), vec2u(7, 6), vec2u(6, 4),
    vec2u(0, 4), vec2u(1, 5), vec2u(2, 6), vec2u(3, 7),
);

struct LineData {
    color_and_width: vec4<f32>,
    transform: mat4x4<f32>,
};

fn create_line_transform(start: vec3<f32>, end: vec3<f32>) -> mat4x4<f32> {
    let dir = end - start;
    let len = length(dir);
    let is_short = len < 0.0001;
    let safe_len = select(len, 1.0, is_short);

    let n = dir / safe_len;
    var up = vec3<f32>(0.0, 1.0, 0.0);
    let near_up = abs(dot(n, up)) > 0.99;
    up = select(up, vec3<f32>(0.0, 0.0, 1.0), near_up);

    let right = normalize(cross(n, up));
    let true_up = normalize(cross(right, n));

    let rot = mat4x4<f32>(
        vec4<f32>(n, 0.0),
        vec4<f32>(true_up, 0.0),
        vec4<f32>(right, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );

    let trans = mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(start, 1.0),
    );

    let scale = mat4_from_scaling(vec3<f32>(len, 1.0, 1.0));
    return trans * rot * scale;
}

fn corner(min_p: vec3<f32>, max_p: vec3<f32>, idx: u32) -> vec3<f32> {
    let x_sel = select(min_p.x, max_p.x, (idx & 1u) != 0u);
    let y_sel = select(min_p.y, max_p.y, (idx & 2u) != 0u);
    let z_sel = select(min_p.z, max_p.z, (idx & 4u) != 0u);
    return vec3<f32>(x_sel, y_sel, z_sel);
}

@group(1) @binding(0) var<storage, read_write> out_line_data: array<LineData>;
@group(1) @binding(1) var<storage, read> blas_directory: array<MeshDirectoryEntry>;
@group(1) @binding(2) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(3) var<storage, read> closest_entities_per_mesh: array<u32>;
@group(1) @binding(4) var<storage, read> bvh2_nodes: array<AABB>; // Direct BVH2 buffer (not in atlas)

@compute @workgroup_size(128, 2)
fn cs(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let mesh_asset_id = gid.y;
    let node_idx = gid.x;
    
    // ============================================================================
    // BOUNDS CHECKING - Early exit for invalid threads
    // ============================================================================
    
    // Check if this mesh has a closest entity assigned
    let entity_resolved = closest_entities_per_mesh[mesh_asset_id];
    if (entity_resolved == 0u) { return; } // No entity assigned for this mesh
    
    let mesh_directory_entry = blas_directory[mesh_asset_id];
    
    // Check if this node index is valid for this mesh
    if (node_idx >= u32(mesh_directory_entry.leaf_count)) { return; }
    
    // ============================================================================
    // UNIQUE LINEAR INDEX CALCULATION - Create unique index per (mesh, node)
    // ============================================================================
    
    // Calculate a unique linear index for this (mesh_asset_id, node_idx) pair
    // We need to ensure this doesn't exceed the allocated line buffer size
    let max_nodes_per_mesh = (MAX_NODES_DEBUG + 15u) / 16u * 16u; // Round up to workgroup size
    let unique_linear_index = mesh_asset_id * max_nodes_per_mesh + node_idx;
    
    // ============================================================================
    // MESH DATA RETRIEVAL & TRANSFORMATION
    // ============================================================================
    
    let entity_transform = entity_transforms[entity_resolved].transform;
    let global_node_index = u32(mesh_directory_entry.bvh2_base) + node_idx;
    var node = bvh2_nodes[global_node_index];
    
    node.min -= vec4<f32>(EPS, EPS, EPS, 0.0);
    node.max += vec4<f32>(EPS, EPS, EPS, 0.0);
    
    // Transform BLAS node bounds to world space using entity transform
    let transformed_node = transform_aabb(node, entity_transform);
    let min_ws = transformed_node.min.xyz;
    let max_ws = transformed_node.max.xyz;
    
    // ============================================================================
    // VALIDITY CHECKS & LINE GENERATION
    // ============================================================================
    
    let has_volume = all(max_ws > min_ws);
    let is_valid_node = is_valid_node(node);
    let is_active = has_volume && is_valid_node;
    let width = select(0.0, 0.004, is_active);
    
    // Generate lines for this box using the unique linear index
    for (var e: u32 = 0u; e < LINES_PER_BOX; e = e + 1u) {
        let line_index = unique_linear_index * LINES_PER_BOX + e;
        
        let idx0 = EDGES[e].x;
        let idx1 = EDGES[e].y;
        let p0 = corner(min_ws, max_ws, idx0);
        let p1 = corner(min_ws, max_ws, idx1);

        var transform = create_line_transform(p0, p1);
        
        // Zero out transform for inactive nodes (makes them invisible)
        transform[0] = select(transform[0], identity_matrix[0], !is_active);
        transform[1] = select(transform[1], identity_matrix[1], !is_active);
        transform[2] = select(transform[2], identity_matrix[2], !is_active);
        transform[3] = select(transform[3], identity_matrix[3], !is_active);

        out_line_data[line_index].transform = transform;
        out_line_data[line_index].color_and_width = vec4<f32>(BVH_COLOR.rgb, width);
    }
}
