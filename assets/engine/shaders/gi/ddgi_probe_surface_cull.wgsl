// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                   DDGI PROBE BVH PRE-CULL (GEOMETRY)                     ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Uses the TLAS + BLAS BVH hierarchy to determine if each probe cell        ║
// ║  overlaps any nearby geometry. Probes with no overlap are immediately      ║
// ║  marked as OFF to avoid unnecessary tracing work.                          ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "blas_common.wgsl"
#include "gi/ddgi_common.wgsl"

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read_write> probe_states: array<ProbeStateData>;
@group(1) @binding(2) var<storage, read> tlas_bvh2_bounds: array<AABB>;
@group(1) @binding(3) var<storage, read> tlas_bvh8_nodes: array<BVH8Node>;
@group(1) @binding(4) var<storage, read> blas_atlas: BLASAtlas;
@group(1) @binding(5) var<storage, read> blas_bvh2_nodes: array<AABB>;
@group(1) @binding(6) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(7) var<storage, read> mesh_asset_ids: array<u32>;

// =============================================================================
// HELPERS
// =============================================================================

fn aabb_overlaps(min_a: vec3<f32>, max_a: vec3<f32>, min_b: vec3<f32>, max_b: vec3<f32>) -> bool {
    return all(min_a <= max_b) && all(max_a >= min_b);
}

fn probe_local_aabb(
    probe_min: vec3<f32>,
    probe_max: vec3<f32>,
    entity_transform: EntityTransform
) -> AABB {
    let probe_aabb = AABB(vec4f(probe_min, 0.0), vec4f(probe_max, 0.0));
    let inverse_model = transpose(entity_transform.transpose_inverse_model_matrix);
    return transform_aabb(probe_aabb, inverse_model);
}

fn probe_overlaps_blas(
    probe_min_local: vec3<f32>,
    probe_max_local: vec3<f32>,
    mesh_asset_id: u32
) -> bool {
    let mesh_directory_entry = atlas_load_directory_entry(mesh_asset_id);
    let bvh8_base = mesh_directory_entry.bvh8_base;

    var node_stack: array<u32, NODE_STACK_SIZE>;
    node_stack[0] = bvh8_base;
    var stack_size = 1u;

    while (stack_size > 0u) {
        stack_size = stack_size - 1u;

        let node_idx = node_stack[stack_size];
        if (node_idx == INVALID_IDX) { continue; }

        let leaf_mask = atlas_load_bvh8_leaf_mask(node_idx);

        for (var i = 0u; i < 8u; i = i + 1u) {
            let child_raw = atlas_load_bvh8_child(node_idx, i);
            if (child_raw < 0.0) { continue; }

            let child_idx = u32(child_raw);

            if (((leaf_mask >> i) & 1u) != 0u) {
                // Leaf node: triangle intersection
                let leaf_indices = atlas_load_bvh8_leaf_indices(node_idx, i);
                let min_point = min(
                    vertex_buffer[leaf_indices.x].position.xyz,
                    min(vertex_buffer[leaf_indices.y].position.xyz, vertex_buffer[leaf_indices.z].position.xyz)
                ) - vec3<f32>(0.001);
                let max_point = max(
                    vertex_buffer[leaf_indices.x].position.xyz,
                    max(vertex_buffer[leaf_indices.y].position.xyz, vertex_buffer[leaf_indices.z].position.xyz)
                ) + vec3<f32>(0.001);
                if (aabb_overlaps(probe_min_local, probe_max_local, min_point, max_point)) {
                    return true;
                }
            } else {
                // Internal node: AABB test before push
                if (aabb_overlaps(probe_min_local, probe_max_local, atlas_load_bvh8_node_min(child_idx), atlas_load_bvh8_node_max(child_idx))) {
                    if (child_idx == node_idx) { continue; }
                    node_stack[stack_size] = child_idx;
                    stack_size = stack_size + 1u;
                }
            }
        }
    }

    return false;
}

fn probe_overlaps_scene(probe_min: vec3<f32>, probe_max: vec3<f32>) -> bool {
    var node_stack: array<u32, NODE_STACK_SIZE>;
    node_stack[0] = 0u;
    var stack_size = 1u;

    while (stack_size > 0u) {
        stack_size = stack_size - 1u;

        let node_idx = node_stack[stack_size];
        if (node_idx == INVALID_IDX) { continue; }

        var node = tlas_bvh8_nodes[node_idx];
        let leaf_mask = bitcast<u32>(node.min.w);

        for (var i = 0u; i < 8u; i = i + 1u) {
            let child_raw = bvh8_child(&node, i);
            if (child_raw < 0.0) { continue; }

            let child_idx = u32(child_raw);

            if (((leaf_mask >> i) & 1u) != 0u) {
                let leaf_bounds = tlas_bvh2_bounds[child_idx];
                let prim_store = u32(leaf_bounds.min.w);
                let mesh_id = mesh_asset_ids[prim_store];
                let entity_transform = entity_transforms[prim_store];

                let probe_aabb_local = probe_local_aabb(probe_min, probe_max, entity_transform);
                if (probe_overlaps_blas(probe_aabb_local.min.xyz, probe_aabb_local.max.xyz, mesh_id)) {
                    return true;
                }
            } else {
                let child_min = tlas_bvh8_nodes[child_idx].min.xyz;
                let child_max = tlas_bvh8_nodes[child_idx].max.xyz;
                if (aabb_overlaps(probe_min, probe_max, child_min, child_max)) {
                    if (child_idx == node_idx) { continue; }
                    node_stack[stack_size] = child_idx;
                    stack_size = stack_size + 1u;
                }
            }
        }
    }

    return false;
}

// =============================================================================
// MAIN COMPUTE SHADER
// =============================================================================

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let probe_count = u32(ddgi_params.probe_counts.x);
    let probe_index = gid.x;

    if (probe_index >= probe_count) {
        return;
    }

    let probe_pos = ddgi_probe_world_position_from_index_with_offset(&ddgi_params, &probe_states, probe_index);
    let spacing = ddgi_probe_spacing_from_index(&ddgi_params, probe_index);
    let state = probe_state_get_state(probe_states[probe_index].packed_state);
    let flags = probe_state_get_flags(probe_states[probe_index].packed_state);

    let probe_min = probe_pos - vec3<f32>(spacing);
    let probe_max = probe_pos + vec3<f32>(spacing);
    let true_probe_min = min(probe_min, probe_max);
    let true_probe_max = max(probe_min, probe_max);

    let overlaps_scene = probe_overlaps_scene(true_probe_min, true_probe_max);
    let cascade_index = ddgi_probe_cascade_index(&ddgi_params, probe_index);
    let snap_active = ddgi_params.cascades[cascade_index].snap_delta.w > 0.0;
    if (!overlaps_scene && state != PROBE_STATE_OFF && (state != PROBE_STATE_UNINITIALIZED || snap_active)) {
        probe_states[probe_index].packed_state = probe_state_pack(PROBE_STATE_SLEEPING, 0u, 0u, flags);
    } else if (overlaps_scene && state == PROBE_STATE_SLEEPING) {
        probe_states[probe_index].packed_state = probe_state_pack(PROBE_STATE_UNINITIALIZED, 0u, 0u, flags);
    }
}
