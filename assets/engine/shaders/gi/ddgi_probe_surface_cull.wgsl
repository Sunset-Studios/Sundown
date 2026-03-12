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
#include "gi/ddgi_common.wgsl"

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read_write> probe_states: array<ProbeStateData>;
@group(1) @binding(2) var<storage, read> probe_cull_flags: array<u32>;
@group(1) @binding(3) var<storage, read> tlas_bvh2_bounds: array<AABB>;
@group(1) @binding(4) var<uniform> tlas_bvh_info: BVHInfo;
@group(1) @binding(5) var<storage, read> blas_bvh2_nodes: array<AABB>;
@group(1) @binding(6) var<storage, read> blas_directory: array<MeshDirectoryEntry>;
@group(1) @binding(7) var<storage, read> entity_transforms: array<EntityTransform>;

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
    let probe_aabb = AABB(vec4<f32>(probe_min, 0.0), vec4<f32>(probe_max, 0.0));
    let inverse_model = transpose(entity_transform.transpose_inverse_model_matrix);
    return transform_aabb(probe_aabb, inverse_model);
}

fn probe_overlaps_blas(
    probe_min_local: vec3<f32>,
    probe_max_local: vec3<f32>,
    mesh_asset_id: u32
) -> bool {
    let mesh_directory_entry = blas_directory[mesh_asset_id];
    let leaf_count = mesh_directory_entry.leaf_count;
    if (leaf_count == 0u) {
        return false;
    }
    let bvh2_node_count = 2u * leaf_count - 1u;
    let root_node_idx = mesh_directory_entry.bvh2_base + bvh2_node_count - 1u;

    var node_stack: array<u32, NODE_STACK_SIZE>;
    node_stack[0] = root_node_idx;
    var stack_size = 1u;

    while (stack_size > 0u) {
        stack_size = stack_size - 1u;

        let node_idx = node_stack[stack_size];
        if (node_idx == INVALID_IDX) { continue; }

        let node = blas_bvh2_nodes[node_idx];
        if (!aabb_overlaps(probe_min_local, probe_max_local, node.min.xyz, node.max.xyz)) {
            continue;
        }

        if (is_leaf(node)) {
            return true;
        } else {
            let left_idx = u32(node.min.w);
            if (left_idx != node_idx
                && left_idx != INVALID_IDX
                && aabb_overlaps(probe_min_local, probe_max_local, blas_bvh2_nodes[left_idx].min.xyz, blas_bvh2_nodes[left_idx].max.xyz)
                ) {
                node_stack[stack_size] = left_idx;
                stack_size = stack_size + 1u;
            }

            let right_idx = u32(node.max.w);
            if (right_idx != node_idx
                && right_idx != INVALID_IDX
                && aabb_overlaps(probe_min_local, probe_max_local, blas_bvh2_nodes[right_idx].min.xyz, blas_bvh2_nodes[right_idx].max.xyz)
                ) {
                node_stack[stack_size] = right_idx;
                stack_size = stack_size + 1u;
            }
        }
    }

    return false;
}

fn probe_overlaps_scene(probe_min: vec3<f32>, probe_max: vec3<f32>) -> bool {
    if (tlas_bvh_info.bvh2_count == 0u) {
        return false;
    }

    var node_stack: array<u32, NODE_STACK_SIZE>;
    node_stack[0] = tlas_bvh_info.bvh2_count - 1u;
    var stack_size = 1u;

    while (stack_size > 0u) {
        stack_size = stack_size - 1u;

        let node_idx = node_stack[stack_size];
        if (node_idx == INVALID_IDX) { continue; }

        let node = tlas_bvh2_bounds[node_idx];
        if (!aabb_overlaps(probe_min, probe_max, node.min.xyz, node.max.xyz)) {
            continue;
        }

        if (is_leaf(node)) {
            let mesh_id = u32(node.min.w);
            let prim_store = u32(-node.max.w - 1.0);
            if (mesh_id == INVALID_IDX) { continue; }
            let entity_transform = entity_transforms[prim_store];

            let probe_aabb_local = probe_local_aabb(probe_min, probe_max, entity_transform);
            if (probe_overlaps_blas(probe_aabb_local.min.xyz, probe_aabb_local.max.xyz, mesh_id)) {
                return true;
            }
        } else {
            let left_idx = u32(node.min.w);
            if (left_idx != node_idx
                && left_idx != INVALID_IDX
                && aabb_overlaps(probe_min, probe_max, tlas_bvh2_bounds[left_idx].min.xyz, tlas_bvh2_bounds[left_idx].max.xyz)
                ) {
                node_stack[stack_size] = left_idx;
                stack_size = stack_size + 1u;
            }

            let right_idx = u32(node.max.w);
            if (right_idx != node_idx
                && right_idx != INVALID_IDX
                && aabb_overlaps(probe_min, probe_max, tlas_bvh2_bounds[right_idx].min.xyz, tlas_bvh2_bounds[right_idx].max.xyz)
                ) {
                node_stack[stack_size] = right_idx;
                stack_size = stack_size + 1u;
            }
        }
    }

    return false;
}

// =============================================================================
// MAIN COMPUTE SHADER
// =============================================================================

@compute @workgroup_size(256, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let probe_count = u32(ddgi_params.probe_counts.x);
    let probe_index = gid.x;

    if (probe_index >= probe_count) {
        return;
    }

    let packed_state = probe_states[probe_index].packed_state;
    let cull_word = probe_cull_flags[probe_index / 32u];
    let is_cull_visible = ((cull_word >> (probe_index % 32u)) & 1u) != 0u;
    if (!is_cull_visible) {
        return;
    }

    let probe_pos = ddgi_probe_world_position_from_index(&ddgi_params, probe_index);
    let spacing = ddgi_probe_spacing_from_index(&ddgi_params, probe_index);
    let state = probe_state_get_state(packed_state);
    let flags = probe_state_get_flags(packed_state);

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
