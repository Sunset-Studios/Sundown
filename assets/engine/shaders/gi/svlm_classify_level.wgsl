#include "gi/svlm_common.wgsl"

// Breadth-first brick classification.
//
// Every invocation owns one node from the current frontier. It queries nearby
// geometry through the TLAS, refines each hit through that entity's BLAS, and
// either emits a leaf brick or atomically appends eight child nodes to the next
// frontier. Empty regions stay as coarse leaves so the bake volume has complete
// brick/probe coverage instead of near-geometry-only holes.

@group(1) @binding(0) var<storage, read_write> svlm_params: SVLMParams;
@group(1) @binding(1) var<storage, read_write> svlm_counters: SVLMCounters;
@group(1) @binding(2) var<uniform> tlas_bvh_info: BVHInfo;
@group(1) @binding(3) var<storage, read> tlas_nodes: array<AABB>;
@group(1) @binding(4) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(5) var<storage, read> entity_index_lookup: array<u32>;
@group(1) @binding(6) var<storage, read> blas_directory: array<MeshDirectoryEntry>;
@group(1) @binding(7) var<storage, read> blas_nodes: array<AABB>;
@group(1) @binding(8) var<storage, read_write> node_pool: array<SVLMNode>;
@group(1) @binding(9) var<storage, read> curr_nodes: array<u32>;
@group(1) @binding(10) var<storage, read_write> next_nodes: array<u32>;
@group(1) @binding(11) var<storage, read_write> leaf_bricks: array<SVLMLeafBrick>;

const SVLM_QUERY_STACK_SIZE = 128u;
const SVLM_MAX_REFINEMENT_HITS = 96u;
const SVLM_MAX_OVERLAP_HITS = 32u;
const SVLM_MAX_BLAS_NODE_VISITS = 256u;

fn svlm_write_node(index: u32, level: u32, coord: vec3<u32>, parent: u32, flags: u32) {
    node_pool[index].morton = svlm_morton3(coord);
    node_pool[index].level = level;
    node_pool[index].flags = flags;
    node_pool[index].child_base = INVALID_IDX;
    node_pool[index].leaf_index = INVALID_IDX;
    node_pool[index].parent_index = parent;
    node_pool[index].coord_x = coord.x;
    node_pool[index].coord_y = coord.y;
    node_pool[index].coord_z = coord.z;
    node_pool[index].score = 0.0;
    node_pool[index].reserved_10 = 0u;
    node_pool[index].reserved_11 = 0u;
    node_pool[index].reserved_12 = 0u;
    node_pool[index].reserved_13 = 0u;
    node_pool[index].reserved_14 = 0u;
    node_pool[index].reserved_15 = 0u;
}

fn svlm_mesh_has_blas(mesh_id: u32) -> bool {
    if (mesh_id == INVALID_IDX || mesh_id >= arrayLength(&blas_directory)) {
        return false;
    }
    return blas_directory[mesh_id].leaf_count > 0u;
}

// Searches one mesh BLAS in local space for triangle-bound overlap. Refinement
// is intentionally gated on overlap, so this query avoids a broad near-distance
// walk that can touch huge parts of a dense mesh like Sponza.
fn svlm_blas_stats(local_brick: AABB, mesh_id: u32, local_keep_distance: f32, face_eps: f32) -> SVLMBlasStats {
    var result: SVLMBlasStats;
    result.overlap_count = 0u;
    result.near_count = 0u;
    result.stack_overflow = 0u;
    result.face_mask = 0u;
    result.min_distance = pos_inf;
    result.occupied_volume = 0.0;

    if (!svlm_mesh_has_blas(mesh_id)) {
        return result;
    }

    let directory = blas_directory[mesh_id];
    let leaf_count = directory.leaf_count;
    if (leaf_count == 0u) {
        return result;
    }

    let root_node = directory.bvh2_base + (leaf_count * 2u - 1u) - 1u;
    let keep_distance_sq = local_keep_distance * local_keep_distance;
    let root_distance_sq = svlm_aabb_distance_sq(local_brick, blas_nodes[root_node]);
    if (root_distance_sq > keep_distance_sq) {
        return result;
    }

    // Keep a cheap near signal for debug/stats, but do not enumerate all nearby
    // leaves. The expensive traversal below only follows overlapping nodes.
    result.near_count = 1u;
    result.min_distance = sqrt(root_distance_sq);

    var stack: array<u32, SVLM_QUERY_STACK_SIZE>;
    stack[0] = root_node;
    var stack_size = 1u;
    var node_visits = 0u;

    while (stack_size > 0u) {
        if (node_visits >= SVLM_MAX_BLAS_NODE_VISITS) {
            result.stack_overflow = 1u;
            return result;
        }
        node_visits = node_visits + 1u;

        stack_size = stack_size - 1u;
        let node_index = stack[stack_size];
        let node = blas_nodes[node_index];
        if (!svlm_aabb_intersects(local_brick, node)) {
            continue;
        }

        if (is_leaf(node)) {
            result.min_distance = 0.0;
            result.near_count = min(result.near_count + 1u, SVLM_MAX_REFINEMENT_HITS);
            result.overlap_count = result.overlap_count + 1u;
            result.occupied_volume = result.occupied_volume + svlm_aabb_intersection_volume(local_brick, node);
            result.face_mask |= svlm_face_mask(local_brick, node, face_eps);

            if (result.overlap_count >= SVLM_MAX_OVERLAP_HITS) {
                return result;
            }
        } else {
            let child_a = u32(node.min.w);
            let child_b = u32(node.max.w);
            let visit_a = svlm_aabb_intersects(local_brick, blas_nodes[child_a]);
            let visit_b = svlm_aabb_intersects(local_brick, blas_nodes[child_b]);

            if (visit_a && visit_b) {
                if (stack_size + 2u <= SVLM_QUERY_STACK_SIZE) {
                    stack[stack_size] = child_a;
                    stack_size = stack_size + 1u;
                    stack[stack_size] = child_b;
                    stack_size = stack_size + 1u;
                } else {
                    result.stack_overflow = 1u;
                    return result;
                }
            } else if (visit_a) {
                if (stack_size < SVLM_QUERY_STACK_SIZE) {
                    stack[stack_size] = child_a;
                    stack_size = stack_size + 1u;
                } else {
                    result.stack_overflow = 1u;
                    return result;
                }
            } else if (visit_b) {
                if (stack_size < SVLM_QUERY_STACK_SIZE) {
                    stack[stack_size] = child_b;
                    stack_size = stack_size + 1u;
                } else {
                    result.stack_overflow = 1u;
                    return result;
                }
            }
        }
    }

    return result;
}

fn svlm_query_brick_stats(brick: AABB, level: u32) -> SVLMBrickStats {
    var result: SVLMBrickStats;
    result.keep = 0u;
    result.overlap_count = 0u;
    result.near_count = 0u;
    result.stack_overflow = 0u;
    result.face_mask = 0u;
    result.min_distance = 3.402823e+38;
    result.occupied_fraction = 0.0;
    result.score = 0.0;
    result.thin_occluder = 0u;

    if (tlas_bvh_info.bvh2_count == 0u) {
        return result;
    }

    let brick_size = svlm_brick_size(&svlm_params, level);
    let keep_distance = brick_size;
    let keep_distance_sq = keep_distance * keep_distance;
    var occupied_fraction = 0.0;

    // TLAS traversal rejects whole entities by world-space AABB distance before
    // paying the cost of entity lookup and BLAS traversal.
    var stack: array<u32, SVLM_QUERY_STACK_SIZE>;
    stack[0] = tlas_bvh_info.bvh2_count - 1u;
    var stack_size = 1u;

    while (stack_size > 0u) {
        stack_size = stack_size - 1u;
        let node_index = stack[stack_size];
        let node = tlas_nodes[node_index];
        let distance_sq = svlm_aabb_distance_sq(brick, node);
        if (distance_sq > keep_distance_sq) {
            continue;
        }

        if (is_leaf(node)) {
            let mesh_id = u32(node.min.w);
            if (mesh_id == INVALID_IDX) {
                continue;
            }

            let prim_store = u32(-node.max.w - 1.0);
            if (prim_store >= arrayLength(&entity_index_lookup)) {
                continue;
            }
            let entity_index = entity_index_lookup[prim_store];
            if (entity_index == INVALID_IDX || entity_index >= arrayLength(&entity_transforms)) {
                continue;
            }

            let has_blas = svlm_mesh_has_blas(mesh_id);
            if (has_blas) {
                let entity_transform = entity_transforms[entity_index];
                let local_brick = svlm_world_aabb_to_local(brick, entity_transform);
                let local_distance_scale = max(svlm_world_to_local_distance_scale(entity_transform), 0.000001);
                let local_keep_distance = keep_distance * local_distance_scale;
                let local_face_eps = local_keep_distance * 0.08;
                // The BLAS is authored in mesh-local space, so both the brick and
                // distance thresholds are transformed before the finer query.
                let blas_stats = svlm_blas_stats(local_brick, mesh_id, local_keep_distance, local_face_eps);
                if (blas_stats.near_count != 0u || blas_stats.overlap_count != 0u) {
                    result.near_count = result.near_count + min(blas_stats.near_count, 16u);
                    result.overlap_count = result.overlap_count + min(blas_stats.overlap_count, 16u);
                    result.stack_overflow = max(result.stack_overflow, blas_stats.stack_overflow);
                    result.min_distance = min(result.min_distance, blas_stats.min_distance / local_distance_scale);
                    occupied_fraction = occupied_fraction + blas_stats.occupied_volume / max(svlm_aabb_volume(local_brick), 0.000001);
                    result.face_mask |= blas_stats.face_mask;
                }

                // Once a mesh has BLAS data, refinement must be driven by that
                // mesh-local triangle-bound hierarchy. Falling back to the TLAS
                // entity AABB here makes fine bricks fill empty entity bounds.
                continue;
            }

            // Conservative fallback for meshes whose BLAS is not available yet.
            // It may keep coarse coverage, but it intentionally does not write
            // overlap_count; only BLAS leaf overlap should drive finer bricks.
            let intersects_leaf = svlm_aabb_intersects(brick, node);
            if (!intersects_leaf) {
                continue;
            }

            result.near_count = result.near_count + 1u;
            result.min_distance = min(result.min_distance, sqrt(distance_sq));
            if (intersects_leaf) {
                occupied_fraction = occupied_fraction + svlm_aabb_intersection_volume(brick, node) / max(svlm_aabb_volume(brick), 0.000001);
                result.face_mask |= svlm_face_mask(brick, node, keep_distance * 0.08);
            }
        } else {
            let child_a = u32(node.min.w);
            let child_b = u32(node.max.w);
            let distance_a = svlm_aabb_distance_sq(brick, tlas_nodes[child_a]);
            let distance_b = svlm_aabb_distance_sq(brick, tlas_nodes[child_b]);
            let visit_a = distance_a <= keep_distance_sq;
            let visit_b = distance_b <= keep_distance_sq;

            if (visit_a && visit_b) {
                let near_child = select(child_b, child_a, distance_a <= distance_b);
                let far_child = select(child_a, child_b, distance_a <= distance_b);
                if (stack_size + 2u <= SVLM_QUERY_STACK_SIZE) {
                    stack[stack_size] = far_child;
                    stack_size = stack_size + 1u;
                    stack[stack_size] = near_child;
                    stack_size = stack_size + 1u;
                } else {
                    // TLAS stack pressure is only a coarse coverage hint. It
                    // should not become a split reason because that refines
                    // toward entity bounds instead of BLAS triangle bounds.
                    result.stack_overflow = 1u;
                    result.near_count = max(result.near_count, 1u);
                    result.min_distance = 0.0;
                    stack[stack_size] = near_child;
                    stack_size = stack_size + 1u;
                }
            } else if (visit_a) {
                if (stack_size < SVLM_QUERY_STACK_SIZE) {
                    stack[stack_size] = child_a;
                    stack_size = stack_size + 1u;
                } else {
                    result.stack_overflow = 1u;
                }
            } else if (visit_b) {
                if (stack_size < SVLM_QUERY_STACK_SIZE) {
                    stack[stack_size] = child_b;
                    stack_size = stack_size + 1u;
                } else {
                    result.stack_overflow = 1u;
                }
            }
        }
    }

    result.occupied_fraction = clamp(occupied_fraction, 0.0, 1.0);
    result.thin_occluder = select(0u, 1u, svlm_is_thin_occluder(result.face_mask));
    result.keep = select(0u, 1u, result.overlap_count > 0u || result.near_count > 0u);

    // Score is primarily for debug/stats today. The actual split decision below
    // stays rule-based so each reason remains obvious while tuning the V1.
    let complexity = clamp(f32(result.overlap_count) / 8.0, 0.0, 1.0);
    result.score =
        result.occupied_fraction * 0.55 +
        select(0.0, 0.25, result.min_distance <= keep_distance) +
        complexity * 0.20 +
        select(0.0, 0.25, result.thin_occluder != 0u);

    return result;
}

fn svlm_should_split(level: u32, stats: SVLMBrickStats) -> bool {
    let min_level = svlm_params.min_level;
    let max_level = svlm_params.max_level;
    if (level >= max_level) {
        return false;
    }
    if (level < min_level) {
        return true;
    }

    let brick_size = svlm_brick_size(&svlm_params, level);
    let near_threshold = brick_size * svlm_params.near_factor;
    let transition = stats.occupied_fraction > svlm_params.occupancy_split_min &&
        stats.occupied_fraction < svlm_params.occupancy_split_max;

    // Above the forced coarse density, only bricks that overlap BLAS leaf
    // bounds are allowed to refine. Nearby-but-non-overlapping bricks remain
    // coarse leaves so fine levels track real mesh detail, not TLAS padding.
    if (stats.overlap_count == 0u) {
        return false;
    }

    if (stats.thin_occluder != 0u) {
        return true;
    }
    if (transition) {
        return true;
    }
    if (stats.min_distance < near_threshold) {
        return true;
    }

    return true;
}

fn svlm_emit_leaf(node_index: u32, level: u32, coord: vec3<u32>, flags: u32, score: f32) {
    let max_leaf_bricks = svlm_params.max_leaf_bricks;
    let leaf_index = atomicAdd(&svlm_counters.leaf_count, 1u);
    if (leaf_index >= max_leaf_bricks) {
        atomicOr(&svlm_counters.status, SVLM_STATUS_LEAF_OVERFLOW);
        return;
    }

    let probe_base = leaf_index * SVLM_PROBES_PER_BRICK;
    let size = svlm_brick_size(&svlm_params, level);
    let origin = svlm_world_min(&svlm_params) + vec3<f32>(coord) * size;
    let leaf_flags = flags | SVLM_FLAG_LEAF;

    // Leaf bricks are the durable bake output. Probe positions are not stored:
    // debug/runtime code derives the 4x4x4 lattice from origin + size.
    node_pool[node_index].flags = leaf_flags;
    node_pool[node_index].leaf_index = leaf_index;

    leaf_bricks[leaf_index].morton = svlm_morton3(coord);
    leaf_bricks[leaf_index].level = level;
    leaf_bricks[leaf_index].probe_base = probe_base;
    leaf_bricks[leaf_index].neighbor_info = 0u;
    leaf_bricks[leaf_index].node_index = node_index;
    leaf_bricks[leaf_index].coord_x = coord.x;
    leaf_bricks[leaf_index].coord_y = coord.y;
    leaf_bricks[leaf_index].coord_z = coord.z;
    leaf_bricks[leaf_index].origin_x = origin.x;
    leaf_bricks[leaf_index].origin_y = origin.y;
    leaf_bricks[leaf_index].origin_z = origin.z;
    leaf_bricks[leaf_index].size = size;
    leaf_bricks[leaf_index].flags = leaf_flags;
    leaf_bricks[leaf_index].score = score;
    leaf_bricks[leaf_index].reserved_14 = 0u;
    leaf_bricks[leaf_index].reserved_15 = 0u;

    atomicAdd(&svlm_counters.probe_count, SVLM_PROBES_PER_BRICK);
}

fn svlm_emit_children(node_index: u32, level: u32, coord: vec3<u32>, flags: u32, score: f32) {
    let child_base = atomicAdd(&svlm_counters.node_count, 8u);
    if (child_base + 7u >= svlm_params.max_nodes) {
        atomicOr(&svlm_counters.status, SVLM_STATUS_NODE_OVERFLOW);
        // Preserve coverage when the pool is exhausted. The stats readback will
        // request a larger allocation and rebake, but this frame still has a
        // conservative leaf rather than a hole in the structure.
        svlm_emit_leaf(node_index, level, coord, flags, score);
        return;
    }

    let next_base = atomicAdd(&svlm_counters.next_count, 8u);
    node_pool[node_index].flags = flags | SVLM_FLAG_SHOULD_SPLIT;
    node_pool[node_index].child_base = child_base;
    node_pool[node_index].score = score;

    // Child coordinates are expressed in the next level's integer grid. Since
    // every level doubles resolution, appending one octant bit per axis is all
    // that is needed to keep the hierarchy aligned.
    for (var child = 0u; child < 8u; child = child + 1u) {
        let child_coord = coord * 2u + vec3<u32>(
            child & 1u,
            (child >> 1u) & 1u,
            (child >> 2u) & 1u
        );
        let child_index = child_base + child;
        svlm_write_node(child_index, level + 1u, child_coord, node_index, SVLM_FLAG_ACTIVE);
        next_nodes[next_base + child] = child_index;
    }
}

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let curr_count = atomicLoad(&svlm_counters.curr_count);
    if (gid.x >= curr_count) {
        return;
    }

    let node_index = curr_nodes[gid.x];
    let level = node_pool[node_index].level;
    let coord = vec3<u32>(
        node_pool[node_index].coord_x,
        node_pool[node_index].coord_y,
        node_pool[node_index].coord_z
    );
    let brick = svlm_node_aabb(&svlm_params, level, coord);
    let stats = svlm_query_brick_stats(brick, level);

    atomicAdd(&svlm_counters.level_counts[level], 1u);
    atomicMax(&svlm_counters.max_level_reached, level);

    node_pool[node_index].score = stats.score;

    var flags = SVLM_FLAG_ACTIVE;
    if (stats.keep != 0u) {
        flags |= SVLM_FLAG_OCCUPIED_OR_NEAR_GEOMETRY;
    }

    if (svlm_should_split(level, stats)) {
        atomicAdd(&svlm_counters.split_counts[level], 1u);
        svlm_emit_children(node_index, level, coord, flags, stats.score);
        return;
    }

    svlm_emit_leaf(node_index, level, coord, flags, stats.score);
}
