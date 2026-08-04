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
@group(1) @binding(12) var<storage, read> index_buffer: array<u32>;

const SVLM_QUERY_STACK_SIZE = 128u;
const SVLM_SAMPLE_BIN_COUNT = 48u;
const SVLM_SAMPLES_PER_BIN = 2u;
const SVLM_MAX_BLAS_NODE_VISITS = 384u;

fn svlm_write_node(index: u32, level: u32, coord: vec3<u32>) {
    node_pool[index].level = level;
    node_pool[index].flags = 0u;
    node_pool[index].child_base = INVALID_IDX;
    node_pool[index].leaf_index = INVALID_IDX;
    node_pool[index].coord_x = coord.x;
    node_pool[index].coord_y = coord.y;
    node_pool[index].coord_z = coord.z;
}

fn svlm_mesh_has_blas(mesh_id: u32) -> bool {
    if (mesh_id == INVALID_IDX || mesh_id >= arrayLength(&blas_directory)) {
        return false;
    }
    return blas_directory[mesh_id].leaf_count > 0u;
}

fn svlm_accumulate_triangle_stats(
    result: ptr<function, SVLMBlasStats>,
    sample_counts: ptr<function, array<u32, SVLM_SAMPLE_BIN_COUNT>>,
    node: AABB,
    directory: MeshDirectoryEntry,
    brick: AABB,
    query_brick: AABB,
    entity_transform: EntityTransform,
    count_overlap: bool,
    sample_limit: u32
) {
    let triangle_id = u32(node.min.w);
    let triangle_base = directory.first_index + triangle_id * 3u;
    if (triangle_base + 2u >= arrayLength(&index_buffer)) {
        return;
    }

    let vertex_indices = vec3<u32>(
        directory.first_vertex + index_buffer[triangle_base],
        directory.first_vertex + index_buffer[triangle_base + 1u],
        directory.first_vertex + index_buffer[triangle_base + 2u]
    );
    if (max(vertex_indices.x, max(vertex_indices.y, vertex_indices.z)) >= arrayLength(&vertex_buffer)) {
        return;
    }

    let local_0 = vertex_position(vertex_buffer[vertex_indices.x]);
    let local_1 = vertex_position(vertex_buffer[vertex_indices.y]);
    let local_2 = vertex_position(vertex_buffer[vertex_indices.z]);
    let local_cross = cross(local_1 - local_0, local_2 - local_0);
    if (dot(local_cross, local_cross) <= 1e-12) {
        return;
    }

    let world_normal = normalize(
        (entity_transform.transpose_inverse_model_matrix * vec4<f32>(local_cross, 0.0)).xyz
    );
    let world_0 = (entity_transform.transform * vec4<f32>(local_0, 1.0)).xyz;
    let world_1 = (entity_transform.transform * vec4<f32>(local_1, 1.0)).xyz;
    let world_2 = (entity_transform.transform * vec4<f32>(local_2, 1.0)).xyz;
    let world_centroid = (world_0 + world_1 + world_2) / 3.0;

    let triangle_bounds = AABB(
        vec4<f32>(min(world_0, min(world_1, world_2)), 0.0),
        vec4<f32>(max(world_0, max(world_1, world_2)), 0.0)
    );
    if (!svlm_aabb_intersects(query_brick, triangle_bounds)) {
        return;
    }
    if (count_overlap && svlm_aabb_intersects(brick, triangle_bounds)) {
        (*result).overlap_count += 1u;
    }

    // Winding-invariant orientation sectors separate different sides of a
    // corner, while four depth slices separate nearby parallel layers. Dense
    // triangles on the first surface visited can no longer consume the entire
    // sample budget before the walk reaches another geometric feature.
    let centroid_unit = clamp(
        (world_centroid - brick.min.xyz) /
            max(brick.max.xyz - brick.min.xyz, vec3<f32>(0.0001)),
        vec3<f32>(0.0),
        vec3<f32>(0.9999)
    );
    let abs_normal = abs(world_normal);
    var canonical_normal = world_normal;
    var orientation_sector = 0u;
    var depth_bin = u32(centroid_unit.x * 4.0);
    if (abs_normal.y > abs_normal.x && abs_normal.y >= abs_normal.z) {
        canonical_normal = select(world_normal, -world_normal, world_normal.y < 0.0);
        orientation_sector = 4u +
            select(0u, 1u, canonical_normal.x < 0.0) +
            select(0u, 2u, canonical_normal.z < 0.0);
        depth_bin = u32(centroid_unit.y * 4.0);
    } else if (abs_normal.z > abs_normal.x) {
        canonical_normal = select(world_normal, -world_normal, world_normal.z < 0.0);
        orientation_sector = 8u +
            select(0u, 1u, canonical_normal.x < 0.0) +
            select(0u, 2u, canonical_normal.y < 0.0);
        depth_bin = u32(centroid_unit.z * 4.0);
    } else {
        canonical_normal = select(world_normal, -world_normal, world_normal.x < 0.0);
        orientation_sector =
            select(0u, 1u, canonical_normal.y < 0.0) +
            select(0u, 2u, canonical_normal.z < 0.0);
    }
    let sample_bin = orientation_sector * 4u + depth_bin;
    if ((*sample_counts)[sample_bin] >= sample_limit) {
        return;
    }
    (*sample_counts)[sample_bin] += 1u;

    // Brick-relative moments avoid catastrophic cancellation in large worlds
    // when variance is reconstructed as E[x^2] - E[x]^2.
    let centroid = world_centroid - brick.min.xyz;

    // n*n^T is invariant to triangle winding, unlike an averaged normal. Its
    // coherence remains one for a flat two-sided surface and falls as distinct
    // orientations enter the brick.
    (*result).normal_diagonal += world_normal * world_normal;
    (*result).normal_cross += vec3<f32>(
        world_normal.x * world_normal.y,
        world_normal.x * world_normal.z,
        world_normal.y * world_normal.z
    );
    (*result).centroid_sum += centroid;
    (*result).centroid_square_sum += centroid * centroid;
    (*result).centroid_cross_sum += vec3<f32>(
        centroid.x * centroid.y,
        centroid.x * centroid.z,
        centroid.y * centroid.z
    );
    (*result).normal_count += 1u;
}

fn svlm_point_aabb_distance_sq(point: vec3<f32>, bounds: AABB) -> f32 {
    let distance = max(max(bounds.min.xyz - point, point - bounds.max.xyz), vec3<f32>(0.0));
    return dot(distance, distance);
}

// Picks one spatially representative leaf without enumerating its subtree. It
// is used only when the regular overlap walk exhausts its node budget.
fn svlm_find_sample_leaf(
    root_node: u32,
    local_brick: AABB,
    sample_point: vec3<f32>,
    sample_index: u32
) -> u32 {
    var node_index = root_node;
    for (var depth = 0u; depth < 64u; depth += 1u) {
        let node = blas_nodes[node_index];
        if (is_leaf(node)) {
            return node_index;
        }

        let child_a = u32(node.min.w);
        let child_b = u32(node.max.w);
        let visit_a = svlm_aabb_intersects(local_brick, blas_nodes[child_a]);
        let visit_b = svlm_aabb_intersects(local_brick, blas_nodes[child_b]);
        if (!visit_a && !visit_b) {
            return INVALID_IDX;
        }
        if (visit_a && visit_b) {
            let distance_a = svlm_point_aabb_distance_sq(sample_point, blas_nodes[child_a]);
            let distance_b = svlm_point_aabb_distance_sq(sample_point, blas_nodes[child_b]);
            let choose_b = distance_b < distance_a ||
                (distance_a == distance_b && (sample_index & 1u) != 0u);
            node_index = select(child_a, child_b, choose_b);
        } else {
            node_index = select(child_b, child_a, visit_a);
        }
    }
    return INVALID_IDX;
}

// Searches one mesh BLAS in local space for triangle-bound overlap. Traversal is
// bounded by node visits, while moment accumulation is bounded independently by
// spatial/orientation bins so the result is stable across BLAS traversal order.
fn svlm_blas_stats(
    local_brick: AABB,
    brick: AABB,
    query_brick: AABB,
    mesh_id: u32,
    entity_transform: EntityTransform
) -> SVLMBlasStats {
    var result: SVLMBlasStats;
    result.triangle_count = 0u;
    result.normal_count = 0u;
    result.overlap_count = 0u;
    result.padding = 0u;
    result.normal_diagonal = vec3<f32>(0.0);
    result.normal_cross = vec3<f32>(0.0);
    result.centroid_sum = vec3<f32>(0.0);
    result.centroid_square_sum = vec3<f32>(0.0);
    result.centroid_cross_sum = vec3<f32>(0.0);

    var sample_counts: array<u32, SVLM_SAMPLE_BIN_COUNT>;
    for (var sample_bin = 0u; sample_bin < SVLM_SAMPLE_BIN_COUNT; sample_bin += 1u) {
        sample_counts[sample_bin] = 0u;
    }

    if (!svlm_mesh_has_blas(mesh_id)) {
        return result;
    }

    let directory = blas_directory[mesh_id];
    let leaf_count = directory.leaf_count;
    if (leaf_count == 0u) {
        return result;
    }

    let root_node = directory.bvh2_base + (leaf_count * 2u - 1u) - 1u;
    if (!svlm_aabb_intersects(local_brick, blas_nodes[root_node])) {
        return result;
    }

    var stack: array<u32, SVLM_QUERY_STACK_SIZE>;
    stack[0] = root_node;
    var stack_size = 1u;
    var node_visits = 0u;
    var traversal_truncated = false;

    while (stack_size > 0u) {
        if (node_visits >= SVLM_MAX_BLAS_NODE_VISITS) {
            traversal_truncated = true;
            break;
        }
        node_visits = node_visits + 1u;

        stack_size = stack_size - 1u;
        let node_index = stack[stack_size];
        let node = blas_nodes[node_index];
        if (!svlm_aabb_intersects(local_brick, node)) {
            continue;
        }

        if (is_leaf(node)) {
            result.triangle_count = result.triangle_count + 1u;
            svlm_accumulate_triangle_stats(
                &result,
                &sample_counts,
                node,
                directory,
                brick,
                query_brick,
                entity_transform,
                true,
                SVLM_SAMPLES_PER_BIN
            );
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
                    traversal_truncated = true;
                    break;
                }
            } else if (visit_a) {
                if (stack_size < SVLM_QUERY_STACK_SIZE) {
                    stack[stack_size] = child_a;
                    stack_size = stack_size + 1u;
                } else {
                    traversal_truncated = true;
                    break;
                }
            } else if (visit_b) {
                if (stack_size < SVLM_QUERY_STACK_SIZE) {
                    stack[stack_size] = child_b;
                    stack_size = stack_size + 1u;
                } else {
                    traversal_truncated = true;
                    break;
                }
            }
        }
    }

    if (traversal_truncated) {
        let local_extent = local_brick.max.xyz - local_brick.min.xyz;
        for (var sample_index = 0u; sample_index < 27u; sample_index += 1u) {
            let sample_coord = vec3<u32>(
                sample_index % 3u,
                (sample_index / 3u) % 3u,
                sample_index / 9u
            );
            let sample_fraction = (vec3<f32>(sample_coord) + vec3<f32>(0.5)) / 3.0;
            let sample_point = local_brick.min.xyz + sample_fraction * local_extent;
            let leaf_index = svlm_find_sample_leaf(
                root_node,
                local_brick,
                sample_point,
                sample_index
            );
            if (leaf_index == INVALID_IDX) {
                continue;
            }
            svlm_accumulate_triangle_stats(
                &result,
                &sample_counts,
                blas_nodes[leaf_index],
                directory,
                brick,
                query_brick,
                entity_transform,
                false,
                SVLM_SAMPLES_PER_BIN * 2u
            );
        }
    }

    return result;
}

fn svlm_query_brick_stats(brick: AABB, level: u32) -> SVLMBrickStats {
    var result: SVLMBrickStats;
    result.triangle_count = 0u;
    result.normal_count = 0u;
    result.overlap_count = 0u;
    result.padding = 0u;
    result.normal_diagonal = vec3<f32>(0.0);
    result.normal_cross = vec3<f32>(0.0);
    result.centroid_sum = vec3<f32>(0.0);
    result.centroid_square_sum = vec3<f32>(0.0);
    result.centroid_cross_sum = vec3<f32>(0.0);

    if (tlas_bvh_info.bvh2_count == 0u) {
        return result;
    }

    let brick_size = svlm_brick_size(&svlm_params, level);
    let complexity_padding = brick_size * clamp(svlm_params.near_factor, 0.0, 0.5);
    let query_brick = AABB(
        brick.min - vec4<f32>(vec3<f32>(complexity_padding), 0.0),
        brick.max + vec4<f32>(vec3<f32>(complexity_padding), 0.0)
    );

    // A narrow neighborhood keeps corners and close layers visible when an
    // octree boundary separates their triangles. Nearby coherent planes still
    // fail every complexity test and therefore do not create a fine shell.
    var stack: array<u32, SVLM_QUERY_STACK_SIZE>;
    stack[0] = tlas_bvh_info.bvh2_count - 1u;
    var stack_size = 1u;

    while (stack_size > 0u) {
        stack_size = stack_size - 1u;
        let node_index = stack[stack_size];
        let node = tlas_nodes[node_index];
        if (!svlm_aabb_intersects(query_brick, node)) {
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
                let local_brick = svlm_world_aabb_to_local(query_brick, entity_transform);
                let blas_stats = svlm_blas_stats(
                    local_brick,
                    brick,
                    query_brick,
                    mesh_id,
                    entity_transform
                );
                if (blas_stats.triangle_count != 0u) {
                    result.triangle_count += blas_stats.triangle_count;
                    result.normal_count += blas_stats.normal_count;
                    result.overlap_count += blas_stats.overlap_count;
                    result.normal_diagonal += blas_stats.normal_diagonal;
                    result.normal_cross += blas_stats.normal_cross;
                    result.centroid_sum += blas_stats.centroid_sum;
                    result.centroid_square_sum += blas_stats.centroid_square_sum;
                    result.centroid_cross_sum += blas_stats.centroid_cross_sum;
                }

                // Once a mesh has BLAS data, refinement must be driven by that
                // mesh-local triangle-bound hierarchy. Falling back to the TLAS
                // entity AABB here makes fine bricks fill empty entity bounds.
                continue;
            }

            // Missing BLAS data retains the existing coarse coverage but cannot
            // provide triangle-level evidence for further refinement.
            continue;
        } else {
            let child_a = u32(node.min.w);
            let child_b = u32(node.max.w);
            let visit_a = svlm_aabb_intersects(query_brick, tlas_nodes[child_a]);
            let visit_b = svlm_aabb_intersects(query_brick, tlas_nodes[child_b]);

            if (visit_a && visit_b) {
                if (stack_size + 2u <= SVLM_QUERY_STACK_SIZE) {
                    stack[stack_size] = child_b;
                    stack_size = stack_size + 1u;
                    stack[stack_size] = child_a;
                    stack_size = stack_size + 1u;
                } else {
                    stack[stack_size] = child_a;
                    stack_size = stack_size + 1u;
                }
            } else if (visit_a) {
                if (stack_size < SVLM_QUERY_STACK_SIZE) {
                    stack[stack_size] = child_a;
                    stack_size = stack_size + 1u;
                } else {
                }
            } else if (visit_b) {
                if (stack_size < SVLM_QUERY_STACK_SIZE) {
                    stack[stack_size] = child_b;
                    stack_size = stack_size + 1u;
                } else {
                }
            }
        }
    }

    return result;
}

fn svlm_symmetric_mul(diagonal: vec3<f32>, cross_terms: vec3<f32>, value: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        diagonal.x * value.x + cross_terms.x * value.y + cross_terms.y * value.z,
        cross_terms.x * value.x + diagonal.y * value.y + cross_terms.z * value.z,
        cross_terms.y * value.x + cross_terms.z * value.y + diagonal.z * value.z
    );
}

fn svlm_dominant_normal(stats: SVLMBrickStats) -> vec3<f32> {
    var direction = vec3<f32>(1.0, 0.0, 0.0);
    if (stats.normal_diagonal.y > stats.normal_diagonal.x &&
        stats.normal_diagonal.y >= stats.normal_diagonal.z) {
        direction = vec3<f32>(0.0, 1.0, 0.0);
    } else if (stats.normal_diagonal.z > stats.normal_diagonal.x) {
        direction = vec3<f32>(0.0, 0.0, 1.0);
    }

    // Two power iterations are sufficient for this small heuristic matrix and
    // avoid the cost and instability of an analytic symmetric eigensolver.
    for (var iteration = 0u; iteration < 2u; iteration += 1u) {
        direction = normalize(
            svlm_symmetric_mul(stats.normal_diagonal, stats.normal_cross, direction)
        );
    }
    return direction;
}

fn svlm_normal_variation(stats: SVLMBrickStats) -> f32 {
    let sample_count = max(f32(stats.normal_count), 1.0);
    let frobenius_sq = dot(stats.normal_diagonal, stats.normal_diagonal) +
        2.0 * dot(stats.normal_cross, stats.normal_cross);
    let coherence = clamp(sqrt(max(frobenius_sq, 0.0)) / sample_count, 0.0, 1.0);
    return 1.0 - coherence;
}

fn svlm_layer_separation(stats: SVLMBrickStats, brick_size: f32) -> f32 {
    let sample_count = max(f32(stats.normal_count), 1.0);
    let direction = svlm_dominant_normal(stats);
    let centroid_mean = stats.centroid_sum / sample_count;
    let second_moment = (
        dot(direction * direction, stats.centroid_square_sum) +
        2.0 * dot(
            vec3<f32>(
                direction.x * direction.y,
                direction.x * direction.z,
                direction.y * direction.z
            ),
            stats.centroid_cross_sum
        )
    ) / sample_count;
    let variance = max(second_moment - pow(dot(direction, centroid_mean), 2.0), 0.0);
    return sqrt(variance) / max(brick_size, 0.0001);
}

fn svlm_should_split(level: u32, stats: SVLMBrickStats) -> bool {
    let min_level = u32(max(svlm_params.min_level, 0.0));
    let max_level = u32(max(svlm_params.max_level, 0.0));
    if (level >= max_level) {
        return false;
    }
    if (level < min_level) {
        return true;
    }

    // Above the forced coarse density, refinement is earned by complexity. A
    // coherent low-density plane deliberately remains at this level.
    if (stats.normal_count == 0u) {
        return false;
    }

    let density_threshold = max(u32(svlm_params.triangle_density_threshold), 2u);
    let dense_geometry = stats.overlap_count >= density_threshold;
    let density_scale = select(1.0, 0.5, dense_geometry);
    let normal_threshold = svlm_params.normal_variation_threshold * density_scale;
    if (svlm_normal_variation(stats) >= normal_threshold) {
        return true;
    }

    let brick_size = svlm_brick_size(&svlm_params, level);
    let layer_threshold = svlm_params.layer_separation_factor * density_scale;
    if (svlm_layer_separation(stats, brick_size) >= layer_threshold) {
        return true;
    }

    // Triangle density only increases sensitivity to measured shape complexity;
    // it is never sufficient on its own. A heavily tessellated but coherent
    // plane therefore stays coarse instead of refining to the maximum level.
    return false;
}

fn svlm_emit_leaf(node_index: u32, level: u32, coord: vec3<u32>) {
    let leaf_capacity = min(
        u32(max(svlm_params.leaf_capacity, 0.0)),
        arrayLength(&leaf_bricks)
    );
    let leaf_index = atomicAdd(&svlm_counters.leaf_count, 1u);
    if (leaf_index >= leaf_capacity) {
        atomicOr(&svlm_counters.status, SVLM_STATUS_LEAF_OVERFLOW);
        return;
    }

    let probe_base = leaf_index * SVLM_PROBES_PER_BRICK;
    let size = svlm_brick_size(&svlm_params, level);
    let origin = svlm_world_min(&svlm_params) + vec3<f32>(coord) * size;
    // Leaf bricks are the durable bake output. Probe positions are not stored:
    // debug/runtime code derives the 4x4x4 lattice from origin + size.
    node_pool[node_index].flags = SVLM_FLAG_LEAF;
    node_pool[node_index].leaf_index = leaf_index;

    leaf_bricks[leaf_index].level = level;
    leaf_bricks[leaf_index].probe_base = probe_base;
    leaf_bricks[leaf_index].origin_x = origin.x;
    leaf_bricks[leaf_index].origin_y = origin.y;
    leaf_bricks[leaf_index].origin_z = origin.z;
    leaf_bricks[leaf_index].size = size;
    atomicAdd(&svlm_counters.probe_count, SVLM_PROBES_PER_BRICK);
}

fn svlm_emit_children(node_index: u32, level: u32, coord: vec3<u32>) {
    let child_base = atomicAdd(&svlm_counters.node_count, 8u);
    if (child_base + 7u >= u32(max(svlm_params.max_nodes, 0.0))) {
        atomicOr(&svlm_counters.status, SVLM_STATUS_NODE_OVERFLOW);
        // Preserve coverage when the pool is exhausted. The stats readback will
        // request a larger allocation and rebake, but this frame still has a
        // conservative leaf rather than a hole in the structure.
        svlm_emit_leaf(node_index, level, coord);
        return;
    }

    let next_base = atomicAdd(&svlm_counters.next_count, 8u);
    node_pool[node_index].child_base = child_base;

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
        svlm_write_node(child_index, level + 1u, child_coord);
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

    if (svlm_should_split(level, stats)) {
        atomicAdd(&svlm_counters.split_counts[level], 1u);
        svlm_emit_children(node_index, level, coord);
        return;
    }

    svlm_emit_leaf(node_index, level, coord);
}
