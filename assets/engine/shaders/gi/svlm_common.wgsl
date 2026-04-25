#include "common.wgsl"
#include "acceleration_common.wgsl"

// Sparse Volumetric Lightmapper shared definitions.
//
// The SVLM build is intentionally GPU-owned: the CPU writes a small parameter
// block, then compute passes seed a root grid, classify a breadth-first frontier
// against the TLAS/BLAS, append child nodes, and finally emit only leaf bricks.
// The node and leaf records are kept as flat u32 arrays so every pass can use
// stable byte layouts without WGSL struct padding surprises.

// Node flags describe the current build state. Runtime lighting is not stored
// here yet; these bits are only about allocation/refinement/debug visibility.
const SVLM_FLAG_ACTIVE = 1u << 0u;
const SVLM_FLAG_OCCUPIED_OR_NEAR_GEOMETRY = 1u << 1u;
const SVLM_FLAG_SHOULD_SPLIT = 1u << 2u;
const SVLM_FLAG_LEAF = 1u << 3u;
const SVLM_FLAG_INVALID = 1u << 4u;

const SVLM_PROBES_PER_BRICK = 64u;

// Status bits are sticky for a bake. The JS owner reads them back and can queue
// a larger GPU allocation without forcing fallback CPU construction.
const SVLM_STATUS_NODE_OVERFLOW = 1u << 0u;
const SVLM_STATUS_LEAF_OVERFLOW = 1u << 1u;
const SVLM_STATUS_ROOT_OVERFLOW = 1u << 2u;

const SVLM_DEBUG_PROBE_RADIUS = 18.0;

// Counter layout mirrors the JS readback indices exactly:
// slots 0..10 are named counters, slots 11..15 are reserved, slots 16..31 are
// split_counts, slots 32..47 are level_counts, and slots 48..63 are reserved.
// Keeping this as a struct makes shader code readable without changing the
// 64-u32 buffer contract used by stats readback.
struct SVLMCounters {
    node_count: atomic<u32>,
    curr_count: atomic<u32>,
    next_count: atomic<u32>,
    leaf_count: atomic<u32>,
    probe_count: atomic<u32>,
    debug_line_count: atomic<u32>,
    status: atomic<u32>,
    max_level_reached: atomic<u32>,
    current_level: atomic<u32>,
    reserved_11: atomic<u32>,
    reserved_12: atomic<u32>,
    reserved_13: atomic<u32>,
    split_counts: array<atomic<u32>, 16>,
    level_counts: array<atomic<u32>, 16>,
    reserved_tail: array<atomic<u32>, 16>,
};

// Node layout mirrors the old 16-u32 node record stride:
// slots 0..8 hold identity/hierarchy/grid data, slot 9 is a float score, and
// slots 10..15 are reserved. JS still allocates max_nodes * 16 words.
struct SVLMNode {
    morton: u32,
    level: u32,
    flags: u32,
    child_base: u32,
    leaf_index: u32,
    parent_index: u32,
    coord_x: u32,
    coord_y: u32,
    coord_z: u32,
    score: f32,
    reserved_10: u32,
    reserved_11: u32,
    reserved_12: u32,
    reserved_13: u32,
    reserved_14: u32,
    reserved_15: u32,
};

// Leaf brick layout mirrors the old 16-u32 leaf record stride:
// slots 0..7 are integer identity/coord data, slots 8..11 are float placement,
// slots 12..13 are flags/score, and slots 14..15 are reserved. Keeping the
// stride fixed lets JS keep allocating max_leaf_bricks * 16 words.
struct SVLMLeafBrick {
    morton: u32,
    level: u32,
    probe_base: u32,
    neighbor_info: u32,
    node_index: u32,
    coord_x: u32,
    coord_y: u32,
    coord_z: u32,
    origin_x: f32,
    origin_y: f32,
    origin_z: f32,
    size: f32,
    flags: u32,
    score: f32,
    reserved_14: u32,
    reserved_15: u32,
};

// Classification distills the TLAS/BLAS overlap search into the handful of
// signals needed by the split heuristic. It is deliberately coarse for this V1:
// enough to allocate bricks around real geometry before irradiance is added.
struct SVLMBrickStats {
    keep: u32,
    overlap_count: u32,
    near_count: u32,
    stack_overflow: u32,
    face_mask: u32,
    min_distance: f32,
    occupied_fraction: f32,
    score: f32,
    thin_occluder: u32,
};

// Keep this field order in lockstep with the JS parameter word offsets in
// SparseVolumetricLightmapper. The buffer is written as 32-bit words and read
// here as a WGSL struct, so each field is exactly one 32-bit slot.
struct SVLMParams {
    world_min_x: f32,
    world_min_y: f32,
    world_min_z: f32,
    root_size: f32,

    root_dim_x: u32,
    root_dim_y: u32,
    root_dim_z: u32,
    max_level: u32,

    max_nodes: u32,
    max_leaf_bricks: u32,
    min_level: u32,
    near_factor: f32,

    occupancy_split_min: f32,
    occupancy_split_max: f32,
    requested_root_size: f32,
    bake_padding: f32,

    bake_serial: u32,
    root_count: u32,
    scene_min_x: f32,
    scene_min_y: f32,

    scene_min_z: f32,
    scene_max_x: f32,
    scene_max_y: f32,
    scene_max_z: f32,

    debug_level: i32,
    debug_leaf_page_groups_y: u32,
    debug_gather_page_groups_x: u32,
    debug_gather_page_groups_y: u32,
};

struct SVLMBlasStats {
    overlap_count: u32,
    near_count: u32,
    stack_overflow: u32,
    face_mask: u32,
    min_distance: f32,
    occupied_volume: f32,
};

fn svlm_root_dims(params: ptr<storage, SVLMParams, read_write>) -> vec3<u32> {
    return vec3<u32>((*params).root_dim_x, (*params).root_dim_y, (*params).root_dim_z);
}

// Morton keys give us a stable spatial identity for each brick without needing
// pointer-heavy tree links. The explicit coord slots are still kept for cheap
// child generation and debug reconstruction.
fn svlm_expand_bits_10(v_in: u32) -> u32 {
    var v = v_in & 0x000003ffu;
    v = (v | (v << 16u)) & 0x030000ffu;
    v = (v | (v << 8u)) & 0x0300f00fu;
    v = (v | (v << 4u)) & 0x030c30c3u;
    v = (v | (v << 2u)) & 0x09249249u;
    return v;
}

fn svlm_morton3(coord: vec3<u32>) -> u32 {
    return svlm_expand_bits_10(coord.x) |
        (svlm_expand_bits_10(coord.y) << 1u) |
        (svlm_expand_bits_10(coord.z) << 2u);
}

fn svlm_brick_size(params: ptr<storage, SVLMParams, read_write>, level: u32) -> f32 {
    return (*params).root_size / f32(1u << level);
}

fn svlm_world_min(params: ptr<storage, SVLMParams, read_write>) -> vec3<f32> {
    return vec3<f32>(
        (*params).world_min_x,
        (*params).world_min_y,
        (*params).world_min_z
    );
}

fn svlm_node_aabb(params: ptr<storage, SVLMParams, read_write>, level: u32, coord: vec3<u32>) -> AABB {
    let size = svlm_brick_size(params, level);
    let origin = svlm_world_min(params) + vec3<f32>(coord) * size;
    return AABB(vec4<f32>(origin, 0.0), vec4<f32>(origin + vec3<f32>(size), 0.0));
}

// The split heuristic uses AABB distance/overlap rather than triangle tests.
// That keeps classification cheap enough to run for every candidate brick while
// still using the scene's GPU acceleration data as the source of truth.
fn svlm_aabb_volume(bounds: AABB) -> f32 {
    let e = max(vec3<f32>(0.0), bounds.max.xyz - bounds.min.xyz);
    return e.x * e.y * e.z;
}

fn svlm_aabb_distance_sq(a: AABB, b: AABB) -> f32 {
    var d = vec3<f32>(0.0);
    d.x = max(max(b.min.x - a.max.x, a.min.x - b.max.x), 0.0);
    d.y = max(max(b.min.y - a.max.y, a.min.y - b.max.y), 0.0);
    d.z = max(max(b.min.z - a.max.z, a.min.z - b.max.z), 0.0);
    return dot(d, d);
}

fn svlm_aabb_intersection_volume(a: AABB, b: AABB) -> f32 {
    let mn = max(a.min.xyz, b.min.xyz);
    let mx = min(a.max.xyz, b.max.xyz);
    let e = max(vec3<f32>(0.0), mx - mn);
    return e.x * e.y * e.z;
}

fn svlm_aabb_intersects(a: AABB, b: AABB) -> bool {
    return all(a.min.xyz <= b.max.xyz) && all(a.max.xyz >= b.min.xyz);
}

// Tracks which brick faces are touched by nearby geometry. Opposite touched
// faces are a cheap "thin occluder" proxy: one brick likely spans both sides of
// a wall/slab and should refine.
fn svlm_face_mask(brick: AABB, geom: AABB, eps: f32) -> u32 {
    var mask = 0u;
    if (geom.min.x <= brick.min.x + eps) { mask |= 1u << 0u; }
    if (geom.max.x >= brick.max.x - eps) { mask |= 1u << 1u; }
    if (geom.min.y <= brick.min.y + eps) { mask |= 1u << 2u; }
    if (geom.max.y >= brick.max.y - eps) { mask |= 1u << 3u; }
    if (geom.min.z <= brick.min.z + eps) { mask |= 1u << 4u; }
    if (geom.max.z >= brick.max.z - eps) { mask |= 1u << 5u; }
    return mask;
}

fn svlm_is_thin_occluder(face_mask: u32) -> bool {
    return ((face_mask & 0x03u) == 0x03u) ||
        ((face_mask & 0x0cu) == 0x0cu) ||
        ((face_mask & 0x30u) == 0x30u);
}

// Convert the world-space brick into mesh-local BLAS space. The TLAS gives us
// coarse entity overlap; the BLAS query must happen in the mesh's local domain.
fn svlm_to_local_point(p_world: vec3<f32>, entity_transform: EntityTransform) -> vec3<f32> {
    let ro_rel = p_world - entity_transform.transform[3].xyz;
    let t_col0 = entity_transform.transpose_inverse_model_matrix[0].xyz;
    let t_col1 = entity_transform.transpose_inverse_model_matrix[1].xyz;
    let t_col2 = entity_transform.transpose_inverse_model_matrix[2].xyz;
    return vec3<f32>(
        dot(ro_rel, t_col0),
        dot(ro_rel, t_col1),
        dot(ro_rel, t_col2)
    );
}

fn svlm_world_to_local_distance_scale(entity_transform: EntityTransform) -> f32 {
    let t_col0 = entity_transform.transpose_inverse_model_matrix[0].xyz;
    let t_col1 = entity_transform.transpose_inverse_model_matrix[1].xyz;
    let t_col2 = entity_transform.transpose_inverse_model_matrix[2].xyz;
    return max(max(length(t_col0), length(t_col1)), length(t_col2));
}

fn svlm_world_aabb_to_local(bounds: AABB, entity_transform: EntityTransform) -> AABB {
    let min_pt = bounds.min.xyz;
    let max_pt = bounds.max.xyz;
    var local_min = svlm_to_local_point(min_pt, entity_transform);
    var local_max = local_min;

    for (var i = 1u; i < 8u; i = i + 1u) {
        let p = vec3<f32>(
            select(min_pt.x, max_pt.x, (i & 1u) != 0u),
            select(min_pt.y, max_pt.y, (i & 2u) != 0u),
            select(min_pt.z, max_pt.z, (i & 4u) != 0u)
        );
        let local = svlm_to_local_point(p, entity_transform);
        local_min = min(local_min, local);
        local_max = max(local_max, local);
    }

    return AABB(vec4<f32>(local_min, 0.0), vec4<f32>(local_max, 0.0));
}
