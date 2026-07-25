#include "common.wgsl"
#include "acceleration_common.wgsl"

// Sparse Volumetric Lightmapper shared definitions.
//
// The SVLM build is intentionally GPU-owned: the CPU writes a small parameter
// block, then compute passes seed a root grid, classify a breadth-first frontier
// against the TLAS/BLAS, append child nodes, and finally emit only leaf bricks.
// The node and leaf records contain only fields consumed by build, preview, or
// debug passes. Their exact word strides are part of the bake artifact format.
const SVLM_FLAG_LEAF = 1u << 3u;

const SVLM_PROBES_PER_BRICK = 64u;
const SVLM_SH_WORDS_PER_PROBE = 6u;

// Status bits are sticky for a bake. The JS owner reads them back and can queue
// a larger GPU allocation without forcing fallback CPU construction.
const SVLM_STATUS_NODE_OVERFLOW = 1u << 0u;
const SVLM_STATUS_LEAF_OVERFLOW = 1u << 1u;
const SVLM_STATUS_ROOT_OVERFLOW = 1u << 2u;

// Counter layout mirrors the JS readback indices exactly: slots 0..7 are build
// counters, 8..23 are split counts, 24..39 are level counts, 40..43 track the
// progressive irradiance bake and hierarchy-readback handoff.
struct SVLMCounters {
    node_count: atomic<u32>,
    curr_count: atomic<u32>,
    next_count: atomic<u32>,
    leaf_count: atomic<u32>,
    probe_count: atomic<u32>,
    status: atomic<u32>,
    max_level_reached: atomic<u32>,
    current_level: atomic<u32>,
    split_counts: array<atomic<u32>, 16>,
    level_counts: array<atomic<u32>, 16>,
    irradiance_cursor: atomic<u32>,
    irradiance_sample_index: atomic<u32>,
    irradiance_completed_probe_samples: atomic<u32>,
    irradiance_status: atomic<u32>,
};

// Compact seven-word runtime octree node.
struct SVLMNode {
    level: u32,
    flags: u32,
    child_base: u32,
    leaf_index: u32,
    coord_x: u32,
    coord_y: u32,
    coord_z: u32,
};

// Compact six-word leaf record. Probe positions remain implicit.
struct SVLMLeafBrick {
    level: u32,
    probe_base: u32,
    origin_x: f32,
    origin_y: f32,
    origin_z: f32,
    size: f32,
};

// Compact transient ray record. hit_payload_t stores the probe origin during
// tracing; after a hit, xyz become UV/section and w becomes signed hit distance.
struct SVLMProbeRayData {
    hit_payload_t: vec4<f32>,
    ray_direction: vec4<f32>,
    nee_light_radiance: vec4<f32>,
    state_u32: vec4<u32>,
    radiance: vec4<f32>,
    meta_u32: vec4<u32>,
};

struct SVLMProbeRayDataHeader {
    active_ray_count: atomic<u32>,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct SVLMProbeRayDataHeaderReadOnly {
    active_ray_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct SVLMProbeRayDataBuffer {
    header: SVLMProbeRayDataHeader,
    rays: array<SVLMProbeRayData>,
};

struct SVLMProbeRayDataBufferReadOnlyHeader {
    header: SVLMProbeRayDataHeaderReadOnly,
    rays: array<SVLMProbeRayData>,
};

// Classification distills the TLAS/BLAS overlap search into the handful of
// signals needed by the split heuristic. It is deliberately coarse for this V1:
// enough to allocate bricks around real geometry before irradiance is added.
struct SVLMBrickStats {
    overlap_count: u32,
    face_mask: u32,
    min_distance: f32,
    occupied_fraction: f32,
    thin_occluder: u32,
};

// Keep this field order in lockstep with the JS parameter word offsets in
// SparseVolumetricLightmapper. The buffer is written as f32 slots; fields that
// represent integer values are cast at runtime where they are used.
struct SVLMParams {
    world_min_x: f32,
    world_min_y: f32,
    world_min_z: f32,
    root_size: f32,

    root_dim_x: f32,
    root_dim_y: f32,
    root_dim_z: f32,
    max_level: f32,

    max_nodes: f32,
    leaf_capacity: f32,
    min_level: f32,
    near_factor: f32,

    occupancy_split_min: f32,
    occupancy_split_max: f32,
    requested_root_size: f32,
    bake_padding: f32,

    bake_serial: f32,
    root_count: f32,
    scene_min_x: f32,
    scene_min_y: f32,

    scene_min_z: f32,
    scene_max_x: f32,
    scene_max_y: f32,
    scene_max_z: f32,

    debug_level: f32,
    debug_leaf_page_groups_y: f32,
    debug_gather_page_groups_x: f32,
    debug_gather_page_groups_y: f32,

    irradiance_rays_per_probe: f32,
    irradiance_probes_per_batch: f32,
    irradiance_sample_count: f32,
    irradiance_max_ray_distance: f32,

    irradiance_format_version: f32,
    irradiance_sh_words_per_probe: f32,
};

struct SVLMBlasStats {
    overlap_count: u32,
    near_count: u32,
    face_mask: u32,
    min_distance: f32,
    occupied_volume: f32,
};

fn svlm_root_dims(params: ptr<storage, SVLMParams, read_write>) -> vec3<u32> {
    return vec3<u32>(
        u32(max((*params).root_dim_x, 0.0)),
        u32(max((*params).root_dim_y, 0.0)),
        u32(max((*params).root_dim_z, 0.0))
    );
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

fn svlm_probe_position(leaf: SVLMLeafBrick, local_probe_index: u32) -> vec3<f32> {
    let local_coord = vec3<u32>(
        local_probe_index & 3u,
        (local_probe_index >> 2u) & 3u,
        (local_probe_index >> 4u) & 3u
    );
    let origin = vec3<f32>(leaf.origin_x, leaf.origin_y, leaf.origin_z);
    let spacing = max(leaf.size / 4.0, 0.0001);
    // Keep probes inside their owning adaptive leaf. The old endpoint lattice
    // placed entire probe planes on shared brick boundaries and directly on
    // thin geometry such as Sponza's floor.
    return origin + (vec3<f32>(local_coord) + vec3<f32>(0.5)) * spacing;
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
