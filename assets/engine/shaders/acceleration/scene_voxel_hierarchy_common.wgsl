// ============================================================================
// Scene Voxel HDDA Hierarchy Layout
// ============================================================================
//
// Camera-centered clip levels are packed into shared buffers. Each active clip
// level owns a dense 256^3 leaf bitset plus a 32^3 -> 4^3 -> 1^3 skip
// hierarchy. The hierarchy changes traversal granularity only; a successful
// trace still resolves to a leaf cell at the selected clip level.
// ============================================================================

struct SceneVoxelizationParams {
    grid_origin: vec3<f32>,
    voxel_size: f32,
    meshlet_count: u32,
    dispatch_width: u32,
    resolution: u32,
    flags: u32,
};

struct SceneVoxelClipmapParams {
    clip_level_count: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
    levels: array<SceneVoxelizationParams, 8>,
};

const SCENE_VOXEL_CLIPMAP_MAX_LEVEL_COUNT: u32 = 8u;
const SCENE_VOXEL_CLIPMAP_LEAF_WORD_COUNT: u32 = 524288u;

const SCENE_VOXEL_HDDA_LEVEL_COUNT: u32 = 4u;
const SCENE_VOXEL_HDDA_MAX_LEVEL: u32 = 3u;

const SCENE_VOXEL_HDDA_LEVEL_1_RESOLUTION: u32 = 32u;
const SCENE_VOXEL_HDDA_LEVEL_2_RESOLUTION: u32 = 4u;
const SCENE_VOXEL_HDDA_LEVEL_3_RESOLUTION: u32 = 1u;

const SCENE_VOXEL_HDDA_LEVEL_1_WORD_OFFSET: u32 = 0u;
const SCENE_VOXEL_HDDA_LEVEL_1_WORD_COUNT: u32 = 1024u;
const SCENE_VOXEL_HDDA_LEVEL_2_WORD_OFFSET: u32 = 1024u;
const SCENE_VOXEL_HDDA_LEVEL_2_WORD_COUNT: u32 = 2u;
const SCENE_VOXEL_HDDA_LEVEL_3_WORD_OFFSET: u32 = 1026u;
const SCENE_VOXEL_HDDA_HIERARCHY_WORD_COUNT: u32 = 1027u;

fn scene_voxel_clipmap_leaf_word_offset(clip_level: u32) -> u32 {
    return clip_level * SCENE_VOXEL_CLIPMAP_LEAF_WORD_COUNT;
}

fn scene_voxel_clipmap_hierarchy_word_offset(clip_level: u32) -> u32 {
    return clip_level * SCENE_VOXEL_HDDA_HIERARCHY_WORD_COUNT;
}

fn scene_voxel_hdda_level_resolution(level: u32) -> u32 {
    switch level {
        case 0u: { return 256u; }
        case 1u: { return SCENE_VOXEL_HDDA_LEVEL_1_RESOLUTION; }
        case 2u: { return SCENE_VOXEL_HDDA_LEVEL_2_RESOLUTION; }
        default: { return SCENE_VOXEL_HDDA_LEVEL_3_RESOLUTION; }
    }
}

fn scene_voxel_hdda_level_span(level: u32) -> u32 {
    switch level {
        case 0u: { return 1u; }
        case 1u: { return 8u; }
        case 2u: { return 64u; }
        default: { return 256u; }
    }
}

fn scene_voxel_hdda_level_word_offset(level: u32) -> u32 {
    switch level {
        case 1u: { return SCENE_VOXEL_HDDA_LEVEL_1_WORD_OFFSET; }
        case 2u: { return SCENE_VOXEL_HDDA_LEVEL_2_WORD_OFFSET; }
        default: { return SCENE_VOXEL_HDDA_LEVEL_3_WORD_OFFSET; }
    }
}

fn scene_voxel_hdda_linear_index(coord: vec3<u32>, resolution: u32) -> u32 {
    return coord.x + resolution * (coord.y + resolution * coord.z);
}
