#if HAS_PRECISION_FLOAT
enable f16;
#endif

#if HAS_SUBGROUPS
#include "subgroup_warp.wgsl"
#else
#include "logical_warp.wgsl"
#endif

// ------------------------------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------------------------------ 

// 32‑bit handle ─ 21 bits chunk index | 7 bits row_index | 4 bits generation
const ENTITY_ROW_BITS = 28;
const LOCAL_SLOT_BITS = 8;
const ENTITY_GEN_BITS = 4;
const ENTITY_ROW_MASK = (1 << ENTITY_ROW_BITS) - 1;
const ENTITY_GEN_MASK = (1 << ENTITY_GEN_BITS) - 1;
const LOCAL_SLOT_MASK = (1 << LOCAL_SLOT_BITS) - 1;
const CHUNK_INDEX_BITS = ENTITY_ROW_BITS - LOCAL_SLOT_BITS;
const CHUNK_INDEX_MASK = ((1 << CHUNK_INDEX_BITS) - 1) << LOCAL_SLOT_BITS;

const EF_ALIVE = 1u << 0;
const EF_DIRTY = 1u << 1;
const EF_IGNORE_PARENT_SCALE = 1u << 2;
const EF_IGNORE_PARENT_ROTATION = 1u << 3;
const EF_TRANSFORM_DIRTY = 1u << 4;
const EF_AABB_DIRTY = 1u << 5;
const EF_BILLBOARD = 1u << 6;
const EF_MOVED = 1u << 7;

const LOG_DEPTH_C = 0.1; // Can adjust this value based on scene scale
const MAX_UINT = 4294967295u;
const INVALID_IDX = 0xffffffffu;

const PI = 3.14159265359;

struct Vertex {
    position: vec4<precision_float>,
    normal: vec4<precision_float>,
    tangent: vec4<precision_float>,
    bitangent: vec4<precision_float>,
    uv: vec2<precision_float>,
    section_index: f32,
    _padding: f32,
};

struct View {
    view_matrix: mat4x4f,
    prev_view_matrix: mat4x4f,
    projection_matrix: mat4x4f,
    prev_projection_matrix: mat4x4f,
    view_projection_matrix: mat4x4f,
    inverse_view_projection_matrix: mat4x4f,
    view_direction: vec4f,
    near: f32,
    far: f32,
    culling_enabled: f32,
    occlusion_enabled: f32,
    frustum: array<vec4f, 6>,
    view_position: vec4f,
    view_rotation: vec4f,
    view_right: vec4f,
    fov: f32,
    aspect_ratio: f32,
    distance_check_enabled: f32,
    velocity: vec4f,
    zoom: f32,
    clipmap_count: u32,
};

struct FrameInfo {
    view_index: f32,
    time: f32,
    frame_index: f32,
    resolution: vec2f,
    cursor_world_position: vec4f,
};

struct EntityTransform {
    transform: mat4x4f,
    transpose_inverse_model_matrix: mat4x4f,
    prev_transform: mat4x4f,
};

struct ObjectInstance {
    batch: u32,
    row: u32,
};

struct DrawCommand {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    vertex_offset: i32,
    first_instance: u32,
};

struct MeshDirectoryEntry {
    bvh2_base: u32,
    bvh2_capacity: u32,
    bvh4_base: u32,
    bvh4_capacity: u32,
    leaf_count: u32,
    first_vertex: u32,
    first_index: u32,
    padding: u32,
};

struct StandardMaterialParams {
    albedo: vec4<precision_float>,
    normal: vec4<precision_float>,
    emission_roughness_metallic_tiling: vec4<precision_float>,
    ao_height_specular: vec4<precision_float>,
    texture_flags1: vec4<f32>, // x: albedo, y: normal, z: roughness, w: metallic
    texture_flags2: vec4<f32>, // x: ao, y: height, z: specular, w: emission 
    albedo_handle: f32,
    normal_handle: f32,
    roughness_handle: f32,
    metallic_handle: f32,
    ao_handle: f32,
    height_handle: f32,
    specular_handle: f32,
    emission_handle: f32,
};

// ------------------------------------------------------------------------------------
// Constants
// ------------------------------------------------------------------------------------ 

// 4x4 Bayer matrix for dithering
const bayer_matrix = array<f32, 16>(
    0.0 / 16.0, 8.0 / 16.0, 2.0 / 16.0, 10.0 / 16.0,
    12.0 / 16.0, 4.0 / 16.0, 14.0 / 16.0, 6.0 / 16.0,
    3.0 / 16.0, 11.0 / 16.0, 1.0 / 16.0, 9.0 / 16.0,
    15.0 / 16.0, 7.0 / 16.0, 13.0 / 16.0, 5.0 / 16.0
);

const identity_matrix = mat4x4f(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0
);

const epsilon = 1e-4;
const zero_vec4 = vec4f(0.0, 0.0, 0.0, 0.0);
const world_up = vec3f(0.0, 1.0, 0.0);
const world_right = vec3f(1.0, 0.0, 0.0);
const world_forward = vec3f(0.0, 0.0, 1.0);
const pos_inf = 3.402823466e+38;
const neg_inf = -3.402823466e+38;

const one_over_float_max = 1.0 / 4294967295.0;

