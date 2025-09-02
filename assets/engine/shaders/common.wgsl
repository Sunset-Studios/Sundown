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

struct Vertex {
    position: vec4<precision_float>,
    normal: vec4<precision_float>,
    tangent: vec4<precision_float>,
    bitangent: vec4<precision_float>,
    uv: vec2<precision_float>,
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

// ------------------------------------------------------------------------------------
// Buffers
// ------------------------------------------------------------------------------------ 

@group(0) @binding(0) var<storage, read> vertex_buffer: array<Vertex>;
@group(0) @binding(1) var<storage, read> view_buffer: array<View>;
@group(0) @binding(2) var global_sampler: sampler;
@group(0) @binding(3) var non_filtering_sampler: sampler;
@group(0) @binding(4) var clamped_sampler: sampler;
@group(0) @binding(5) var comparison_sampler: sampler_comparison;
@group(0) @binding(6) var<uniform> frame_info: FrameInfo;
@group(0) @binding(7) var<storage, read> entity_index_lookup: array<u32>;

// ------------------------------------------------------------------------------------
// Helper Functions
// ------------------------------------------------------------------------------------ 

fn mat4_from_scaling(scale: vec3f) -> mat4x4f {
    return mat4x4f(
        vec4f(scale.x, 0.0, 0.0, 0.0),
        vec4f(0.0, scale.y, 0.0, 0.0),
        vec4f(0.0, 0.0, scale.z, 0.0),
        vec4f(0.0, 0.0, 0.0, 1.0),
    );
}

fn get_entity_row(entity: u32) -> u32 {
    // row_field = (chunk_index << LOCAL_SLOT_BITS) | local_index 
    let entity_row = entity & ENTITY_ROW_MASK;
    return entity_index_lookup[entity_row];
}

fn cubemap_direction_to_uv(direction: vec3f) -> vec3f {
    let abs_dir = abs(direction);
    var layer: f32;
    var texcoord: vec2f;

    if (abs_dir.x > abs_dir.y && abs_dir.x > abs_dir.z) {
        layer = select(1.0, 0.0, direction.x > 0.0);
        texcoord = vec2f(-direction.z, direction.y) / abs_dir.x;
    } else if (abs_dir.y > abs_dir.z) {
        layer = select(3.0, 2.0, direction.y > 0.0);
        texcoord = vec2f(direction.x, -direction.z) / abs_dir.y;
    } else {
        layer = select(5.0, 4.0, direction.z > 0.0);
        texcoord = vec2f(direction.x, direction.y) / abs_dir.z;
    }
    
    // Flip the x coordinate for the positive faces
    if (layer % 2.0 == 0.0) {
        texcoord.x = -texcoord.x;
    }
    // Bottom face needs additional adjustment due to other face flips
    if (layer == 3.0) {
        texcoord.y = -texcoord.y;
        texcoord.x = -texcoord.x;
    }
    
    // Convert from [-1, 1] to [0, 1]
    texcoord = texcoord * 0.5 + 0.5;

    return vec3f(texcoord, layer);
}

fn random_seed(seed: u32) -> u32 {
    let x = seed * 1103515245u + 12345u;
    let y = x ^ (x >> 16u);
    return y * 2654435769u;
}

// Converts a 32-bit seed to a uniform float in the range [0,1).
fn rand_float(seed: u32) -> f32 {
  return f32(random_seed(seed)) * one_over_float_max;
}

fn dither_mask(uv: vec2f, resolution: vec2f) -> f32 {
    // Scale UV coordinates to the size of the screen
    let scaled_uv = uv * resolution;

    // Calculate the index in the Bayer matrix
    let x = u32(scaled_uv.x) % 4u;
    let y = u32(scaled_uv.y) % 4u;
    let index = y * 4u + x;

    // Return the dither value from the Bayer matrix
    return bayer_matrix[index];
}

fn approx(a: f32, b: f32) -> bool {
    return abs(a - b) <= epsilon; 
}

// get the max value between three values
fn max3(v: vec3f) -> f32 {
    return max(max(v.x, v.y), v.z);
}

fn isinf(x: f32) -> bool {
    return x == x && x != 0.0 && x * 2.0 == x;
}

// For vector types:
fn isinf3(v: vec3<f32>) -> vec3<bool> {
    return vec3<bool>(isinf(v.x), isinf(v.y), isinf(v.z));
}

// A helper function to compute the median of three values.
fn median3(a: precision_float, b: precision_float, c: precision_float) -> precision_float {
    // Sort the three values and pick the middle one
    // A simple way: median = a + b + c - min(a,b,c) - max(a,b,c)
    let min_val = min(a, min(b, c));

    let max_val = max(a, max(b, c));
    return (a + b + c) - min_val - max_val;
}

// Simple hash function
fn hash(x: u32) -> u32 {
    var y = x;
    y = y ^ (y >> u32(16));
    y = y * 0x85ebca6bu;
    y = y ^ (y >> u32(13));
    y = y * 0xc2b2ae35u;
    y = y ^ (y >> u32(16));
    return y;
}

// Convert uint to float in [0, 1) range
fn uint_to_normalized_float(x: u32) -> precision_float {
    return precision_float(f32(x) * one_over_float_max);
}

// Interpolate between two values
fn interpolate(v0: precision_float, v1: precision_float, t: precision_float) -> precision_float {
    return v0 * (1.0 - t) + v1 * t;
}

// Helper function to safely normalize a vector
fn safe_normalize(v: vec3f) -> vec3f {
  let len = length(v);
  return select(normalize(v), vec3f(0.0), len < 1e-6);
}

// A billboard function that works with local position and entity transform
// Uses model-view matrix manipulation for robust billboarding
fn billboard_vertex_local(uv: vec2f, entity_transform: mat4x4f) -> vec4f {
    // Get view and projection matrices
    let view_index = u32(frame_info.view_index);
    let view = view_buffer[view_index].view_matrix;
    // Extract translation from the entity transform (4th column)
    let world_position = vec3f(
        entity_transform[3][0],
        entity_transform[3][1],
        entity_transform[3][2]
    );
    // Extract scale from the entity transform
    // Scale is the magnitude of each of the first three column vectors
    let scale = vec3f(
        length(vec3f(entity_transform[0][0], entity_transform[0][1], entity_transform[0][2])),
        length(vec3f(entity_transform[1][0], entity_transform[1][1], entity_transform[1][2])),
        length(vec3f(entity_transform[2][0], entity_transform[2][1], entity_transform[2][2]))
    );
    // Calculate the billboard size - use the entity's scale
    // Using average of X and Y scale for consistent sizing
    let billboard_size = 0.6 * (scale.x + scale.y) * 0.5;
    // Calculate the vertex position in local space (centered quad)
    let billboard_local_pos = vec4f(
        (uv.x - 0.5) * billboard_size,
        (uv.y - 0.5) * billboard_size,
        0.0,
        1.0
    );
    // Transform back to world space using the inverse view matrix
    let inverse_view = mat4x4f(
        view[0][0], view[1][0], view[2][0], 0.0,
        view[0][1], view[1][1], view[2][1], 0.0,
        view[0][2], view[1][2], view[2][2], 0.0,
        0.0, 0.0, 0.0, 1.0
    );
    // Calculate the final world position
    let final_world_position = world_position + 
        inverse_view[0].xyz * billboard_local_pos.x +
        inverse_view[1].xyz * billboard_local_pos.y;
    
    return vec4f(final_world_position, 1.0);
}

fn log_depth(view_space_z: f32) -> f32 {
    let view_index = u32(frame_info.view_index);
    let far_plane = -view_buffer[view_index].frustum[5].w;
    let near_plane = -view_buffer[view_index].frustum[4].w;
    let z = -view_space_z;
    return
    (log(LOG_DEPTH_C * z + 1.0) - log(LOG_DEPTH_C * near_plane + 1.0)) 
        / (log(LOG_DEPTH_C * far_plane + 1.0) - log(LOG_DEPTH_C * near_plane + 1.0));
}

fn linearize_depth(d: f32, near_plane: f32, far_plane: f32, view_index: u32) -> f32 {
    // Works for both perspective and orthographic projections.
    // Depth coming from the texture is in the [-1,1] clip-space range.
    let depth01 = (d * 0.5) + 0.5;               // -> [0,1]

    // Orthographic matrices have projection[3][3] == 1, perspective == 0
    let proj_33    = view_buffer[view_index].projection_matrix[3][3];
    let is_ortho   = proj_33 > 0.5;

    // Perspective: z_eye = (n f) / (f – depth01·(f – n))
    let persp_z = (near_plane * far_plane) /
                  (far_plane - depth01 * (far_plane - near_plane));

    // Orthographic: z_eye = n + depth01·(f – n)
    let ortho_z = near_plane + depth01 * (far_plane - near_plane);

    // `select(a, b, cond)` chooses b when cond is true
    return select(persp_z, ortho_z, is_ortho);
}

fn rotate_hue(color: vec4f, hue_rotation: f32) -> vec4f {
    // Convert RGB to HSV
    let rgb = color.rgb;
    let max_val = max(max(rgb.r, rgb.g), rgb.b);
    let min_val = min(min(rgb.r, rgb.g), rgb.b);
    let delta = max_val - min_val;
    
    // Calculate hue
    var hue: f32 = 0.0;
    if (delta > 0.0) {
        // Use a formula that avoids branching for hue calculation
        let r_dist = select((rgb.g - rgb.b) / delta, 0.0, max_val == rgb.r);
        let g_dist = select((rgb.b - rgb.r) / delta, 0.0, max_val == rgb.g);
        let b_dist = select((rgb.r - rgb.g) / delta, 0.0, max_val == rgb.b);
        
        hue = fract((r_dist + 6.0 * select(1.0, 0.0, max_val == rgb.r) + 
                     g_dist + 2.0 * select(1.0, 0.0, max_val == rgb.g) + 
                     b_dist + 4.0 * select(1.0, 0.0, max_val == rgb.b)) / 6.0);
    }
    
    // Calculate saturation and value
    let saturation = select(0.0, delta / max_val, max_val == 0.0);
    let value = max_val;
    
    // Apply hue rotation
    hue = fract(hue + hue_rotation / (2.0 * 3.14159265359));
    
    // Convert back to RGB using a more efficient approach
    let hue_6 = hue * 6.0;
    let hue_sector = floor(hue_6);
    let hue_fract = hue_6 - hue_sector;
    
    // Calculate the RGB components using the HSV color wheel
    let p = value * (1.0 - saturation);
    let q = value * (1.0 - saturation * hue_fract);
    let t = value * (1.0 - saturation * (1.0 - hue_fract));
    
    // Create a lookup table for the RGB values based on the hue sector
    let sector_0 = vec3f(value, t, p);
    let sector_1 = vec3f(q, value, p);
    let sector_2 = vec3f(p, value, t);
    let sector_3 = vec3f(p, q, value);
    let sector_4 = vec3f(t, p, value);
    let sector_5 = vec3f(value, p, q);
    
    // Select the appropriate sector using dot products with a mask
    let sector_mask = vec3f(
        select(1.0, 0.0, hue_sector == 0.0 || hue_sector == 5.0),
        select(1.0, 0.0, hue_sector == 1.0 || hue_sector == 2.0),
        select(1.0, 0.0, hue_sector == 3.0 || hue_sector == 4.0)
    );
    
    let r = dot(vec3f(sector_0.x, sector_1.x, sector_2.x) * sector_mask, vec3f(1.0)) + 
            dot(vec3f(sector_3.x, sector_4.x, sector_5.x) * sector_mask, vec3f(1.0));
    let g = dot(vec3f(sector_0.y, sector_1.y, sector_2.y) * sector_mask, vec3f(1.0)) + 
            dot(vec3f(sector_3.y, sector_4.y, sector_5.y) * sector_mask, vec3f(1.0));
    let b = dot(vec3f(sector_0.z, sector_1.z, sector_2.z) * sector_mask, vec3f(1.0)) + 
            dot(vec3f(sector_3.z, sector_4.z, sector_5.z) * sector_mask, vec3f(1.0));
    
    return vec4f(r, g, b, color.a);
}

// Computes the inverse of a 4x4 matrix using Cramer's rule.
// Returns the inverse matrix. If the matrix is not invertible, the result is undefined.
fn inverse4x4(m: mat4x4<f32>) -> mat4x4<f32> {
    let m00 = m[0][0]; let m01 = m[0][1]; let m02 = m[0][2]; let m03 = m[0][3];
    let m10 = m[1][0]; let m11 = m[1][1]; let m12 = m[1][2]; let m13 = m[1][3];
    let m20 = m[2][0]; let m21 = m[2][1]; let m22 = m[2][2]; let m23 = m[2][3];
    let m30 = m[3][0]; let m31 = m[3][1]; let m32 = m[3][2]; let m33 = m[3][3];

    let coef00 = m22 * m33 - m32 * m23;
    let coef02 = m12 * m33 - m32 * m13;
    let coef03 = m12 * m23 - m22 * m13;

    let coef04 = m21 * m33 - m31 * m23;
    let coef06 = m11 * m33 - m31 * m13;
    let coef07 = m11 * m23 - m21 * m13;

    let coef08 = m21 * m32 - m31 * m22;
    let coef10 = m11 * m32 - m31 * m12;
    let coef11 = m11 * m22 - m21 * m12;

    let coef12 = m20 * m33 - m30 * m23;
    let coef14 = m10 * m33 - m30 * m13;
    let coef15 = m10 * m23 - m20 * m13;

    let coef16 = m20 * m32 - m30 * m22;
    let coef18 = m10 * m32 - m30 * m12;
    let coef19 = m10 * m22 - m20 * m12;

    let coef20 = m20 * m31 - m30 * m21;
    let coef22 = m10 * m31 - m30 * m11;
    let coef23 = m10 * m21 - m20 * m11;

    let fac0 = vec4<f32>(coef00, coef00, coef02, coef03);
    let fac1 = vec4<f32>(coef04, coef04, coef06, coef07);
    let fac2 = vec4<f32>(coef08, coef08, coef10, coef11);
    let fac3 = vec4<f32>(coef12, coef12, coef14, coef15);
    let fac4 = vec4<f32>(coef16, coef16, coef18, coef19);
    let fac5 = vec4<f32>(coef20, coef20, coef22, coef23);

    let v0 = vec4<f32>(m10, m00, m00, m00);
    let v1 = vec4<f32>(m11, m01, m01, m01);
    let v2 = vec4<f32>(m12, m02, m02, m02);
    let v3 = vec4<f32>(m13, m03, m03, m03);

    let inv0 =  v1 * fac0 - v2 * fac1 + v3 * fac2;
    let inv1 = -v0 * fac0 + v2 * fac3 - v3 * fac4;
    let inv2 =  v0 * fac1 - v1 * fac3 + v3 * fac5;
    let inv3 = -v0 * fac2 + v1 * fac4 - v2 * fac5;

    let sign_a = vec4<f32>( 1.0, -1.0,  1.0, -1.0);
    let sign_b = vec4<f32>(1.0, -1.0, 1.0, -1.0);

    let col1 = inv0 * sign_a;
    let col2 = inv1 * sign_b;
    let col3 = inv2 * sign_a;
    let col4 = inv3 * sign_b;

    let row0 = vec4<f32>(col1[0], col2[0], col3[0], col4[0]);
    let det = dot(vec4<f32>(m00, m01, m02, m03), row0);

    let inverse = mat4x4<f32>(
        col1 / det,
        col2 / det,
        col3 / det,
        col4 / det
    );

    return inverse;
}

fn mask_popcount(mask: vec4<u32>) -> u32 {
    let ones = countOneBits(mask);
    return ones.x + ones.y + ones.z + ones.w;
}