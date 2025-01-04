struct Vertex {
    position: vec4f,
    normal: vec4f,
    color: vec4f,
    uv: vec2f,
    tangent: vec4f,
    bitangent: vec4f,
};

struct View {
    view_matrix: mat4x4f,
    prev_view_matrix: mat4x4f,
    projection_matrix: mat4x4f,
    prev_projection_matrix: mat4x4f,
    view_projection_matrix: mat4x4f,
    inverse_view_projection_matrix: mat4x4f,
    view_position: vec4f,
    view_direction: vec4f,
    frustum: array<vec4f, 6>,
};

struct FrameInfo {
    view_index: u32,
    time: f32,
    resolution: vec2f,
    cursor_world_position: vec4f,
};

struct EntityTransform {
    transform: mat4x4f,
    prev_transform: mat4x4f,
    inverse_model_matrix: mat4x4f,
    transpose_inverse_model_matrix: mat4x4f,
};

struct EntityMetadata {
    offset: u32,
    count: u32,
};

struct EntityBoundsData {
    bounds_pos_radius: vec4f,
    bounds_extent_and_custom_scale: vec4f,
};

struct ObjectInstance {
    batch: u32,
    entity: u32
};

struct CompactedObjectInstance {
    entity: u32,
    base_instance: u32,
};

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

const epsilon = 1e-5;
const world_up = vec3f(0.0, 1.0, 0.0);

@group(0) @binding(0) var<storage, read> vertex_buffer: array<Vertex>;
@group(0) @binding(1) var<storage, read> view_buffer: array<View>;
@group(0) @binding(2) var global_sampler: sampler;
@group(0) @binding(3) var non_filtering_sampler: sampler;
@group(0) @binding(4) var<uniform> frame_info: FrameInfo;
@group(0) @binding(5) var<storage, read> entity_metadata: array<EntityMetadata>;

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
    return abs(a - b) <= select(abs(b), abs(a), abs(a) < abs(b)) * epsilon; 
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
fn median3(a: f32, b: f32, c: f32) -> f32 {
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
const one_over_float_max = 1.0 / 4294967296.0;
fn uint_to_normalized_float(x: u32) -> f32 {
    return f32(x) * one_over_float_max;
}