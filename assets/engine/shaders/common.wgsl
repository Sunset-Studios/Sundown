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
    view_direction: vec4f,
    frustum: array<vec4f, 6>,
};

struct FrameInfo {
    view_index: u32,
};

struct EntityTransform {
    transform: mat4x4<f32>,
    inverse_model_matrix: mat4x4<f32>,
    transpose_inverse_model_matrix: mat4x4<f32>,
    bounds_pos_radius: vec4<f32>,
    bounds_extent_and_custom_scale: vec4<f32>,
};

struct ObjectInstance {
    batch: u32,
    entity: u32
};

struct CompactedObjectInstance {
    entity: u32,
};

@group(0) @binding(0) var<storage, read> vertex_buffer: array<Vertex>;
@group(0) @binding(1) var<storage, read> view_buffer: array<View>;
@group(0) @binding(2) var global_sampler: sampler;
@group(0) @binding(3) var non_filtering_sampler: sampler;
@group(0) @binding(4) var<uniform> frame_info: FrameInfo;

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