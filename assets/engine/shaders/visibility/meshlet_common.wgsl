struct MeshletRecord {
    vertex_offset: u32,
    vertex_count: u32,
    triangle_offset: u32,
    triangle_count: u32,
    center_radius: vec4<f32>,
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
    normal_cone: vec4<f32>,
};

struct MeshletDrawCommand {
    vertex_count: u32,
    instance_count: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
};

struct MeshletInstance {
    object_instance_index: u32,
    meshlet_index: u32,
};

const MESHLET_TRIANGLE_BITS: u32 = 8u;
const MESHLET_TRIANGLE_MASK: u32 = 0xffu;

fn meshlet_object_index(entry: vec4<u32>) -> u32 {
    return entry.x;
}

fn meshlet_index(entry: vec4<u32>) -> u32 {
    return entry.y;
}

fn transform_max_scale(transform: mat4x4<f32>) -> f32 {
    return max(
        max(length(transform[0].xyz), length(transform[1].xyz)),
        length(transform[2].xyz)
    );
}

fn pack_surface_id(meshlet_index_value: u32, triangle_index: u32) -> u32 {
    return (meshlet_index_value << MESHLET_TRIANGLE_BITS) | (triangle_index & MESHLET_TRIANGLE_MASK);
}

fn unpack_surface_meshlet(surface: u32) -> u32 {
    return surface >> MESHLET_TRIANGLE_BITS;
}

fn unpack_surface_triangle(surface: u32) -> u32 {
    return surface & MESHLET_TRIANGLE_MASK;
}
