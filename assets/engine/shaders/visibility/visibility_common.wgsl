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

struct MeshletDrawCommandNoAtomics {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

struct MeshletInstance {
    object_instance_index: u32,
    meshlet_index: u32,
};

const MESHLET_TRIANGLE_BITS: u32 = 8u;
const MESHLET_TRIANGLE_MASK: u32 = 0xffu;
const MESHLET_EPSILON: f32 = 0.00001;

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

fn interpolate_vec2(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, bary: vec3<f32>) -> vec2<f32> {
    return a * bary.x + b * bary.y + c * bary.z;
}

fn interpolate_vec3(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, bary: vec3<f32>) -> vec3<f32> {
    return a * bary.x + b * bary.y + c * bary.z;
}

fn edge_function(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> f32 {
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

fn calc_full_barycentric(
    pixel_uv: vec2<f32>,
    clip0: vec4<f32>,
    clip1: vec4<f32>,
    clip2: vec4<f32>
) -> vec3<f32> {
    let ndc0 = clip0.xy / clip0.w;
    let ndc1 = clip1.xy / clip1.w;
    let ndc2 = clip2.xy / clip2.w;

    let pixel_ndc = vec2<f32>(
        pixel_uv.x * 2.0 - 1.0,
        (1.0 - pixel_uv.y) * 2.0 - 1.0
    );

    let area = edge_function(ndc0, ndc1, ndc2);
    let safe_area = select(area, MESHLET_EPSILON, abs(area) < MESHLET_EPSILON);
    let bary_screen = vec3<f32>(
        edge_function(ndc1, ndc2, pixel_ndc),
        edge_function(ndc2, ndc0, pixel_ndc),
        edge_function(ndc0, ndc1, pixel_ndc)
    ) / safe_area;

    let inv_w = vec3<f32>(1.0 / clip0.w, 1.0 / clip1.w, 1.0 / clip2.w);
    let bary_perspective = bary_screen * inv_w;
    let bary_sum = bary_perspective.x + bary_perspective.y + bary_perspective.z;
    let safe_sum = select(bary_sum, MESHLET_EPSILON, abs(bary_sum) < MESHLET_EPSILON);
    return bary_perspective / safe_sum;
}
