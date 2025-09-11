// ------------------------------------------------------------------------------------
// Constants
// ------------------------------------------------------------------------------------

const HPLOC_WAVE_SIZE = 128u;
const QUANT_BITS = 10u;
const QUANT_MAX = 1023u;

// ------------------------------------------------------------------------------------
// Data Structures 
// ------------------------------------------------------------------------------------

// World-space AABB with 4x f32 components for packing
struct AABB {
    min: vec4<f32>,
    max: vec4<f32>,
};

// BVH4 (4-wide) node
struct BVH4Node {
    min: vec4<f32>,
    max: vec4<f32>,
    children: vec4<f32>,
};



// Ray structure for intersection tests
struct Ray {
    origin_and_tmin: vec4<f32>,
    direction_and_tmax: vec4<f32>,
    inv_direction: vec4<f32>,
}

// Ray hit structure for intersection tests
struct RayHit {
    position_and_t: vec4<f32>,
    normal_and_user_data: vec4<f32>,
};

// ------------------------------------------------------------------------------------
// Functions
// ------------------------------------------------------------------------------------

// Packing and unpacking quantised AABBs
fn pack_quant3(x: u32, y: u32, z: u32) -> u32 {
    return (x & 0x3ffu) | ((y & 0x3ffu) << 10u) | ((z & 0x3ffu) << 20u);
}

// Unpacking quantised AABBs
fn unpack_quant3(packed: u32) -> vec3<u32> {
    let x = packed & 0x3ffu;
    let y = (packed >> 10u) & 0x3ffu;
    let z = (packed >> 20u) & 0x3ffu;
    return vec3<u32>(x, y, z);
}

// Decoding quantised AABBs
fn decode_quant_aabb(base_min: vec3<f32>, base_extent: vec3<f32>, qmin: u32, qmax: u32) -> AABB {
    let qmin_v = vec3<f32>(unpack_quant3(qmin)) / f32(QUANT_MAX);
    let qmax_v = vec3<f32>(unpack_quant3(qmax)) / f32(QUANT_MAX);
    return AABB(
        vec4<f32>(base_min + base_extent * qmin_v, 0.0),
        vec4<f32>(base_min + base_extent * qmax_v, 0.0),
    );
}

// Check if a node is a leaf
fn is_leaf(node: AABB) -> bool {
    return node.min.w >= 0.0 && node.max.w < 0.0;
}

// Check if a node is valid
fn is_valid_node(node: AABB) -> bool {
    return node.min.w >= 0.0;
}

// Transform an AABB - properly handles rotation/scaling by transforming all 8 corners
fn transform_aabb(node: AABB, transform: mat4x4<f32>) -> AABB {
    // Define all 8 corners of the AABB
    let min_pt = node.min.xyz;
    let max_pt = node.max.xyz;
    
    // Transform all 8 corners of the bounding box
    let corner_0 = (transform * vec4<f32>(min_pt.x, min_pt.y, min_pt.z, 1.0)).xyz;
    let corner_1 = (transform * vec4<f32>(max_pt.x, min_pt.y, min_pt.z, 1.0)).xyz;
    let corner_2 = (transform * vec4<f32>(min_pt.x, max_pt.y, min_pt.z, 1.0)).xyz;
    let corner_3 = (transform * vec4<f32>(max_pt.x, max_pt.y, min_pt.z, 1.0)).xyz;
    let corner_4 = (transform * vec4<f32>(min_pt.x, min_pt.y, max_pt.z, 1.0)).xyz;
    let corner_5 = (transform * vec4<f32>(max_pt.x, min_pt.y, max_pt.z, 1.0)).xyz;
    let corner_6 = (transform * vec4<f32>(min_pt.x, max_pt.y, max_pt.z, 1.0)).xyz;
    let corner_7 = (transform * vec4<f32>(max_pt.x, max_pt.y, max_pt.z, 1.0)).xyz;
    
    // Find actual min/max from all transformed corners
    var result_min = corner_0;
    var result_max = corner_0;
    
    result_min = min(result_min, corner_1);
    result_max = max(result_max, corner_1);
    result_min = min(result_min, corner_2);
    result_max = max(result_max, corner_2);
    result_min = min(result_min, corner_3);
    result_max = max(result_max, corner_3);
    result_min = min(result_min, corner_4);
    result_max = max(result_max, corner_4);
    result_min = min(result_min, corner_5);
    result_max = max(result_max, corner_5);
    result_min = min(result_min, corner_6);
    result_max = max(result_max, corner_6);
    result_min = min(result_min, corner_7);
    result_max = max(result_max, corner_7);

    return AABB(vec4<f32>(result_min, node.min.w), vec4<f32>(result_max, node.max.w));
}

// Ray-AABB intersection (slab method)
fn intersect_aabb(ray: Ray, min_point: vec3<f32>, max_point: vec3<f32>) -> f32 {
    var tmin = ray.origin_and_tmin.w;
    var tmax = ray.direction_and_tmax.w;
    for (var i = 0; i < 3; i++) {
        let inv_d = ray.inv_direction.xyz[i];
        var t1 = (min_point[i] - ray.origin_and_tmin.xyz[i]) * inv_d;
        var t2 = (max_point[i] - ray.origin_and_tmin.xyz[i]) * inv_d;
        if (inv_d < 0.0) {
            let temp = t1;
            t1 = t2;
            t2 = temp;
        }
        tmin = max(tmin, t1);
        tmax = min(tmax, t2);
    }
    return select(-1.0, tmin, tmin <= tmax);
}

// Calculate the surface area of an AABB
fn calculate_aabb_surface_area(min_point: vec3<f32>, max_point: vec3<f32>) -> f32 {
    let size = max(vec3(0.0), max_point - min_point);
    return 2.0 * (size.x * size.y + size.x * size.z + size.y * size.z);
}

// Merge two AABBs
fn merge_aabbs(a_min: vec3<f32>, a_max: vec3<f32>, b_min: vec3<f32>, b_max: vec3<f32>) -> AABB {
    let merged_min = vec4<f32>(
        min(a_min.x, b_min.x),
        min(a_min.y, b_min.y),
        min(a_min.z, b_min.z),
        0.0
    );
    let merged_max = vec4<f32>(
        max(a_max.x, b_max.x),
        max(a_max.y, b_max.y),
        max(a_max.z, b_max.z),
        0.0
    );
    return AABB(merged_min, merged_max);
}

// Intersection with a triangle
fn intersect_triangle(ray: Ray, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> f32 {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let dir  = ray.direction_and_tmax.xyz;
    let orig = ray.origin_and_tmin.xyz;

    let pvec = cross(dir, e2);
    let det  = dot(e1, pvec);
    if (abs(det) < 0.00001) { return -1.0; }

    let inv_det = 1.0 / det;
    let tvec = orig - v0;
    let u = dot(tvec, pvec) * inv_det;
    if (u < 0.0 || u > 1.0) { return -1.0; }

    let qvec = cross(tvec, e1);
    let v = dot(dir, qvec) * inv_det;
    if (v < 0.0 || u + v > 1.0) { return -1.0; }

    let t = dot(e2, qvec) * inv_det;
    return select(-1.0, t, t > 0.0001); // Epsilon check
}