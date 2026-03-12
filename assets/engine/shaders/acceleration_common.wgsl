// ------------------------------------------------------------------------------------
// Defines
// ------------------------------------------------------------------------------------
#define BVH_TRAVERSAL_ORDER_CHILDREN

// ------------------------------------------------------------------------------------
// Constants
// ------------------------------------------------------------------------------------

const HPLOC_WAVE_SIZE = 128u;
const NODE_STACK_SIZE = 24;

// ------------------------------------------------------------------------------------
// Data Structures 
// ------------------------------------------------------------------------------------

// BVH info structure
struct BVHInfo {
    leaf_count: u32,
    bvh2_count: u32,
    prim_count: u32,
    prim_base: u32,
};

// World-space AABB with 4x f32 components for packing
struct AABB {
    min: vec4<f32>,
    max: vec4<f32>,
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
    prim_meshid_padding: vec4<f32>,
    ray_local: Ray,
    hit_triangle_data: vec4<u32>,
};

// Ray hit compact structure for intersection tests
struct RayHitCompact {
    t_hit: f32,
    prim_store: u32,
    mesh_id: u32,
    tri_id_local: u32,
    tri_indices: vec4<u32>,
    has_hit: u32,
};

// ------------------------------------------------------------------------------------
// Functions
// ------------------------------------------------------------------------------------

// Make a miss ray hit compact
fn make_miss_ray_hit_compact(t_max: f32) -> RayHitCompact {
    var result: RayHitCompact;
    result.t_hit = t_max;
    result.prim_store = 0xffffffffu;
    result.mesh_id = 0xffffffffu;
    result.tri_id_local = 0xffffffffu;
    result.tri_indices = vec4<u32>(0u, 0u, 0u, 0u);
    result.has_hit = 0u;
    return result;
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

    let final_min = min(result_min, result_max);
    let final_max = max(result_min, result_max);

    return AABB(vec4<f32>(final_min, node.min.w), vec4<f32>(final_max, node.max.w));
}

// Ray-AABB intersection (slab method)
fn intersect_aabb(ray: ptr<function, Ray>, min_point: vec3<f32>, max_point: vec3<f32>) -> vec2<f32> {
    let min_x = select(min_point.x, max_point.x, (*ray).inv_direction.x < 0.0);
    let max_x = select(max_point.x, min_point.x, (*ray).inv_direction.x < 0.0);
    let tx1 = (min_x - (*ray).origin_and_tmin.x) * (*ray).inv_direction.x;
    let tx2 = (max_x - (*ray).origin_and_tmin.x) * (*ray).inv_direction.x;

    let min_y = select(min_point.y, max_point.y, (*ray).inv_direction.y < 0.0);
    let max_y = select(max_point.y, min_point.y, (*ray).inv_direction.y < 0.0);
    let ty1 = (min_y - (*ray).origin_and_tmin.y) * (*ray).inv_direction.y;
    let ty2 = (max_y - (*ray).origin_and_tmin.y) * (*ray).inv_direction.y;

    let min_z = select(min_point.z, max_point.z, (*ray).inv_direction.z < 0.0);
    let max_z = select(max_point.z, min_point.z, (*ray).inv_direction.z < 0.0);
    let tz1 = (min_z - (*ray).origin_and_tmin.z) * (*ray).inv_direction.z;
    let tz2 = (max_z - (*ray).origin_and_tmin.z) * (*ray).inv_direction.z;

    let tmin = max((*ray).origin_and_tmin.w, max(min(tx1, tx2), max(min(ty1, ty2), min(tz1, tz2))));
    let tmax = min((*ray).direction_and_tmax.w, min(max(tx1, tx2), min(max(ty1, ty2), max(tz1, tz2))));
    return vec2<f32>(tmin, tmax);
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
fn intersect_triangle(ray: ptr<function, Ray>, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> f32 {
    let dir  = (*ray).direction_and_tmax.xyz;
    let orig = (*ray).origin_and_tmin.xyz;

    let plane_normal = normalize(cross(v1 - v0, v2 - v1));
    let t = dot(plane_normal, (v0 - orig) / dot(plane_normal, dir));
    let p = orig + dir * t;

    let n1 = normalize(cross(v1 - v0, p - v1));
    let n2 = normalize(cross(v2 - v1, p - v2));
    let n3 = normalize(cross(v0 - v2, p - v0));

    let d0 = dot(n1, n2);
    let d1 = dot(n2, n3);

    let threshold = 1.0 - 0.001;
    return select(-1.0, t, d0 > threshold && d1 > threshold);

    // let e1 = v1 - v0;
    // let e2 = v2 - v0;
    // let pvec = cross(dir, e2);
    // let det  = dot(e1, pvec);

    // let one_over_det = 1.0 / det;
    // let tvec = orig - v0;
    // let u = dot(tvec, pvec) * one_over_det;

    // let qvec = cross(tvec, e1);
    // let v = dot(dir, qvec) * one_over_det;

    // let t = dot(e2, qvec) * one_over_det;
    // let valid = abs(det) > 0.00001 && t > 0.0001 && u > 0.0 && v > 0.0 && u + v < 1.0;
    // return select(-1.0, t, valid); // Epsilon check
}

fn build_local_ray(
    ray_world: ptr<function, Ray>,
    model: mat4x4<f32>,
    transpose_inverse_model: mat4x4<f32>
) -> Ray {
    let ro_world = (*ray_world).origin_and_tmin.xyz;
    let rd_world = (*ray_world).direction_and_tmax.xyz;

    let t_col0 = transpose_inverse_model[0].xyz;
    let t_col1 = transpose_inverse_model[1].xyz;
    let t_col2 = transpose_inverse_model[2].xyz;

    let trans = model[3].xyz;
    let ro_rel = ro_world - trans;

    let rd_local = vec3<f32>(
        dot(rd_world, t_col0),
        dot(rd_world, t_col1),
        dot(rd_world, t_col2)
    );
    let ro_local = vec3<f32>(
        dot(ro_rel, t_col0),
        dot(ro_rel, t_col1),
        dot(ro_rel, t_col2)
    );

    var ray_local: Ray;
    ray_local.origin_and_tmin = vec4<f32>(ro_local, (*ray_world).origin_and_tmin.w);
    ray_local.direction_and_tmax = vec4<f32>(rd_local, (*ray_world).direction_and_tmax.w);

    let d = rd_local;
    ray_local.inv_direction = vec4<f32>(
        1.0 / max(abs(d.x), 1e-8) * select(1.0, -1.0, d.x < 0.0),
        1.0 / max(abs(d.y), 1e-8) * select(1.0, -1.0, d.y < 0.0),
        1.0 / max(abs(d.z), 1e-8) * select(1.0, -1.0, d.z < 0.0),
        0.0
    );
    return ray_local;
}
