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
};

// Ray hit compact structure for intersection tests
struct RayHitCompact {
    t_hit: f32,
    prim_store: u32,
    mesh_id: u32,
    tri_id_local: u32,
    tri_indices: vec4<u32>,
    barycentrics: vec2<f32>,
    has_hit: u32,
};

// ------------------------------------------------------------------------------------
// Functions
// ------------------------------------------------------------------------------------

// Make a miss ray hit compact
fn make_miss_ray_hit_compact(t_max: f32) -> RayHitCompact {
    var result: RayHitCompact;
    result.t_hit = t_max;
    result.barycentrics = vec2<f32>(0.0);
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

    // The endpoints above are already ordered by the inverse-direction sign:
    // t*1 is the near plane and t*2 is the far plane on every axis. Avoid six
    // redundant min/max operations for every BVH node visited by every ray.
    let tmin = max(
        (*ray).origin_and_tmin.w,
        max(tx1, max(ty1, tz1))
    );
    let tmax = min(
        (*ray).direction_and_tmax.w,
        min(tx2, min(ty2, tz2))
    );
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
fn intersect_triangle(ray: ptr<function, Ray>, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> vec3<f32> {
    let dir  = (*ray).direction_and_tmax.xyz;
    let orig = (*ray).origin_and_tmin.xyz;

    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let pvec = cross(dir, e2);
    let det  = dot(e1, pvec);
    let tvec = orig - v0;
    let qvec = cross(tvec, e1);

    // Keep the determinant out of the denominator until the triangle is known to be a hit.
    // Most BVH leaf candidates miss, so numerator-space rejection avoids their divide and the
    // three reciprocal-dependent multiplies while preserving double-sided winding behavior.
    let det_abs = abs(det);
    let det_sign = select(-1.0, 1.0, det > 0.0);
    let u_scaled = dot(tvec, pvec) * det_sign;
    let v_scaled = dot(dir, qvec) * det_sign;
    let t_scaled = dot(e2, qvec) * det_sign;
    let valid = det_abs > 0.00001
        && t_scaled > 0.0001 * det_abs
        && u_scaled > 0.0
        && v_scaled > 0.0
        && u_scaled + v_scaled < det_abs;

    if (!valid) {
        return vec3<f32>(-1.0, 0.0, 0.0);
    }
    let inverse_determinant = 1.0 / det_abs;
    return vec3<f32>(
        t_scaled * inverse_determinant,
        u_scaled * inverse_determinant,
        v_scaled * inverse_determinant
    );
}

// Closest-hit variant for radiance rays emitted from a surface hemisphere.
// Positive determinant is the front-facing winding for the Moller-Trumbore
// formulation used above. Rejecting the opposite winding lets traversal
// continue instead of shading the non-rendered side of a triangle.
fn intersect_triangle_front_face(
    ray: ptr<function, Ray>,
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>
) -> vec3<f32> {
    let direction = (*ray).direction_and_tmax.xyz;
    let origin = (*ray).origin_and_tmin.xyz;
    let edge_1 = v1 - v0;
    let edge_2 = v2 - v0;
    let p_vector = cross(direction, edge_2);
    let determinant = dot(edge_1, p_vector);
    let t_vector = origin - v0;
    let q_vector = cross(t_vector, edge_1);
    let u_scaled = dot(t_vector, p_vector);
    let v_scaled = dot(direction, q_vector);
    let t_scaled = dot(edge_2, q_vector);

    if (
        determinant <= 0.00001 ||
        t_scaled <= 0.0001 * determinant ||
        u_scaled <= 0.0 ||
        v_scaled <= 0.0 ||
        u_scaled + v_scaled >= determinant
    ) {
        return vec3<f32>(-1.0, 0.0, 0.0);
    }
    let inverse_determinant = 1.0 / determinant;
    return vec3<f32>(
        t_scaled * inverse_determinant,
        u_scaled * inverse_determinant,
        v_scaled * inverse_determinant
    );
}

// Any-hit traversal only needs interval membership. Preserve the exact
// triangle and distance tests while avoiding barycentric result construction
// and its two reciprocal-dependent multiplies on accepted candidates.
fn intersect_triangle_any(
    ray: ptr<function, Ray>,
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>
) -> bool {
    let direction = (*ray).direction_and_tmax.xyz;
    let origin = (*ray).origin_and_tmin.xyz;
    let edge_1 = v1 - v0;
    let edge_2 = v2 - v0;
    let p_vector = cross(direction, edge_2);
    let determinant = dot(edge_1, p_vector);
    let t_vector = origin - v0;
    let q_vector = cross(t_vector, edge_1);
    let determinant_absolute = abs(determinant);
    let determinant_sign = select(-1.0, 1.0, determinant > 0.0);
    let u_scaled = dot(t_vector, p_vector) * determinant_sign;
    let v_scaled = dot(direction, q_vector) * determinant_sign;
    let t_scaled = dot(edge_2, q_vector) * determinant_sign;

    if (
        determinant_absolute <= 0.00001 ||
        t_scaled <= 0.0001 * determinant_absolute ||
        u_scaled <= 0.0 ||
        v_scaled <= 0.0 ||
        u_scaled + v_scaled >= determinant_absolute
    ) {
        return false;
    }
    let inverse_determinant = 1.0 / determinant_absolute;
    let t_hit = t_scaled * inverse_determinant;
    return t_hit >= (*ray).origin_and_tmin.w &&
        t_hit < (*ray).direction_and_tmax.w;
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

fn build_local_ray_from_instance(
    ray_world: ptr<function, Ray>,
    instance_transform: RayInstanceTransform
) -> Ray {
    let ro_world = (*ray_world).origin_and_tmin.xyz;
    let rd_world = (*ray_world).direction_and_tmax.xyz;

    let t_col0 = instance_transform.world_to_local0.xyz;
    let t_col1 = instance_transform.world_to_local1.xyz;
    let t_col2 = instance_transform.world_to_local2.xyz;
    let trans = instance_transform.local_to_world3.xyz;
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

fn transform_local_point_from_instance(
    instance_transform: RayInstanceTransform,
    point_local: vec3<f32>
) -> vec3<f32> {
    return
        instance_transform.local_to_world0.xyz * point_local.x +
        instance_transform.local_to_world1.xyz * point_local.y +
        instance_transform.local_to_world2.xyz * point_local.z +
        instance_transform.local_to_world3.xyz;
}

fn transform_local_direction_from_instance(
    instance_transform: RayInstanceTransform,
    direction_local: vec3<f32>
) -> vec3<f32> {
    return
        instance_transform.world_to_local0.xyz * direction_local.x +
        instance_transform.world_to_local1.xyz * direction_local.y +
        instance_transform.world_to_local2.xyz * direction_local.z;
}
