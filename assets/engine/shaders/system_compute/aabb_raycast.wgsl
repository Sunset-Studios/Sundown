#include "common.wgsl"

// Ray structure
struct Ray {
    origin_tmin: vec4<f32>,
    direction_tmax: vec4<f32>,
    inv_direction: vec4<f32>,
}

// Hit result structure
struct RaycastHit {
    user_data: u32,
    distance: f32,
    position: vec3<f32>,
    normal: vec3<f32>,
}

// Input uniforms
struct RaycastUniforms {
    ray_count: u32,
    max_distance: f32,
    find_closest: u32, // 1 if we should find closest hit, 0 if any hit is sufficient
    max_traversal_steps: u32,
}

// Bindings
@group(1) @binding(0) var<uniform> uniforms: RaycastUniforms;
@group(1) @binding(1) var<storage, read> aabb_bounds: array<AABBNodeBounds>;
@group(1) @binding(2) var<storage, read> aabb_nodes: array<AABBTreeNode>;
@group(1) @binding(3) var<storage, read> input_rays: array<Ray>;
@group(1) @binding(4) var<storage, read_write> output_hits: array<RaycastHit>;

// Epsilon to avoid precision issues
const EPSILON: f32 = 0.0001;

// Check if ray intersects AABB, returns distance to intersection or -1 if no intersection
fn ray_aabb_intersection(ray: Ray, min_point: vec3<f32>, max_point: vec3<f32>) -> f32 {
    var tmin = ray.origin_tmin.w;
    var tmax = ray.direction_tmax.w;
    
    for (var i = 0; i < 3; i++) {
        let inv_d = ray.inv_direction[i];
        var t1 = (min_point[i] - ray.origin_tmin[i]) * inv_d;
        var t2 = (max_point[i] - ray.origin_tmin[i]) * inv_d;
        
        if (inv_d < 0.0) {
            let temp = t1;
            t1 = t2;
            t2 = temp;
        }
        
        tmin = max(tmin, t1);
        tmax = min(tmax, t2);
        
        if (tmax < tmin) {
            return -1.0;
        }
    }
    
    return tmin;
}

// Calculate face normal from hit point and AABB
fn calculate_face_normal(hit_point: vec3<f32>, min_point: vec3<f32>, max_point: vec3<f32>) -> vec3<f32> {
    // Find the closest face by comparing distances to each face
    let distances = array<f32, 6>(
        abs(hit_point.x - min_point.x), // -X face
        abs(hit_point.x - max_point.x), // +X face
        abs(hit_point.y - min_point.y), // -Y face
        abs(hit_point.y - max_point.y), // +Y face
        abs(hit_point.z - min_point.z), // -Z face
        abs(hit_point.z - max_point.z)  // +Z face
    );
    
    var min_distance = distances[0];
    var min_index = 0;
    
    for (var i = 1; i < 6; i++) {
        if (distances[i] < min_distance) {
            min_distance = distances[i];
            min_index = i;
        }
    }
    
    // Return the normal based on the closest face
    switch (min_index) {
        case 0: { return vec3<f32>(-1.0, 0.0, 0.0); }
        case 1: { return vec3<f32>(1.0, 0.0, 0.0); }
        case 2: { return vec3<f32>(0.0, -1.0, 0.0); }
        case 3: { return vec3<f32>(0.0, 1.0, 0.0); }
        case 4: { return vec3<f32>(0.0, 0.0, -1.0); }
        case 5: { return vec3<f32>(0.0, 0.0, 1.0); }
        default: { return vec3<f32>(0.0, 0.0, 0.0); }
    }
}

// Main compute shader
@compute @workgroup_size(64, 1, 1)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ray_index = global_id.x;
    
    // Check if ray index is in range
    if (ray_index >= uniforms.ray_count) {
        return;
    }
    
    // Get input ray
    var ray = input_rays[ray_index];
    
    // Initialize hit result
    var hit: RaycastHit;
    hit.user_data = 0u;
    hit.distance = uniforms.max_distance;
    hit.position = vec3<f32>(0.0, 0.0, 0.0);
    hit.normal = vec3<f32>(0.0, 0.0, 0.0);
    
    // First check if ray hits root node
    let root_node = aabb_bounds[0];
    let t_root = ray_aabb_intersection(ray, root_node.min_point.xyz, root_node.max_point.xyz);
    
    if (t_root < 0.0) {
        // Ray doesn't hit the root, early out
        output_hits[ray_index] = hit;
        return;
    }
    
    // We'll use a stack-based traversal to avoid recursion
    var stack: array<u32, 64>; // Stack of node indices
    var stack_ptr = 0u;
    
    // Start at the root
    stack[stack_ptr] = 0u;
    stack_ptr += 1u;
    
    var traversal_steps = 0u;
    
    // Traverse the AABB tree
    while (stack_ptr > 0u && traversal_steps < uniforms.max_traversal_steps) {
        traversal_steps += 1u;
        stack_ptr -= 1u;
        let node_index = stack[stack_ptr];
        let node = aabb_nodes[node_index];
        let bounds = aabb_bounds[node_index];
        
        // Skip free nodes
        if ((u32(node.flags_and_node_data.x) & AABB_NODE_FLAGS_FREE) != 0u) {
            continue;
        }
        
        // Check intersection with this node's AABB
        let t = ray_aabb_intersection(ray, bounds.min_point.xyz, bounds.max_point.xyz);
        
        if (t < 0.0 || t > hit.distance) {
            // No intersection or intersection is farther than current hit
            continue;
        }
        
        if (u32(node.flags_and_node_data.y) == AABB_NODE_TYPE_LEAF) {
            // This is a leaf node, perform detailed intersection test with entity
            // For simplicity, we'll use the AABB test as the entity intersection
            
            // Calculate hit point
            let hit_point = ray.origin_tmin.xyz + ray.direction_tmax.xyz * t;
            
            // Calculate normal
            let normal = calculate_face_normal(hit_point, bounds.min_point.xyz, bounds.max_point.xyz);
            
            // Update hit if this is closer
            if (t < hit.distance) {
                hit.user_data = u32(node.left_right_parent_ud.w);  // user_data
                hit.distance = t;
                hit.position = hit_point;
                hit.normal = normal;
                
                // If we're not looking for the closest hit, we can stop now
                if (uniforms.find_closest == 0u) {
                    break;
                }
            }
        } else {
            // This is an internal node, add children to stack
            let left_child = u32(node.left_right_parent_ud.x);  // left
            let right_child = u32(node.left_right_parent_ud.y); // right
            
            // Add children to stack (right then left for front-to-back traversal)
            // We handle invalid child indices (0) by checking in the next iteration
            if (left_child != 0u && right_child != 0u) {
                // Both children exist, determine the traversal order
                let left_bounds = aabb_bounds[left_child];
                let right_bounds = aabb_bounds[right_child];
                
                let t_left = ray_aabb_intersection(ray, left_bounds.min_point.xyz, left_bounds.max_point.xyz);
                let t_right = ray_aabb_intersection(ray, right_bounds.min_point.xyz, right_bounds.max_point.xyz);
                
                // Add nodes to stack in the order of closest intersection first
                if (t_left > t_right) {
                    // Right is closer
                    if (t_left >= 0.0 && stack_ptr < 64u) {
                        stack[stack_ptr] = left_child;
                        stack_ptr += 1u;
                    }
                    if (t_right >= 0.0 && stack_ptr < 64u) {
                        stack[stack_ptr] = right_child;
                        stack_ptr += 1u;
                    }
                } else {
                    // Left is closer
                    if (t_right >= 0.0 && stack_ptr < 64u) {
                        stack[stack_ptr] = right_child;
                        stack_ptr += 1u;
                    }
                    if (t_left >= 0.0 && stack_ptr < 64u) {
                        stack[stack_ptr] = left_child;
                        stack_ptr += 1u;
                    }
                }
            } else {
                // Only one child exists, add it if valid
                if (left_child != 0u && stack_ptr < 64u) {
                    stack[stack_ptr] = left_child;
                    stack_ptr += 1u;
                }
                if (right_child != 0u && stack_ptr < 64u) {
                    stack[stack_ptr] = right_child;
                    stack_ptr += 1u;
                }
            }
        }
    }
    
    // Write hit result
    output_hits[ray_index] = hit;
} 