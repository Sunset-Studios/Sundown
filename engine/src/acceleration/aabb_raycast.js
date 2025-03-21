import { AABB, AABB_NODE_TYPE, AABB_NODE_FLAGS } from "./aabb.js";
import { profile_scope } from "../utility/performance.js";

const EPSILON = 0.0001;
const raycast_scope = "AABBRaycast.raycast";

export class Ray {
    constructor(origin, direction) {
        this.origin = origin ? [...origin] : [0, 0, 0];
        
        if (direction) {
            this.set_direction(direction);
        } else {
            this.direction = [0, 0, 1]; // Default forward
            this.inv_direction = [0, 0, 1];
        }
    }
    
    set_direction(direction) {
        // Normalize direction
        const dir_length = Math.sqrt(
            direction[0] * direction[0] + 
            direction[1] * direction[1] + 
            direction[2] * direction[2]
        );
        const one_over_dir_length = 1.0 / dir_length;
        
        if (dir_length < EPSILON) {
            this.direction = [0, 0, 0];
            this.inv_direction = [0, 0, 0];
        } else {
            this.direction = [
                direction[0] * one_over_dir_length,
                direction[1] * one_over_dir_length,
                direction[2] * one_over_dir_length
            ];
            
            // Pre-compute inverse direction for AABB tests
            this.inv_direction = [
                Math.abs(this.direction[0]) < EPSILON ? Infinity : 1.0 / this.direction[0],
                Math.abs(this.direction[1]) < EPSILON ? Infinity : 1.0 / this.direction[1],
                Math.abs(this.direction[2]) < EPSILON ? Infinity : 1.0 / this.direction[2]
            ];
        }
    }
    
    point_at(t) {
        return [
            this.origin[0] + this.direction[0] * t,
            this.origin[1] + this.direction[1] * t,
            this.origin[2] + this.direction[2] * t
        ];
    }
}

export class RaycastHit {
    constructor() {
        this.user_data = 0;
        this.distance = Infinity;
        this.point = [0, 0, 0];
        this.normal = [0, 0, 0];
    }
    
    reset() {
        this.user_data = 0;
        this.distance = Infinity;
        this.point = [0, 0, 0];
        this.normal = [0, 0, 0];
    }
}

export class AABBRaycast {
    // Options for controlling raycast behavior
    static default_options = {
        max_distance: Infinity,
        sort_results: true,
        return_face_normal: true,
        first_hit_only: false
    };
    
    /**
     * Cast a ray into the scene and return the first hit
     * @param {Ray} ray - The ray to cast
     * @param {Object} options - Options for the raycast
     * @returns {Promise<RaycastHit|null>} - Promise that resolves to the hit information or null if no hit
     */
    static raycast(ray, options = {}, callback = null) {
        const opts = {...this.default_options, ...options};
        
        const hit = new RaycastHit();
        
        profile_scope(raycast_scope, () => {
            if (!AABB.data || AABB.root_node === 0) {
                return null;
            }
            
            // Get the root node
            const root_node = AABB.get_node_data(AABB.root_node);
            
            if (!this._intersect_aabb(ray, root_node, 0, opts.max_distance)) {
                return null;
            }
            
            // Pass an empty Set for cycle detection
            const visited_nodes = new Set();
            this._traverse_tree(ray, AABB.root_node, hit, opts, visited_nodes);
        });
        
        if (callback) {
            callback(hit.user_data !== 0 ? hit : null);
        }
    }
    
    /**
     * Cast a ray into the scene and return all hits
     * @param {Ray} ray - The ray to cast
     * @param {Object} options - Options for the raycast
     * @returns {Promise<RaycastHit[]>} - Promise that resolves to array of hits
     */
    static raycast_all(ray, options = {}, callback = null) {
        const opts = {...this.default_options, ...options};
        const hits = [];
        
        profile_scope(raycast_scope, () => {
            if (!AABB.data || AABB.root_node === 0) {
                return [];
            }
            
            // Get the root node
            const root_node = AABB.get_node_data(AABB.root_node);
            
            if (!this._intersect_aabb(ray, root_node, 0, opts.max_distance)) {
                return [];
            }
            
            // Pass an empty Set for cycle detection
            const visited_nodes = new Set();
            this._traverse_tree_all(ray, AABB.root_node, hits, opts, visited_nodes);
            
            // Sort hits by distance if requested
            if (opts.sort_results && hits.length > 1) {
                hits.sort((a, b) => a.distance - b.distance);
            }
        });

        if (callback) {
            callback(hits);
        }
    }
    
    /**
     * Traverse the AABB tree for the closest hit
     * @param {Ray} ray - The ray to cast
     * @param {number} node_index - Index of the current node
     * @param {RaycastHit} hit - Current hit information
     * @param {Object} options - Raycast options
     * @param {Set} visited_nodes - Set of already visited node indices to detect cycles
     */
    static _traverse_tree(ray, node_index, hit, options, visited_nodes) {
        // Check for cycles
        if (visited_nodes.has(node_index)) {
            return;
        }
        
        // Add this node to visited set
        visited_nodes.add(node_index);
        
        const node = AABB.get_node_data(node_index);
        
        // Skip free nodes
        if ((node.flags & AABB_NODE_FLAGS.FREE) != 0) {
            return;
        }
        
        // If this is a leaf node, test for intersection with the entity
        if (node.node_type === AABB_NODE_TYPE.LEAF) {
            const user_data = node.user_data;
            
            // Test intersection with the entity's geometry
            const result = this._intersect_node(ray, node_index, options);
            
            if (result && result.distance < hit.distance) {
                hit.user_data = user_data;
                hit.distance = result.distance;
                hit.point = result.point;
                hit.normal = result.normal;
            }
            
            if (options.first_hit_only) {
                return;
            }
        }
        
        // This is an internal node - test children
        const left_child = node.left;
        const right_child = node.right;
        
        let left_t_min = Infinity;
        let right_t_min = Infinity;
        
        // Test left child
        if (left_child !== 0) {
            const left_node = AABB.get_node_data(left_child);
            left_t_min = this._intersect_aabb(ray, left_node, 0, hit.distance);
        }
        
        // Test right child
        if (right_child !== 0) {
            const right_node = AABB.get_node_data(right_child);
            right_t_min = this._intersect_aabb(ray, right_node, 0, hit.distance);
        }
        
        // Traverse in order of closest intersection
        if (left_t_min < right_t_min) {
            // Left is closer, test it first
            if (left_t_min !== false) {
                this._traverse_tree(ray, left_child, hit, options, visited_nodes);
            }
            
            // Only test right if it's potentially closer than current hit
            if (right_t_min !== false && right_t_min < hit.distance) {
                this._traverse_tree(ray, right_child, hit, options, visited_nodes);
            }
        } else {
            // Right is closer, test it first
            if (right_t_min !== false) {
                this._traverse_tree(ray, right_child, hit, options, visited_nodes);
            }
            
            // Only test left if it's potentially closer than current hit
            if (left_t_min !== false && left_t_min < hit.distance) {
                this._traverse_tree(ray, left_child, hit, options, visited_nodes);
            }
        }
    }
    
    /**
     * Traverse the AABB tree for all hits
     * @param {Ray} ray - The ray to cast
     * @param {number} node_index - Index of the current node
     * @param {Array} hits - Array to collect hits
     * @param {Object} options - Raycast options
     * @param {Set} visited_nodes - Set of already visited node indices to detect cycles
     */
    static _traverse_tree_all(ray, node_index, hits, options, visited_nodes) {
        // Check for cycles
        if (visited_nodes.has(node_index)) {
            return;
        }
        
        // Add this node to visited set
        visited_nodes.add(node_index);
        
        const node = AABB.get_node_data(node_index);
        
        // Skip free nodes
        if ((node.flags & AABB_NODE_FLAGS.FREE) != 0) {
            return;
        }
        
        // If this is a leaf node, test for intersection with the entity
        if (node.node_type === AABB_NODE_TYPE.LEAF) {
            const user_data = node.user_data;
            
            // Test intersection with the entity's geometry
            const result = this._intersect_node(ray, node_index, options);
            
            if (result && result.distance <= options.max_distance) {
                hits.push({
                    user_data,
                    distance: result.distance,
                    point: result.point,
                    normal: result.normal
                });
            }
            
            return;
        }
        
        // This is an internal node - test children
        const left_child = node.left;
        const right_child = node.right;
        
        // Test left child
        if (left_child !== 0) {
            const left_node = AABB.get_node_data(left_child);
            const left_hit = this._intersect_aabb(ray, left_node, 0, options.max_distance);
            
            if (left_hit !== false) {
                this._traverse_tree_all(ray, left_child, hits, options, visited_nodes);
            }
        }
        
        // Test right child
        if (right_child !== 0) {
            const right_node = AABB.get_node_data(right_child);
            const right_hit = this._intersect_aabb(ray, right_node, 0, options.max_distance);
            
            if (right_hit !== false) {
                this._traverse_tree_all(ray, right_child, hits, options, visited_nodes);
            }
        }
    }
    
    /**
     * Test ray intersection with an AABB (Axis-Aligned Bounding Box)
     * Uses slab method with precomputed inverse direction
     * @returns {number|false} Distance to intersection or false if no intersection
     */
    static _intersect_aabb(ray, node, t_min = 0, t_max = Infinity) {
        const min_pt = node.min_point;
        const max_pt = node.max_point;
        
        // For each axis, calculate the intersection with the slab
        for (let i = 0; i < 3; i++) {
            const t1 = (min_pt[i] - ray.origin[i]) * ray.inv_direction[i];
            const t2 = (max_pt[i] - ray.origin[i]) * ray.inv_direction[i];
            
            const t_near = Math.min(t1, t2);
            const t_far = Math.max(t1, t2);
            
            t_min = Math.max(t_min, t_near);
            t_max = Math.min(t_max, t_far);
            
            if (t_min > t_max) {
                return false;
            }
        }
        
        return t_min;
    }
    
    /**
     * Test ray intersection with a node
     * This is a simplified implementation - for a real engine you would 
     * use actual mesh geometry, but for this example we'll use node bounds
     */
    static _intersect_node(ray, node_index, options) {
        // Get the node
        const node = AABB.get_node_data(node_index);
        
        // Skip free nodes
        if ((node.flags & AABB_NODE_FLAGS.FREE) != 0) {
            return null;
        }
        
        // Simple AABB test for now
        const t = this._intersect_aabb(ray, node, 0, options.max_distance);
        
        if (t === false || t > options.max_distance) {
            return null;
        }
        
        // Calculate hit point
        const hit_point = ray.point_at(t);
        
        // Calculate normal (crude approximation from AABB)
        let normal = [0, 0, 0];
        
        if (options.return_face_normal) {
            // Determine which face was hit by finding the closest approach to a face
            const min_pt = node.min_point;
            const max_pt = node.max_point;
            
            const distances = [
                Math.abs(hit_point[0] - min_pt[0]), // -X face
                Math.abs(hit_point[0] - max_pt[0]), // +X face
                Math.abs(hit_point[1] - min_pt[1]), // -Y face
                Math.abs(hit_point[1] - max_pt[1]), // +Y face
                Math.abs(hit_point[2] - min_pt[2]), // -Z face
                Math.abs(hit_point[2] - max_pt[2])  // +Z face
            ];
            
            // Find the minimum distance
            let min_distance = distances[0];
            let min_index = 0;
            
            for (let i = 1; i < distances.length; i++) {
                if (distances[i] < min_distance) {
                    min_distance = distances[i];
                    min_index = i;
                }
            }
            
            // Create normal vector based on which face was hit
            switch (min_index) {
                case 0: normal = [-1, 0, 0]; break; // -X face
                case 1: normal = [1, 0, 0]; break;  // +X face
                case 2: normal = [0, -1, 0]; break; // -Y face
                case 3: normal = [0, 1, 0]; break;  // +Y face
                case 4: normal = [0, 0, -1]; break; // -Z face
                case 5: normal = [0, 0, 1]; break;  // +Z face
            }
        }
        
        return {
            distance: t,
            point: hit_point,
            normal
        };
    }

    /**
     * Get the AABB for a node
     * @param {number} node_index - The index of the node
     * @returns {Object|null} - The node's AABB or null if not found
     */
    static get_node_aabb(node_index) {
        if (node_index >= AABB.size) return null;
        
        const node = AABB.get_node_data(node_index);
        return {
            min: node.min_point,
            max: node.max_point
        };
    }
} 