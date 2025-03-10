import { Buffer } from "../renderer/buffer.js";
import { ComputeTaskQueue } from "../renderer/compute_task_queue.js";
import { AABB } from "./aabb.js";
import { profile_scope } from "../utility/performance.js";
import { Ray, RaycastHit } from "./aabb_raycast.js";

const MAX_RAYS = 1024;
const gpu_raycast_scope = "AABBGPURaycast";

export class AABBGPURaycast {
    static ray_buffer = null;
    static hits_buffer = null;
    static uniforms = null;
    static uniforms_buffer = null;
    static rays = [];
    static hits = [];
    static pending_callback = null;
    static is_initialized = false;
    
    /**
     * Initialize the GPU raycast system
     */
    static initialize() {
        if (this.is_initialized) return;
        
        // Create ray buffer
        this.ray_buffer = Buffer.create({
            name: "aabb_raycast_rays",
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            size: MAX_RAYS * 8 // Ray structure is 8 floats
        });
        
        // Create hits buffer
        this.hits_buffer = Buffer.create({
            name: "aabb_raycast_hits",
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            size: MAX_RAYS * 8 // Hit structure is 8 floats
        });
        
        // Create uniforms buffer
        this.uniforms_buffer = Buffer.create({
            name: "aabb_raycast_uniforms",
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            size: 4 // 4 uints (ray_count, max_distance, find_closest, max_traversal_steps)
        });

        this.uniforms = new Uint32Array([
            0,
            Infinity,
            1,
            128
        ]);
        
        // Allocate CPU-side arrays
        this.rays = new Float32Array(MAX_RAYS * 8); // Each ray is 8 floats
        this.hits = new Uint32Array(MAX_RAYS * 8); // Each hit is 8 uint32s/floats
        
        this.is_initialized = true;
    }
    
    /**
     * Submit rays for GPU raycasting
     * @param {Ray[]} rays - Array of rays to cast
     * @param {Object} options - Raycast options
     * @param {Function} callback - Callback function to receive results
     */
    static raycast_batch(rays, options = {}, callback = null) {
        profile_scope(gpu_raycast_scope, () => {
            if (!this.is_initialized) {
                this.initialize();
            }
            
            // Set default options
            const max_distance = options.max_distance || Infinity;
            const find_closest = options.find_closest !== false; // Default to true
            const max_traversal_steps = options.max_traversal_steps || 128;
            
            // Clamp ray count to MAX_RAYS
            const ray_count = Math.min(rays.length, MAX_RAYS);
            
            // Store callback for later
            this.pending_callback = callback;
            
            // Fill ray buffer
            for (let i = 0; i < ray_count; i++) {
                const ray = rays[i];
                const offset = i * 8;
                
                // origin (vec3)
                this.rays[offset] = ray.origin[0];
                this.rays[offset + 1] = ray.origin[1];
                this.rays[offset + 2] = ray.origin[2];
                
                // direction (vec3)
                this.rays[offset + 3] = ray.direction[0];
                this.rays[offset + 4] = ray.direction[1];
                this.rays[offset + 5] = ray.direction[2];
                
                // inv_direction and t values are computed in the shader
                this.rays[offset + 6] = 0.0001; // t_min: Small offset to avoid self-intersection
                this.rays[offset + 7] = max_distance; // t_max
            }
            
            // Write ray data to GPU
            this.ray_buffer.write(this.rays, 0, ray_count * 8);
            
            // Write uniform data
            this.uniforms[0] = ray_count;
            this.uniforms[1] = max_distance === Infinity ? 0x7F800000 : Math.fround(max_distance);
            this.uniforms[2] = find_closest ? 1 : 0;
            this.uniforms[3] = max_traversal_steps;
            this.uniforms_buffer.write(this.uniforms);
            
            // Submit compute task
            ComputeTaskQueue.get().new_task(
                "aabb_raycast",
                "system_compute/aabb_raycast.wgsl",
                [
                    this.uniforms_buffer,
                    AABB.data.node_data_buffer,
                    this.ray_buffer,
                    this.hits_buffer
                ],
                [this.hits_buffer],
                Math.ceil(ray_count / 64), // Work groups of 64 threads
                1,
                1,
                this._on_raycast_complete.bind(this, ray_count)
            );
        });
    }
    
    /**
     * Called when the GPU raycast is complete
     */
    static async _on_raycast_complete(ray_count) {
        if (!this.pending_callback) return;
        
        // Read hit data from GPU
        await this.hits_buffer.read(
            this.hits,
            ray_count * 8 * 4, // Size in bytes
            0, // Offset in buffer
            0, // Offset in array
            Uint32Array // Type
        );
        
        // Process the hits
        const hits = [];
        
        for (let i = 0; i < ray_count; i++) {
            const offset = i * 8;
            const user_data = this.hits[offset];
            
            // Skip hits with user_data 0 (no hit)
            if (user_data === 0) continue;
            
            // Convert Uint32 to Float32 for the floats
            const float_data = new Float32Array(this.hits.buffer, offset * 4, 8);
            
            const hit = new RaycastHit();
            hit.user_data = user_data;
            hit.distance = float_data[1];
            hit.point = [float_data[2], float_data[3], float_data[4]];
            hit.normal = [float_data[5], float_data[6], float_data[7]];
            
            hits.push(hit);
        }
        
        // Call the callback with the results
        this.pending_callback(hits);
        this.pending_callback = null;
    }
    
    /**
     * Convenience method for single ray casting
     * @param {Ray} ray - The ray to cast
     * @param {Object} options - Raycast options
     * @returns {Promise<RaycastHit[]>} - Promise that resolves to an array of hits
     */
    static raycast(ray, options = {}) {
        return new Promise((resolve) => {
            this.raycast_batch([ray], options, resolve);
        });
    }
    
    /**
     * Destroy resources
     */
    static cleanup() {
        if (this.ray_buffer) {
            this.ray_buffer.destroy();
            this.ray_buffer = null;
        }
        
        if (this.hits_buffer) {
            this.hits_buffer.destroy();
            this.hits_buffer = null;
        }
        
        if (this.uniforms_buffer) {
            this.uniforms_buffer.destroy();
            this.uniforms_buffer = null;
        }
        
        this.rays = null;
        this.hits = null;
        this.pending_callback = null;
        this.is_initialized = false;
    }
} 