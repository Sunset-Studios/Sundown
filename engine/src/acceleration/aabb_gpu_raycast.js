import { Renderer } from "../renderer/renderer.js";
import { Buffer, BufferSync } from "../renderer/buffer.js";
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
            size: MAX_RAYS * 12 // Ray structure is 12 floats
        });
        
        // Create hits buffer
        this.hits_buffer = Buffer.create({
            name: "aabb_raycast_hits",
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            size: MAX_RAYS * 8, // Hit structure is 8 floats
            cpu_readback: true,
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
        this.last_ray_count = 0;

        Renderer.get().on_post_render(this._on_post_render.bind(this));
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
            this.last_ray_count = Math.min(rays.length, MAX_RAYS);

            // Store callback for later
            this.pending_callback = callback;
            
            // Fill ray buffer
            for (let i = 0; i < this.last_ray_count; i++) {
                const ray = rays[i];
                const offset = i * 8;
                
                // origin (vec3)
                this.rays[offset] = ray.origin[0];
                this.rays[offset + 1] = ray.origin[1];
                this.rays[offset + 2] = ray.origin[2];
                this.rays[offset + 3] = 0.0001; // t_min: Small offset to avoid self-intersection
                
                // direction (vec3)
                this.rays[offset + 4] = ray.direction[0];
                this.rays[offset + 5] = ray.direction[1];
                this.rays[offset + 6] = ray.direction[2];
                this.rays[offset + 7] = max_distance; // t_max

                // inv_direction (vec3)
                this.rays[offset + 8] = ray.inv_direction[0];
                this.rays[offset + 9] = ray.inv_direction[1];
                this.rays[offset + 10] = ray.inv_direction[2];
                this.rays[offset + 11] = 0.0;
            }
            
            // Write ray data to GPU
            this.ray_buffer.write(this.rays, 0, this.last_ray_count * 12);
            
            // Write uniform data
            this.uniforms[0] = this.last_ray_count;
            this.uniforms[1] = max_distance === Infinity ? 0x7F800000 : Math.fround(max_distance);
            this.uniforms[2] = find_closest ? 1 : 0;
            this.uniforms[3] = max_traversal_steps;
            this.uniforms_buffer.write(this.uniforms);
            
            // Submit compute task
            ComputeTaskQueue.new_task(
                "aabb_raycast",
                "system_compute/aabb_raycast.wgsl",
                [
                    this.uniforms_buffer,
                    AABB.node_bounds_buffer,
                    AABB.node_data_buffer,
                    this.ray_buffer,
                    this.hits_buffer
                ],
                [this.hits_buffer],
                Math.ceil(this.last_ray_count / 64), // Work groups of 64 threads
            );
        });
    }

    /**
     * Called when the GPU raycast is complete
     */
    static async _on_post_render() {
        BufferSync.request_readback(this);
    }

    static async readback_buffers() {
        // Read hit data from GPU
        await this.hits_buffer.read(
            this.hits,
            this.last_ray_count * 8 * 4,
            0,
            0,
            Uint32Array
        );
        
        if (!this.pending_callback) return;

        // Process the hits
        const hits = [];
        
        for (let i = 0; i < this.last_ray_count; i++) {
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
    static raycast(ray, options = {}, callback = null) {
        this.raycast_batch([ray], options, callback);
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