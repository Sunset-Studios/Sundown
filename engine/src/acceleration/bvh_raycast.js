import { BVH } from "./bvh.js";
import { EntityManager } from "../core/ecs/entity.js";
import { TransformFragment } from "../core/ecs/fragments/transform_fragment.js";
import { ComputeTaskQueue } from "../renderer/task_queues/compute_task_queue.js";
import { Buffer } from "../renderer/buffer.js";
import { RandomAccessAllocator, RingBufferAllocator } from "../memory/allocator.js";
import { MeshBLAS } from "./mesh_blas.js";
import { MeshData } from "../renderer/mesh_data.js";
import { FragmentGpuBuffer } from "../core/ecs/solar/memory.js";
import { TransformProcessor } from "../core/subsystems/transform_processor.js";

const EPSILON = 0.0001;
const bounds_name = "bounds";

export const RayHitMode = {
  BLAS: 0,
  TLAS: 1,
};

export class Ray {
  constructor(origin, direction, t_min = 0.0, t_max = Infinity, user_data = 0) {
    this.t_min = t_min;
    this.t_max = t_max;
    this.user_data = user_data;
    this.hit_mode = RayHitMode.BLAS;
    this.index = 0;
    this.setup(origin, direction);
  }

  setup(origin, direction) {
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
      direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]
    );
    const one_over_dir_length = 1.0 / dir_length;

    if (dir_length < EPSILON) {
      this.direction = [0, 0, 0];
      this.inv_direction = [0, 0, 0];
    } else {
      this.direction = [
        direction[0] * one_over_dir_length,
        direction[1] * one_over_dir_length,
        direction[2] * one_over_dir_length,
      ];

      // Pre-compute inverse direction for bounding box tests
      this.inv_direction = [
        Math.abs(this.direction[0]) < EPSILON ? Infinity : 1.0 / this.direction[0],
        Math.abs(this.direction[1]) < EPSILON ? Infinity : 1.0 / this.direction[1],
        Math.abs(this.direction[2]) < EPSILON ? Infinity : 1.0 / this.direction[2],
      ];
    }
  }

  position_at(t) {
    return [
      this.origin[0] + this.direction[0] * t,
      this.origin[1] + this.direction[1] * t,
      this.origin[2] + this.direction[2] * t,
    ];
  }
}

export class RaycastHit {
  constructor() {
    this.user_data = 0;
    this.distance = Infinity;
    this.position = [0, 0, 0];
    this.normal = [0, 0, 0];
  }

  reset() {
    this.user_data = 0;
    this.distance = Infinity;
    this.position = [0, 0, 0];
    this.normal = [0, 0, 0];
  }
}

/**
 * Static helper for BVH ray-casting.
 * Systems enqueue rays during the frame via `request_ray`, then call `flush`
 * once per-frame to run the compute pass. Results can be queried through
 * `gather_results`.
 */
export class BVHRaycast {
  // -----------------------------------------------------------------------
  // Static state
  // -----------------------------------------------------------------------
  static pending_rays = new RandomAccessAllocator(256, Ray);
  static cpu_hit_result_allocator = new RingBufferAllocator(256, RaycastHit);
  static rays_buffer = null;
  static hits_buffer = null;
  static max_rays = 0;
  static last_dispatch_count = 0;
  static ray_data = new Float32Array(12);
  static last_hit_results = new Float32Array(8);

  // -----------------------------------------------------------------------
  // Public API
  // -----------------------------------------------------------------------

  /**
   * Allocate and queue a ray for intersection testing.
   * @param {Ray} ray - The ray instance to test against the scene BVH.
   * @returns {number} Index into the results array for this ray.
   */
  static request_ray(hit_mode = RayHitMode.BLAS) {
    const ray_index = this.pending_rays.length;
    const ray = this.pending_rays.allocate();
    ray.index = ray_index;
    ray.hit_mode = hit_mode;
    return ray;
  }

  /**
   * Dispatch all queued rays. Should be invoked once per-frame after all
   * systems have queued their raycasts.
   * @param {any} bvh_buffers - GPU buffers returned from BVH.get_gpu_data().
   */
  static flush() {
    this.last_dispatch_count = this.pending_rays.length;
    if (this.pending_rays.length === 0) {
      return;
    }

    if (this.pending_rays.length > this.max_rays) {
      this.max_rays = this.pending_rays.length;
      this._recreate_buffers();
    }

    // -------------------------------------------------------------------
    // Upload ray packet
    // Layout matches Ray struct in acceleration_common.wgsl
    // -------------------------------------------------------------------
    for (let i = 0; i < this.pending_rays.length; i++) {
      const r = this.pending_rays.get(i);
      const off = i * 12;

      // origin.xyz | t_min
      this.ray_data[off + 0] = r.origin[0];
      this.ray_data[off + 1] = r.origin[1];
      this.ray_data[off + 2] = r.origin[2];
      this.ray_data[off + 3] = r.t_min ?? 0.0;

      // direction.xyz | t_max
      this.ray_data[off + 4] = r.direction[0];
      this.ray_data[off + 5] = r.direction[1];
      this.ray_data[off + 6] = r.direction[2];
      this.ray_data[off + 7] = r.t_max ?? Infinity;

      // inv_direction.xyz | hit mode
      this.ray_data[off + 8] = r.inv_direction[0];
      this.ray_data[off + 9] = r.inv_direction[1];
      this.ray_data[off + 10] = r.inv_direction[2];
      this.ray_data[off + 11] = r.hit_mode ?? RayHitMode.BLAS;
    }
    this.rays_buffer.write_raw(this.ray_data);

    // -------------------------------------------------------------------
    // GPU dispatch
    // -------------------------------------------------------------------
    const bvh_buffers = BVH.to_gpu_data();

    const bounds_gpu = EntityManager.get_fragment_gpu_buffer(
      TransformFragment,
      bounds_name
    );

    const compact_transforms = TransformProcessor.get_compact_transforms_buffer();

    const blas_gpu_data = MeshBLAS.to_gpu_data();
    const index_buffer = MeshData.index_buffer;
    const entity_index_map_buffer = FragmentGpuBuffer.entity_index_map_buffer;

    ComputeTaskQueue.new_task(
      "bvh_raycast",
      "acceleration/bvh_traversal.wgsl",
      [
        bvh_buffers.bvh_info_buffer,
        this.rays_buffer,
        this.hits_buffer,
        bounds_gpu.buffer,
        compact_transforms,
        blas_gpu_data.bvh2_nodes_buffer,
        blas_gpu_data.directory_buffer,
        index_buffer,
        entity_index_map_buffer.buffer,
      ],
      [this.hits_buffer],
      Math.ceil(this.pending_rays.length / 128), 1, 1,
      "traverse_tlas_bvh"
    );

    // Clear the queue for the next frame
    this.pending_rays.reset();
  }

  /**
   * Read back the results buffer for the rays dispatched in the last call
   * to `flush`.
   * @returns {Promise<any[]>} Promise resolving to an array of hit records.
   */
  static async gather_results() {
    if (!this.hits_buffer || this.last_dispatch_count === 0) {
      return [];
    }
    await this.hits_buffer.read(this.last_hit_results, this.last_dispatch_count * 8 * 4);
  }

  static get_hit_result(ray) {
    const base = ray.index * 8;
    const hit = this.cpu_hit_result_allocator.allocate();
    hit.position[0] = this.last_hit_results[base + 0];
    hit.position[1] = this.last_hit_results[base + 1];
    hit.position[2] = this.last_hit_results[base + 2];
    hit.distance = this.last_hit_results[base + 3];
    hit.normal[0] = this.last_hit_results[base + 4];
    hit.normal[1] = this.last_hit_results[base + 5];
    hit.normal[2] = this.last_hit_results[base + 6];
    hit.user_data = this.last_hit_results[base + 7];
    return hit;
  }

  // -------------------------------------------------------------------
  // Internal helpers
  // -------------------------------------------------------------------
  static _recreate_buffers() {
    this.rays_buffer = Buffer.create({
      name: "raycast_rays_buffer",
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      size: this.max_rays * 12,
      force: true,
    });

    this.hits_buffer = Buffer.create({
      name: "raycast_hits_buffer",
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      size: this.max_rays * 8,
      force: true,
      cpu_readback: true,
    });

    // Re-allocate the ray data buffer
    this.ray_data = new Float32Array(this.max_rays * 12);
    this.last_hit_results = new Float32Array(this.max_rays * 8);
  }

  static destroy() {
    this.rays_buffer?.destroy();
    this.hits_buffer?.destroy();
    this.pending_rays = null;
    this.max_rays = 0;
    this.last_dispatch_count = 0;
  }
}
